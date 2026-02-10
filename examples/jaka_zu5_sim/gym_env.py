"""Gymnasium environment for RL training on the JAKA Zu5 pick-cube task.

UR5-style approach: RL learns XY positioning only (2D action/obs).
When the gripper is close enough, a scripted sequence descends, grasps, and lifts.
"""

import pathlib

import gymnasium
import mujoco
import numpy as np

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

# Arm joint names and indices used for control.
_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT_NAME = "finger_left"
_GRIPPER_ACTUATOR_NAME = "act_gripper"
_GRIPPER_MAX_OPEN = 0.04

# Top-down approach: only J1-J3 are controlled by IK.
# J4 and J6 are locked at 0. J5 = -(J2+J3) keeps gripper pointing down.
_CONTROLLED_JOINTS = [0, 1, 2]  # indices into the 6 arm joints

# Physics.
_PHYSICS_STEPS_PER_CONTROL = 33  # ~15 Hz control at 0.002s timestep
_MAX_EPISODE_STEPS = 100

# Table surface height (from MJCF: table body z=0.35, top half-height=0.02).
_TABLE_Z = 0.37

# Cube randomization bounds (x, y on table surface).
_CUBE_X_RANGE = (0.55, 0.82)
_CUBE_Y_RANGE = (-0.35, 0.35)

# Workspace bounds for action/observation space.
_WS_X_RANGE = (0.45, 0.85)
_WS_Y_RANGE = (-0.40, 0.40)

# Scripted grasp constants (gripper_base Z; fingers are ~0.055m below).
# Finger tips extend 0.095m below gripper_base; table at 0.37.
_APPROACH_Z = 0.52    # gripper_base height during XY approach (fingers above cube)
_GRASP_Z = 0.475      # gripper_base height for grasping (finger tips just above table)
_LIFT_Z = 0.62        # target gripper_base height for lift
_LIFT_THRESHOLD = 0.50  # cube Z must exceed this to count as success
_REACH_THRESHOLD = 0.02  # XY distance to trigger scripted grasp

# IK solver parameters.
_IK_MAX_ITER = 20
_IK_DAMPING = 1e-2
_IK_TOL = 1e-3


class JakaZu5PickCubeEnv(gymnasium.Env):
    """Pick-cube env with XY positioning (RL) + scripted grasp.

    Action (2D): absolute target [x, y] within workspace bounds.
    Observation (2D): cube [x, y] on table.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super().__init__()

        self._model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self._data = mujoco.MjData(self._model)
        self._render_mode = render_mode
        self._renderer = None

        # Cache IDs.
        self._arm_joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES
        ]
        self._arm_actuator_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES
        ]
        self._gripper_joint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT_NAME
        )
        self._gripper_actuator_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME
        )

        self._cube_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._cube_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        self._gripper_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

        # Geom IDs for contact detection.
        self._left_finger_geom = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom")
        self._right_finger_geom = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom")
        self._cube_geom = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom")

        # Spaces — 2D action, 4D observation.
        self.action_space = gymnasium.spaces.Box(
            low=np.array([_WS_X_RANGE[0], _WS_Y_RANGE[0]], dtype=np.float32),
            high=np.array([_WS_X_RANGE[1], _WS_Y_RANGE[1]], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        obs_low = np.array([_WS_X_RANGE[0], _WS_Y_RANGE[0], _WS_X_RANGE[0], _WS_Y_RANGE[0]], dtype=np.float32)
        obs_high = np.array([_WS_X_RANGE[1], _WS_Y_RANGE[1], _WS_X_RANGE[1], _WS_Y_RANGE[1]], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=obs_low, high=obs_high, shape=(4,), dtype=np.float32,
        )

        self._step_count = 0
        self._rng = np.random.default_rng()
        self._grasp_frames = []  # populated during scripted grasp if render_mode set

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        # Randomize cube position on table.
        cube_x = self._rng.uniform(*_CUBE_X_RANGE)
        cube_y = self._rng.uniform(*_CUBE_Y_RANGE)
        cube_z = _TABLE_Z + 0.03  # half cube size above table
        cube_qpos_adr = self._model.jnt_qposadr[self._cube_joint_id]
        self._data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = [cube_x, cube_y, cube_z]
        self._data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]  # identity quat

        # Randomize arm starting position: pick a random XY in workspace, IK to reach it.
        # First set a nominal home pose so IK has a reasonable starting config.
        j2_home, j3_home = -1.48, 1.8
        home = [0.0, j2_home, j3_home, 0.0, -(j2_home + j3_home), 0.0]
        for i, (jid, val) in enumerate(zip(self._arm_joint_ids, home)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._arm_actuator_ids[i]] = val
        mujoco.mj_forward(self._model, self._data)

        # Solve IK to a random workspace position.
        start_x = self._rng.uniform(*_WS_X_RANGE)
        start_y = self._rng.uniform(*_WS_Y_RANGE)
        start_target = np.array([start_x, start_y, _APPROACH_Z])
        joint_targets = self._solve_ik(start_target)
        for i, (jid, val) in enumerate(zip(self._arm_joint_ids, joint_targets)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._arm_actuator_ids[i]] = val

        # Start gripper open.
        self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]] = _GRIPPER_MAX_OPEN
        self._data.ctrl[self._gripper_actuator_id] = _GRIPPER_MAX_OPEN

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos = np.array([action[0], action[1], _APPROACH_Z])

        # IK solve → joint targets.
        joint_targets = self._solve_ik(target_pos)
        for i, aid in enumerate(self._arm_actuator_ids):
            self._data.ctrl[aid] = joint_targets[i]
        self._data.ctrl[self._gripper_actuator_id] = _GRIPPER_MAX_OPEN  # keep open

        # Step physics.
        for _ in range(_PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self._model, self._data)

        # Measure XY distance between gripper and cube.
        gripper_xy = self._data.xpos[self._gripper_body_id][:2]
        cube_xy = self._data.xpos[self._cube_body_id][:2]
        distance = np.linalg.norm(gripper_xy - cube_xy)

        if distance <= _REACH_THRESHOLD:
            # Scripted grasp sequence.
            success = self._scripted_grasp_and_lift()
            steps_bonus = max(0, _MAX_EPISODE_STEPS - self._step_count)
            reward = 100.0 + float(steps_bonus) if success else -10.0
            done = True
        elif self._step_count >= _MAX_EPISODE_STEPS:
            reward = -10.0 * distance
            done = True
        else:
            reward = -10.0 * distance
            done = False

        obs = self._get_obs()
        info = {"dist": float(distance), "is_success": done and distance <= _REACH_THRESHOLD and reward > 0}
        return obs, reward, done, False, info

    def _get_obs(self):
        """Flat 4D observation: [gripper_x, gripper_y, cube_x, cube_y]."""
        gripper_xy = self._data.xpos[self._gripper_body_id][:2].astype(np.float32)
        cube_xy = self._data.xpos[self._cube_body_id][:2].astype(np.float32)
        return np.concatenate([gripper_xy, cube_xy])

    def _solve_ik(self, target_pos):
        """Jacobian-based IK for arm joints to reach target_pos with gripper_base.

        Returns array of 6 joint targets.
        """
        # Save state.
        qpos_saved = self._data.qpos.copy()
        qvel_saved = self._data.qvel.copy()

        nv = self._model.nv
        jacp = np.zeros((3, nv))

        for _ in range(_IK_MAX_ITER):
            mujoco.mj_forward(self._model, self._data)
            current_pos = self._data.xpos[self._gripper_body_id].copy()
            error = target_pos - current_pos
            if np.linalg.norm(error) < _IK_TOL:
                break

            # Compute Jacobian for gripper_base body.
            jacp[:] = 0
            mujoco.mj_jacBody(self._model, self._data, jacp, None, self._gripper_body_id)

            # Extract sub-Jacobian for J1, J2, J3 DOFs.
            dof_indices = [self._model.jnt_dofadr[self._arm_joint_ids[j]] for j in _CONTROLLED_JOINTS]
            J = jacp[:, dof_indices]  # (3, 3)

            # Damped least squares.
            JJT = J @ J.T + (_IK_DAMPING ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)

            # Update controlled joints (clamped to limits).
            for idx, ctrl_j in enumerate(_CONTROLLED_JOINTS):
                jid = self._arm_joint_ids[ctrl_j]
                adr = self._model.jnt_qposadr[jid]
                new_val = self._data.qpos[adr] + dq[idx]
                lo = self._model.jnt_range[jid, 0]
                hi = self._model.jnt_range[jid, 1]
                self._data.qpos[adr] = np.clip(new_val, lo, hi)

            # Enforce wrist constraints: J4=0, J5=-(J2+J3), J6=0.
            j2_val = self._data.qpos[self._model.jnt_qposadr[self._arm_joint_ids[1]]]
            j3_val = self._data.qpos[self._model.jnt_qposadr[self._arm_joint_ids[2]]]
            raw_j5 = -(j2_val + j3_val)
            j5_id = self._arm_joint_ids[4]
            j5_lo, j5_hi = self._model.jnt_range[j5_id]
            self._data.qpos[self._model.jnt_qposadr[self._arm_joint_ids[3]]] = 0.0
            self._data.qpos[self._model.jnt_qposadr[j5_id]] = np.clip(raw_j5, j5_lo, j5_hi)
            self._data.qpos[self._model.jnt_qposadr[self._arm_joint_ids[5]]] = 0.0

        # Record solution.
        joint_targets = np.array([
            self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids
        ])

        # Restore state.
        self._data.qpos[:] = qpos_saved
        self._data.qvel[:] = qvel_saved
        mujoco.mj_forward(self._model, self._data)

        return joint_targets

    def _scripted_grasp_and_lift(self):
        """Execute scripted descend → close → lift sequence. Returns success bool."""
        # Use the cube's XY (not gripper's) to ensure we descend right on target.
        cube_xy = self._data.xpos[self._cube_body_id][:2].copy()
        self._grasp_frames = []

        # 1. Descend to grasp height.
        descend_target = np.array([cube_xy[0], cube_xy[1], _GRASP_Z])
        joint_targets = self._solve_ik(descend_target)
        for i, aid in enumerate(self._arm_actuator_ids):
            self._data.ctrl[aid] = joint_targets[i]
        self._data.ctrl[self._gripper_actuator_id] = _GRIPPER_MAX_OPEN
        for s in range(300):
            mujoco.mj_step(self._model, self._data)
            if s % 33 == 0:
                self._maybe_capture_frame()

        # 2. Close gripper.
        self._data.ctrl[self._gripper_actuator_id] = 0.0
        for s in range(500):
            mujoco.mj_step(self._model, self._data)
            if s % 33 == 0:
                self._maybe_capture_frame()

        # 3. Check grasp.
        if not self._check_grasp():
            return False

        # 4. Lift gradually.
        n_lift_steps = 15
        for k in range(1, n_lift_steps + 1):
            z = _GRASP_Z + (_LIFT_Z - _GRASP_Z) * k / n_lift_steps
            lift_target = np.array([cube_xy[0], cube_xy[1], z])
            joint_targets = self._solve_ik(lift_target)
            for i, aid in enumerate(self._arm_actuator_ids):
                self._data.ctrl[aid] = joint_targets[i]
            self._data.ctrl[self._gripper_actuator_id] = 0.0  # keep closed
            for s in range(150):
                mujoco.mj_step(self._model, self._data)
            self._maybe_capture_frame()

        # 5. Verify lift.
        cube_z = self._data.xpos[self._cube_body_id][2]
        return bool(cube_z > _LIFT_THRESHOLD)

    def _maybe_capture_frame(self):
        """Capture a render frame if render_mode is set."""
        if self._render_mode == "rgb_array":
            frame = self.render()
            if frame is not None:
                self._grasp_frames.append(frame.copy())

    def _check_grasp(self):
        """Check if both fingers are in contact with the cube."""
        left_contact = False
        right_contact = False
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == self._left_finger_geom and g2 == self._cube_geom) or \
               (g2 == self._left_finger_geom and g1 == self._cube_geom):
                left_contact = True
            if (g1 == self._right_finger_geom and g2 == self._cube_geom) or \
               (g2 == self._right_finger_geom and g1 == self._cube_geom):
                right_contact = True
        return left_contact and right_contact

    def render(self):
        if self._render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "external_cam")
        self._renderer.update_scene(self._data, camera=cam_id)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
