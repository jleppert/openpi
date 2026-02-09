"""Gymnasium environment for RL training on the JAKA Zu5 pick-cube task."""

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

# Top-down approach: only J1-J3 are controlled by the RL agent.
# J4 and J6 are locked at 0. J5 is computed as -(J2+J3) each step so the
# gripper always points straight down regardless of shoulder/elbow angles.
_CONTROLLED_JOINTS = [0, 1, 2]  # indices into the 6 arm joints

# Physics.
_PHYSICS_STEPS_PER_CONTROL = 33  # ~15 Hz control at 0.002s timestep
_MAX_EPISODE_STEPS = 200

# Table surface height (from MJCF: table body z=0.35, top half-height=0.02).
_TABLE_Z = 0.37

# Cube randomization bounds (x, y on table surface).
_CUBE_X_RANGE = (0.50, 0.70)
_CUBE_Y_RANGE = (-0.10, 0.10)

# Lift success threshold (10cm above table).
_LIFT_THRESHOLD = _TABLE_Z + 0.10
_HOLD_STEPS_REQUIRED = 8  # ~0.5 second at 15 Hz


class JakaZu5PickCubeEnv(gymnasium.Env):
    """Pick-cube task for RL training with top-down grasp constraint.

    Observation (11D): [joint_pos(6), gripper_opening(1), cube_xyz(3), gripper_to_cube_dist(1)]
    Action (4D): [J1_vel, J2_vel, J3_vel, gripper_cmd]
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


        # Spaces.
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        # 4D: J1_vel, J2_vel, J3_vel, gripper_cmd
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._step_count = 0
        self._hold_count = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        # Randomize cube position on table.
        cube_x = self._rng.uniform(*_CUBE_X_RANGE)
        cube_y = self._rng.uniform(*_CUBE_Y_RANGE)
        cube_z = _TABLE_Z + 0.025  # half cube size above table
        cube_qpos_adr = self._model.jnt_qposadr[self._cube_joint_id]
        self._data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = [cube_x, cube_y, cube_z]
        self._data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]  # identity quat

        # Set arm home pose: gripper near workspace, pointing down.
        # J5 = -(J2+J3) keeps the gripper pointing straight down.
        j2_home, j3_home = -1.48, 1.8
        home = [0.0, j2_home, j3_home, 0.0, -(j2_home + j3_home), 0.0]
        for i, (jid, val) in enumerate(zip(self._arm_joint_ids, home)):
            if i in _CONTROLLED_JOINTS:
                val += self._rng.uniform(-0.05, 0.05)
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._arm_actuator_ids[i]] = self._data.qpos[self._model.jnt_qposadr[jid]]

        # Start gripper half-open.
        self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]] = 0.02
        self._data.ctrl[self._gripper_actuator_id] = 0.02

        mujoco.mj_forward(self._model, self._data)

        # Persistent commanded positions — avoids feedback drift from reading qpos.
        self._cmd_pos = np.array([
            self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids
        ])

        self._step_count = 0
        self._hold_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Velocity scaling for controlled joints.
        vel_scale = 3.0
        joint_vel = action[:3] * vel_scale
        gripper_cmd = action[3]

        dt = self._model.opt.timestep * _PHYSICS_STEPS_PER_CONTROL

        # Update commanded positions (persistent, avoids feedback drift).
        for idx, ctrl_j in enumerate(_CONTROLLED_JOINTS):
            new_val = self._cmd_pos[ctrl_j] + joint_vel[idx] * dt
            lo = self._model.jnt_range[self._arm_joint_ids[ctrl_j], 0]
            hi = self._model.jnt_range[self._arm_joint_ids[ctrl_j], 1]
            self._cmd_pos[ctrl_j] = np.clip(new_val, lo, hi)

        # Lock wrist: J4=0, J5=-(J2+J3) for downward gripper, J6=0.
        self._cmd_pos[3] = 0.0
        self._cmd_pos[4] = -(self._cmd_pos[1] + self._cmd_pos[2])
        self._cmd_pos[5] = 0.0

        # Apply actuator controls.
        for i, aid in enumerate(self._arm_actuator_ids):
            self._data.ctrl[aid] = self._cmd_pos[i]

        # Gripper: map [-1,1] to [open, closed]. >0 = close, <0 = open.
        gripper_target = 0.0 if gripper_cmd > 0.0 else _GRIPPER_MAX_OPEN
        self._data.ctrl[self._gripper_actuator_id] = gripper_target

        # Step physics.
        for _ in range(_PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # Compute reward and check termination.
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()
        truncated = self._step_count >= _MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """11D observation: [joint_pos(6), gripper(1), cube_xyz(3), dist(1)]."""
        joint_pos = np.array([
            self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids
        ], dtype=np.float32)

        gripper_opening = np.array([
            self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]] / _GRIPPER_MAX_OPEN
        ], dtype=np.float32)

        cube_pos = self._data.xpos[self._cube_body_id].astype(np.float32)
        gripper_pos = self._data.xpos[self._gripper_body_id].astype(np.float32)
        dist = np.array([np.linalg.norm(gripper_pos - cube_pos)], dtype=np.float32)

        return np.concatenate([joint_pos, gripper_opening, cube_pos, dist])

    def _compute_reward(self):
        gripper_pos = self._data.xpos[self._gripper_body_id]
        cube_pos = self._data.xpos[self._cube_body_id]
        dist = np.linalg.norm(gripper_pos - cube_pos)

        cube_z = cube_pos[2]

        # Check finger-cube contacts.
        has_grasp = self._check_grasp()

        # 1. Reach reward: negative distance.
        r_reach = -2.0 * dist

        # 2. Grasp reward.
        r_grasp = 2.0 if has_grasp else 0.0

        # 3. Lift reward (only if grasped) — strong signal to lift high.
        lift_height = max(0.0, cube_z - _TABLE_Z)
        r_lift = 15.0 * lift_height if has_grasp else 0.0

        # 4. Height bonus: extra reward for reaching near/above threshold.
        r_height_bonus = 0.0
        if has_grasp and cube_z >= _LIFT_THRESHOLD - 0.03:
            r_height_bonus = 3.0
        if has_grasp and cube_z >= _LIFT_THRESHOLD:
            r_height_bonus = 5.0

        # 5. Success check.
        terminated = False
        success = False
        if cube_z >= _LIFT_THRESHOLD and has_grasp:
            self._hold_count += 1
            if self._hold_count >= _HOLD_STEPS_REQUIRED:
                success = True
                terminated = True
        else:
            self._hold_count = 0

        r_success = 20.0 if success else 0.0

        # 6. Penalty if cube falls off table.
        r_penalty = 0.0
        if cube_z < _TABLE_Z - 0.05:
            r_penalty = -5.0
            terminated = True

        reward = r_reach + r_grasp + r_lift + r_height_bonus + r_success + r_penalty

        info = {
            "dist": dist,
            "cube_z": cube_z,
            "has_grasp": has_grasp,
            "hold_count": self._hold_count,
            "success": success,
            "is_success": success,  # SB3 EvalCallback tracks this automatically
        }
        return reward, terminated, info

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
            self._renderer.close()
            self._renderer = None
