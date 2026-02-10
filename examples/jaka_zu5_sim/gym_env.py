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
_MAX_EPISODE_STEPS = 100

# Table surface height (from MJCF: table body z=0.35, top half-height=0.02).
_TABLE_Z = 0.37

# Cube randomization bounds (x, y on table surface).
_CUBE_X_RANGE = (0.62, 0.78)
_CUBE_Y_RANGE = (-0.08, 0.08)

# HER distance threshold — cube must be within this of desired_goal to count as success.
_HER_DIST_THRESHOLD = 0.05


class JakaZu5PickCubeEnv(gymnasium.Env):
    """Pick-cube GoalEnv for HER (Hindsight Experience Replay).

    Observation dict:
        'observation': (22,) — joint_pos(6), joint_vel(6), gripper(1),
                                gripper_xyz(3), cube_xyz(3), rel_xyz(3)
        'achieved_goal': (3,) — cube current xyz
        'desired_goal': (3,) — target xyz for cube (randomized)
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


        # Spaces — GoalEnv-style dict observation for HER.
        # observation (22D): joint_pos(6), joint_vel(6), gripper(1), gripper_xyz(3), cube_xyz(3), rel_xyz(3)
        self.observation_space = gymnasium.spaces.Dict({
            "observation": gymnasium.spaces.Box(-np.inf, np.inf, (22,), np.float32),
            "achieved_goal": gymnasium.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "desired_goal": gymnasium.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })
        # 4D: J1_vel, J2_vel, J3_vel, gripper_cmd
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._step_count = 0
        self._grasp_steps = 0
        self._gripper_state = _GRIPPER_MAX_OPEN  # start open
        self._desired_goal = np.zeros(3, dtype=np.float32)
        self._rng = np.random.default_rng()

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

        # Randomize goal: 50% on table (push), 50% in air (lift).
        # Ensure minimum distance from cube start so goals aren't trivially solved.
        cube_start = np.array([cube_x, cube_y, cube_z], dtype=np.float32)
        for _ in range(100):  # rejection sampling
            goal_x = cube_x + self._rng.uniform(-0.10, 0.10)
            goal_y = cube_y + self._rng.uniform(-0.10, 0.10)
            if self._rng.random() < 0.5:
                goal_z = _TABLE_Z + 0.03  # table level (push target)
            else:
                goal_z = self._rng.uniform(_TABLE_Z + 0.03, _TABLE_Z + 0.15)
            goal = np.array([goal_x, goal_y, goal_z], dtype=np.float32)
            if np.linalg.norm(goal - cube_start) > _HER_DIST_THRESHOLD:
                break
        self._desired_goal = goal

        # Set arm home pose: gripper near workspace, pointing down.
        # J5 = -(J2+J3) keeps the gripper pointing straight down.
        j2_home, j3_home = -1.48, 1.8
        home = [0.0, j2_home, j3_home, 0.0, -(j2_home + j3_home), 0.0]
        for i, (jid, val) in enumerate(zip(self._arm_joint_ids, home)):
            if i in _CONTROLLED_JOINTS:
                val += self._rng.uniform(-0.05, 0.05)
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._arm_actuator_ids[i]] = self._data.qpos[self._model.jnt_qposadr[jid]]

        # Start gripper open — use _GRIPPER_MAX_OPEN consistently.
        self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]] = _GRIPPER_MAX_OPEN
        self._data.ctrl[self._gripper_actuator_id] = _GRIPPER_MAX_OPEN

        mujoco.mj_forward(self._model, self._data)

        # Persistent commanded positions — avoids feedback drift from reading qpos.
        self._cmd_pos = np.array([
            self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids
        ])

        self._step_count = 0
        self._grasp_steps = 0
        self._gripper_state = _GRIPPER_MAX_OPEN

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
        # Clip J5 to its physical limits to prevent command windup.
        raw_j5 = -(self._cmd_pos[1] + self._cmd_pos[2])
        j5_id = self._arm_joint_ids[4]
        j5_lo = self._model.jnt_range[j5_id, 0]
        j5_hi = self._model.jnt_range[j5_id, 1]
        self._cmd_pos[3] = 0.0
        self._cmd_pos[4] = np.clip(raw_j5, j5_lo, j5_hi)
        self._cmd_pos[5] = 0.0

        # Apply actuator controls.
        for i, aid in enumerate(self._arm_actuator_ids):
            self._data.ctrl[aid] = self._cmd_pos[i]

        # Gripper: map [-1,1] to [open, closed] with dead zone to prevent chattering.
        # Only change state when action is decisive (|cmd| > 0.2).
        if gripper_cmd > 0.2:
            self._gripper_state = 0.0  # close
        elif gripper_cmd < -0.2:
            self._gripper_state = _GRIPPER_MAX_OPEN  # open
        # else: hold previous state (dead zone)
        self._data.ctrl[self._gripper_actuator_id] = self._gripper_state

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
        """GoalEnv dict observation (22D)."""
        joint_pos = np.array([
            self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids
        ], dtype=np.float32)

        joint_vel = np.array([
            self._data.qvel[self._model.jnt_dofadr[jid]] for jid in self._arm_joint_ids
        ], dtype=np.float32)

        gripper_opening = np.array([
            self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]] / _GRIPPER_MAX_OPEN
        ], dtype=np.float32)

        gripper_xyz = self._data.xpos[self._gripper_body_id].astype(np.float32)
        cube_xyz = self._data.xpos[self._cube_body_id].astype(np.float32)
        rel_xyz = cube_xyz - gripper_xyz

        obs = np.concatenate([joint_pos, joint_vel, gripper_opening, gripper_xyz, cube_xyz, rel_xyz])  # (22,)
        return {
            "observation": obs,
            "achieved_goal": cube_xyz.copy(),
            "desired_goal": self._desired_goal.copy(),
        }

    def compute_reward(self, achieved_goal, desired_goal, _info):
        """Sparse reward for HER. Called on batches by HerReplayBuffer."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > _HER_DIST_THRESHOLD).astype(np.float32)

    def _compute_reward(self):
        cube_pos = self._data.xpos[self._cube_body_id]
        gripper_pos = self._data.xpos[self._gripper_body_id]
        achieved = cube_pos.astype(np.float32)

        # Sparse reward: 0 if cube within threshold of goal, else -1.
        reward = float(self.compute_reward(achieved[None], self._desired_goal[None], {})[0])

        # Success = cube within threshold of desired goal.
        is_success = np.linalg.norm(achieved - self._desired_goal) < _HER_DIST_THRESHOLD

        # Keep grasp tracking for diagnostics only.
        has_grasp = self._check_grasp()
        if has_grasp:
            self._grasp_steps += 1

        # Terminate on cube drop (off table).
        terminated = False
        if cube_pos[2] < _TABLE_Z - 0.05:
            terminated = True

        # NO success termination — fixed-length episodes work best with HER
        # (HER "future" strategy needs future states to relabel).

        info = {
            "dist": float(np.linalg.norm(gripper_pos - cube_pos)),
            "cube_z": float(cube_pos[2]),
            "has_grasp": has_grasp,
            "grasp_steps": self._grasp_steps,
            "is_success": bool(is_success),
            "success": bool(is_success),
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
            del self._renderer
            self._renderer = None
