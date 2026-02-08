"""MuJoCo environment for JAKA Zu5 simulation with DROID-format observations."""

import pathlib

import mujoco
import mujoco.renderer
import mujoco.viewer
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

# Joint names in the MJCF model (6-DOF arm + 1 gripper).
_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT_NAME = "finger_left"

# Actuator names matching the MJCF.
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_ACTUATOR_NAME = "act_gripper"

# Gripper range from MJCF.
_GRIPPER_MAX_OPEN = 0.04

# Image resolution expected by DROID policy.
_IMAGE_SIZE = 224

# Physics sub-steps per control step (~15 Hz control with 0.002s timestep).
_PHYSICS_STEPS_PER_CONTROL = 33


class JakaZu5SimEnvironment(_environment.Environment):
    """JAKA Zu5 simulation environment producing DROID-format observations."""

    def __init__(self, prompt: str = "pick up the red cube", display: bool = False) -> None:
        self._prompt = prompt
        self._model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._step_count = 0
        self._viewer = None
        self._display = display

        # Cache joint/actuator indices.
        self._arm_joint_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES]
        self._gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT_NAME)
        self._arm_actuator_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES]
        self._gripper_actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME)

        # Camera IDs.
        self._external_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "external_cam")
        self._wrist_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    @override
    def reset(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        # Set a reasonable home position (slightly raised elbow).
        home = [0.0, 0.5, -1.0, 0.0, 0.5, 0.0]
        for jid, val in zip(self._arm_joint_ids, home):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0

        if self._display and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    @override
    def is_episode_complete(self) -> bool:
        # Run for a fixed number of steps (40s at 15 Hz = 600 steps).
        return self._step_count >= 600

    @override
    def get_observation(self) -> dict:
        ext_img = self._render_camera(self._external_cam_id)
        wrist_img = self._render_camera(self._wrist_cam_id)

        # Resize to 224x224 and ensure uint8 HWC.
        ext_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(ext_img, _IMAGE_SIZE, _IMAGE_SIZE))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, _IMAGE_SIZE, _IMAGE_SIZE))

        # 6-DOF JAKA joints → pad to 7 to match DROID/Franka format.
        joint_pos = np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids])
        joint_pos_7 = np.concatenate([joint_pos, [0.0]])

        # Gripper position normalized to [0, 1].
        gripper_raw = self._data.qpos[self._model.jnt_qposadr[self._gripper_joint_id]]
        gripper_pos = np.array([gripper_raw / _GRIPPER_MAX_OPEN])

        return {
            "observation/exterior_image_1_left": ext_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos_7.astype(np.float32),
            "observation/gripper_position": gripper_pos.astype(np.float32),
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action["actions"])  # shape [8]

        # actions[:6] → joint velocity deltas for JAKA joints.
        # actions[6] → ignored (padded 7th DOF).
        # actions[7] → gripper command.
        joint_vel = actions[:6]
        gripper_cmd = actions[7]

        dt = self._model.opt.timestep * _PHYSICS_STEPS_PER_CONTROL

        # Integrate velocity deltas into position targets.
        current_pos = np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._arm_joint_ids])
        target_pos = np.clip(
            current_pos + joint_vel * dt,
            [self._model.jnt_range[jid, 0] for jid in self._arm_joint_ids],
            [self._model.jnt_range[jid, 1] for jid in self._arm_joint_ids],
        )

        # Set arm actuator controls.
        for aid, pos in zip(self._arm_actuator_ids, target_pos):
            self._data.ctrl[aid] = pos

        # Binarize gripper: >0.5 → open, else closed.
        self._data.ctrl[self._gripper_actuator_id] = _GRIPPER_MAX_OPEN if gripper_cmd > 0.5 else 0.0

        # Step physics.
        for _ in range(_PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self._model, self._data)

        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1

    def _render_camera(self, cam_id: int) -> np.ndarray:
        """Render a camera and return HWC uint8 RGB image."""
        self._renderer.update_scene(self._data, camera=cam_id)
        return self._renderer.render()
