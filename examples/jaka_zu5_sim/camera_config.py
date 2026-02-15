"""Camera configuration for JAKA Zu5 simulation.

Customizable camera parameters for matching real-world camera setups.
Save/load configs as JSON for reproducibility across runs.

Example JSON format (save with --args.save-camera-config cameras.json):

{
  "external": {
    "name": "external_cam",
    "pos": [1.2, -0.8, 1.2],
    "xyaxes": [0.6, 0.8, 0.0, -0.35, 0.25, 0.9],
    "fov": 55.0,
    "render_width": 640,
    "render_height": 480,
    "output_size": 224
  },
  "wrist": {
    "name": "wrist_cam",
    "pos": [0.0, 0.0, -0.08],
    "xyaxes": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "fov": 60.0,
    "render_width": 640,
    "render_height": 480,
    "output_size": 224
  }
}

Camera orientation uses MuJoCo's xyaxes convention:
  xyaxes = [x_axis(3), y_axis(3)]
  Z axis is computed as cross(x, y).

For body-mounted cameras (e.g. wrist_cam on gripper_base), pos is relative
to the parent body. For world-frame cameras (e.g. external_cam), pos is absolute.
"""

import dataclasses
import json
import pathlib

import mujoco
import numpy as np


@dataclasses.dataclass
class CameraConfig:
    """Configuration for a single MuJoCo camera.

    Attributes:
        name: Camera name in the MJCF model.
        pos: Position [x, y, z]. Body-relative for mounted cameras,
             absolute for world-frame cameras.
        xyaxes: Orientation [x_axis(3), y_axis(3)], MuJoCo convention.
                Z axis = cross(x, y).
        fov: Vertical field of view in degrees.
        render_width: Render width (pixels) before resize.
        render_height: Render height (pixels) before resize.
        output_size: Final square image size after resize-with-pad.
    """

    name: str
    pos: list[float]
    xyaxes: list[float]
    fov: float
    render_width: int = 640
    render_height: int = 480
    output_size: int = 224


def xyaxes_to_quat(xyaxes: list[float]) -> np.ndarray:
    """Convert MuJoCo xyaxes [x(3), y(3)] to quaternion [w, x, y, z]."""
    x = np.asarray(xyaxes[:3], dtype=np.float64)
    y = np.asarray(xyaxes[3:], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)  # re-orthogonalize
    R = np.column_stack([x, y, z])
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def apply_camera_config(model: mujoco.MjModel, cfg: CameraConfig) -> int:
    """Patch a MuJoCo model's camera parameters. Returns camera ID."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.name)
    if cid < 0:
        raise ValueError(f"Camera '{cfg.name}' not found in model")
    model.cam_pos[cid] = cfg.pos
    model.cam_fovy[cid] = cfg.fov
    model.cam_quat[cid] = xyaxes_to_quat(cfg.xyaxes)
    return cid


def default_external() -> CameraConfig:
    """External camera defaults (matches jaka_zu5.xml)."""
    return CameraConfig(
        name="external_cam",
        pos=[1.2, -0.8, 1.2],
        xyaxes=[0.6, 0.8, 0.0, -0.35, 0.25, 0.9],
        fov=55.0,
    )


def default_wrist() -> CameraConfig:
    """Wrist camera defaults (matches jaka_zu5.xml). Pos relative to gripper_base."""
    return CameraConfig(
        name="wrist_cam",
        pos=[0.0, 0.0, -0.08],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        fov=60.0,
    )


def save_configs(cfgs: dict[str, CameraConfig], path: str | pathlib.Path) -> None:
    """Save camera configs to a JSON file."""
    data = {k: dataclasses.asdict(v) for k, v in cfgs.items()}
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


def load_configs(path: str | pathlib.Path) -> dict[str, CameraConfig]:
    """Load camera configs from a JSON file."""
    raw = json.loads(pathlib.Path(path).read_text())
    return {k: CameraConfig(**v) for k, v in raw.items()}
