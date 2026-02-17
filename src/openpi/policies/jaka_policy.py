"""Transforms for the JAKA Zu5 robot with 4D action space.

The JAKA Zu5 has 6 joints but only 3 are independently controlled (J1-J3).
The dataset stores 4D actions [J1_vel, J2_vel, J3_vel, gripper] and 3D joint
positions [J1, J2, J3] natively, so no dimension selection is needed.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class JakaInputs(transforms.DataTransformFn):
    """Pack JAKA observation into 4D state + 4D actions for pi0-FAST."""

    def __call__(self, data: dict) -> dict:
        # 4D state: [J1_pos, J2_pos, J3_pos, gripper_pos]
        joint_pos = np.asarray(data["observation/joint_position"])
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([joint_pos, gripper_pos])

        # Images: use DROID camera key names for pretrained weight compatibility.
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        images = {
            "base_0_rgb": base_image,
            "base_1_rgb": np.zeros_like(base_image),
            "wrist_0_rgb": wrist_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "base_1_rgb": np.True_,
            "wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Only forward actions with training shape (horizon, dim).  During
        # inference the eval script sends a 1-D dummy array so that
        # RepackTransform doesn't error, but we must NOT include it — the
        # FAST tokenizer would encode it into the prompt and prevent the
        # model from generating actions autoregressively.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim == 2:
                inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class JakaOutputs(transforms.DataTransformFn):
    """Pass through 4D actions — eval script handles expansion to 6-DOF."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
