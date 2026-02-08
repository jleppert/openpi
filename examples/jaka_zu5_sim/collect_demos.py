"""Collect demonstrations from a trained RL policy and save as a LeRobot dataset."""

import dataclasses
import logging
import pathlib
import shutil

import mujoco
import mujoco.renderer
import numpy as np
from openpi_client import image_tools
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm
import tyro

import gym_env

_IMAGE_SIZE = 224
_REPO_NAME = "levelhq/jaka_zu5_pick_cube"


@dataclasses.dataclass
class Args:
    # Path to trained RL model (e.g., data/jaka_zu5_sim/rl/best_model/best_model.zip).
    model_path: str = "data/jaka_zu5_sim/rl/best_model/best_model.zip"

    # Number of episodes to collect.
    n_episodes: int = 500

    # Only save successful episodes.
    only_successful: bool = True

    # Language instruction for the task.
    prompt: str = "pick up the red cube"

    # Dataset repo name.
    repo_id: str = _REPO_NAME

    # Push to HuggingFace Hub.
    push_to_hub: bool = False


def main(args: Args) -> None:
    from stable_baselines3 import PPO

    logging.info("Loading RL model from %s", args.model_path)
    model = PPO.load(args.model_path)

    # Create a single env for rendering.
    env = gym_env.JakaZu5PickCubeEnv()

    # Set up a renderer for camera images.
    mj_model = env._model
    mj_data = env._data
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    external_cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "external_cam")
    wrist_cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    # Clean up any existing dataset.
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="jaka_zu5",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (_IMAGE_SIZE, _IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (_IMAGE_SIZE, _IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    saved_count = 0
    attempted = 0
    pbar = tqdm(total=args.n_episodes, desc="Collecting demos")

    while saved_count < args.n_episodes:
        attempted += 1
        obs, _ = env.reset()
        episode_frames = []
        episode_success = False

        for step_idx in range(200):  # max episode steps
            # Get RL action (4D: J1_vel, J2_vel, J3_vel, gripper).
            rl_action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(rl_action)

            # Render cameras for the dataset.
            renderer.update_scene(mj_data, camera=external_cam_id)
            ext_img = renderer.render().copy()
            ext_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(ext_img, _IMAGE_SIZE, _IMAGE_SIZE)
            )

            renderer.update_scene(mj_data, camera=wrist_cam_id)
            wrist_img = renderer.render().copy()
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, _IMAGE_SIZE, _IMAGE_SIZE)
            )

            # Read joint positions (6D, padded to 7D).
            joint_pos = np.array([
                mj_data.qpos[mj_model.jnt_qposadr[env._arm_joint_ids[i]]]
                for i in range(6)
            ], dtype=np.float32)
            joint_pos_7 = np.concatenate([joint_pos, [0.0]]).astype(np.float32)

            # Gripper position normalized [0, 1].
            gripper_raw = mj_data.qpos[mj_model.jnt_qposadr[env._gripper_joint_id]]
            gripper_pos = np.array([gripper_raw / gym_env._GRIPPER_MAX_OPEN], dtype=np.float32)

            # Map 4D RL action to 8D DROID format.
            # RL: [J1_vel, J2_vel, J3_vel, gripper_cmd]
            # DROID: [J1_vel, J2_vel, J3_vel, J4_vel(0), J5_vel(0), J6_vel(0), J7_vel(0), gripper]
            droid_action = np.zeros(8, dtype=np.float32)
            droid_action[0] = rl_action[0]
            droid_action[1] = rl_action[1]
            droid_action[2] = rl_action[2]
            # dims 3-6 are zero (locked wrist + padded 7th DOF)
            droid_action[7] = 1.0 if rl_action[3] <= 0.0 else 0.0  # invert: RL >0=close → DROID >0.5=open

            episode_frames.append({
                "exterior_image_1_left": ext_img,
                "wrist_image_left": wrist_img,
                "joint_position": joint_pos_7,
                "gripper_position": gripper_pos,
                "actions": droid_action,
                "task": args.prompt,
            })

            obs = next_obs

            if info.get("success", False):
                episode_success = True

            if terminated or truncated:
                break

        # Save episode if successful (or if we're saving all).
        if episode_success or not args.only_successful:
            for frame in episode_frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            saved_count += 1
            pbar.update(1)
            pbar.set_postfix(
                attempted=attempted,
                success_rate=f"{saved_count / attempted:.1%}",
            )

    pbar.close()
    logging.info(
        "Collected %d successful episodes out of %d attempts (%.1f%% success rate)",
        saved_count, attempted, 100.0 * saved_count / attempted,
    )
    logging.info("Dataset saved to %s", HF_LEROBOT_HOME / args.repo_id)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["jaka_zu5", "pick_cube", "simulation"],
            private=False,
            push_videos=True,
        )

    renderer.close()
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    tyro.cli(main)
