"""Validate a collected LeRobot dataset by re-rendering MuJoCo scenes.

Loads sample frames from the dataset, sets the robot to the recorded joint
positions, renders both cameras, and produces side-by-side comparison images
(dataset image vs. re-rendered image). Also prints action statistics to flag
zero-variance dimensions or other issues.

Usage:
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/validate_dataset.py \
        --dataset-dir data/jaka_zu5_sim/datasets/jaka_zu5_pick_cube_v2

    # Sample more episodes / frames
    MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/validate_dataset.py \
        --dataset-dir data/jaka_zu5_sim/datasets/jaka_zu5_pick_cube_v2 \
        --n-episodes 10 --frames-per-episode 5
"""

import argparse
import io
import json
import pathlib
import sys

import mujoco
import mujoco.renderer
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import camera_config as cam_cfg

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT = "finger_left"
_GRIPPER_MAX_OPEN = 0.04


def load_parquet(path: pathlib.Path):
    """Load a parquet file and return as a list of row dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return table.to_pydict()


def decode_png(blob: bytes) -> np.ndarray:
    """Decode a PNG byte blob to a numpy RGB array."""
    img = Image.open(io.BytesIO(blob))
    return np.asarray(img.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/jaka_zu5_sim/datasets/jaka_zu5_pick_cube_v2",
    )
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes to sample")
    parser.add_argument(
        "--frames-per-episode", type=int, default=3, help="Frames per episode"
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_output/dataset_validation"
    )
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info.
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    print(f"Dataset: {dataset_dir}")
    print(f"  Episodes: {info['total_episodes']}")
    print(f"  Frames:   {info['total_frames']}")
    print(f"  FPS:      {info['fps']}")
    print(f"  Features: {list(info['features'].keys())}")

    # Set up MuJoCo.
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    data = mujoco.MjData(model)

    ext_cam = cam_cfg.default_external()
    wrist_cam = cam_cfg.default_wrist()
    ext_id = cam_cfg.apply_camera_config(model, ext_cam)
    wrist_id = cam_cfg.apply_camera_config(model, wrist_cam)

    ext_rend = mujoco.Renderer(model, height=ext_cam.render_height, width=ext_cam.render_width)
    wrist_rend = mujoco.Renderer(model, height=wrist_cam.render_height, width=wrist_cam.render_width)

    # Cache joint IDs.
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES]
    grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT)

    # Find parquet files.
    data_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    total_episodes = len(parquet_files)
    print(f"\n  Parquet files found: {total_episodes}")

    # Select episodes to sample (spread evenly).
    n_episodes = min(args.n_episodes, total_episodes)
    if n_episodes <= 0:
        print("No episodes to validate.")
        return
    ep_indices = np.linspace(0, total_episodes - 1, n_episodes, dtype=int)
    print(f"  Sampling episodes: {ep_indices.tolist()}")

    # Collect action stats across ALL episodes.
    print("\n--- Action Statistics (across all episodes) ---")
    all_actions = []
    all_joint_positions = []
    all_gripper_positions = []
    for pf in parquet_files:
        cols = load_parquet(pf)
        actions = np.array(cols["actions"])
        all_actions.append(actions)
        all_joint_positions.append(np.array(cols["joint_position"]))
        all_gripper_positions.append(np.array(cols["gripper_position"]))

    all_actions = np.concatenate(all_actions, axis=0)
    all_joint_positions = np.concatenate(all_joint_positions, axis=0)
    all_gripper_positions = np.concatenate(all_gripper_positions, axis=0)

    print(f"\n  Total frames: {len(all_actions)}")
    print(f"\n  Action dims ({all_actions.shape[1]}D):")
    print(f"  {'Dim':>4} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Q01':>10} {'Q99':>10} {'Zero%':>8}")
    for d in range(all_actions.shape[1]):
        col = all_actions[:, d]
        zero_pct = 100.0 * np.sum(np.abs(col) < 1e-8) / len(col)
        print(
            f"  {d:>4} {col.mean():>10.4f} {col.std():>10.4f} "
            f"{col.min():>10.4f} {col.max():>10.4f} "
            f"{np.percentile(col, 1):>10.4f} {np.percentile(col, 99):>10.4f} "
            f"{zero_pct:>7.1f}%"
        )

    # Flag zero-variance dims.
    zero_var_dims = [d for d in range(all_actions.shape[1]) if all_actions[:, d].std() < 1e-8]
    if zero_var_dims:
        print(f"\n  WARNING: Zero-variance action dimensions: {zero_var_dims}")
        print("  These will cause pi0-FAST mode collapse (see docs/zero_variance_actions.md)")
    else:
        print("\n  OK: No zero-variance action dimensions")

    print(f"\n  Joint positions ({all_joint_positions.shape[1]}D):")
    print(f"  {'Dim':>4} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for d in range(all_joint_positions.shape[1]):
        col = all_joint_positions[:, d]
        print(f"  {d:>4} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f}")

    print(f"\n  Gripper position:")
    gp = all_gripper_positions.flatten()
    print(f"    Mean={gp.mean():.4f}  Std={gp.std():.4f}  Min={gp.min():.4f}  Max={gp.max():.4f}")

    # Render side-by-side comparisons.
    print("\n--- Rendering Comparisons ---")
    from openpi_client import image_tools

    for ep_idx in ep_indices:
        pf = parquet_files[ep_idx]
        cols = load_parquet(pf)
        n_frames = len(cols["joint_position"])

        # Select frames: first, middle, last.
        n_sample = min(args.frames_per_episode, n_frames)
        frame_indices = np.linspace(0, n_frames - 1, n_sample, dtype=int)

        print(f"\n  Episode {ep_idx} ({n_frames} frames), sampling frames: {frame_indices.tolist()}")

        for fi in frame_indices:
            # Read recorded data.
            joint_pos = np.array(cols["joint_position"][fi], dtype=np.float64)
            gp_raw = cols["gripper_position"][fi]
            grip_pos = float(gp_raw[0]) if isinstance(gp_raw, (list, np.ndarray)) else float(gp_raw)

            # Decode stored images.
            ext_blob = cols["exterior_image_1_left"][fi]
            wrist_blob = cols["wrist_image_left"][fi]
            # Handle dict format (LeRobot stores as {'bytes': ..., 'path': ...}).
            if isinstance(ext_blob, dict):
                ext_blob = ext_blob["bytes"]
            if isinstance(wrist_blob, dict):
                wrist_blob = wrist_blob["bytes"]
            stored_ext = decode_png(ext_blob)
            stored_wrist = decode_png(wrist_blob)

            # Set MuJoCo state to recorded joint positions.
            mujoco.mj_resetData(model, data)
            for i, jid in enumerate(jids):
                if i < len(joint_pos):
                    data.qpos[model.jnt_qposadr[jid]] = joint_pos[i]
            # Set gripper (grip_pos is normalized [0,1]).
            data.qpos[model.jnt_qposadr[grip_jid]] = grip_pos * _GRIPPER_MAX_OPEN
            # Mirror finger.
            rfid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_right")
            data.qpos[model.jnt_qposadr[rfid]] = grip_pos * _GRIPPER_MAX_OPEN

            # We don't know the cube position, so place it at a visible spot.
            cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
            adr = model.jnt_qposadr[cube_jid]
            data.qpos[adr: adr + 3] = [0.70, 0.0, 0.400]
            data.qpos[adr + 3: adr + 7] = [1, 0, 0, 0]

            mujoco.mj_forward(model, data)

            # Render both cameras.
            ext_rend.update_scene(data, camera=ext_id)
            rendered_ext_raw = ext_rend.render().copy()
            rendered_ext = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rendered_ext_raw, ext_cam.output_size, ext_cam.output_size)
            )

            wrist_rend.update_scene(data, camera=wrist_id)
            rendered_wrist_raw = wrist_rend.render().copy()
            rendered_wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rendered_wrist_raw, wrist_cam.output_size, wrist_cam.output_size)
            )

            # Build side-by-side comparison: [stored_ext | rendered_ext | stored_wrist | rendered_wrist]
            h = stored_ext.shape[0]
            w = stored_ext.shape[1]
            gap = 4  # pixels between images
            label_h = 20  # space for labels

            canvas = np.ones((h + label_h, w * 4 + gap * 3, 3), dtype=np.uint8) * 40

            # Place images.
            canvas[label_h: label_h + h, 0:w] = stored_ext
            canvas[label_h: label_h + h, w + gap: 2 * w + gap] = rendered_ext
            canvas[label_h: label_h + h, 2 * w + 2 * gap: 3 * w + 2 * gap] = stored_wrist
            canvas[label_h: label_h + h, 3 * w + 3 * gap: 4 * w + 3 * gap] = rendered_wrist

            # Pixel difference stats (external camera only, ignoring cube position difference).
            diff = np.abs(stored_ext.astype(float) - rendered_ext.astype(float))
            mean_diff = diff.mean()
            max_diff = diff.max()

            # Save as PNG.
            out_path = output_dir / f"ep{ep_idx:04d}_frame{fi:04d}.png"
            Image.fromarray(canvas).save(str(out_path))

            print(
                f"    Frame {fi}: joints={joint_pos[:3].tolist()} grip={grip_pos:.3f} "
                f"ext_diff(mean={mean_diff:.1f}, max={max_diff:.0f}) -> {out_path}"
            )

    # Summary.
    print(f"\n--- Validation Complete ---")
    print(f"  Comparison images saved to: {output_dir}")
    print(f"  Episodes sampled: {n_episodes}")
    print(f"  Total images: {n_episodes * args.frames_per_episode}")
    if zero_var_dims:
        print(f"  ACTION WARNING: Zero-variance dims {zero_var_dims} will break pi0-FAST")


if __name__ == "__main__":
    main()
