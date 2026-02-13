"""Validate a collected LeRobot dataset.

Checks data shapes, value ranges, episode structure, and action statistics.
Optionally saves sample images for visual inspection.

Usage:
    # Basic validation
    uv run python examples/jaka_zu5_sim/test_dataset.py

    # Save sample images
    uv run python examples/jaka_zu5_sim/test_dataset.py \
        --args.save-samples data/jaka_zu5_sim/dataset_samples

    # Custom dataset
    uv run python examples/jaka_zu5_sim/test_dataset.py \
        --args.repo-id my_org/my_dataset
"""

import dataclasses
import logging
import pathlib

import numpy as np
from PIL import Image
import tyro


@dataclasses.dataclass
class Args:
    """Arguments for dataset validation."""

    # LeRobot dataset repo ID.
    repo_id: str = "levelhq/jaka_zu5_pick_cube"

    # Number of sample frames to check.
    n_samples: int = 50

    # Directory to save sample images (skip if not set).
    save_samples: str | None = None


def _to_numpy(val):
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(val, "numpy"):
        return val.numpy()
    return val


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    logging.info("Loading dataset: %s", args.repo_id)
    dataset = LeRobotDataset(args.repo_id)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {args.repo_id}")
    print(f"{'=' * 60}")
    print(f"  Episodes:     {dataset.num_episodes}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  FPS:          {dataset.fps}")
    print(f"  Features:     {list(dataset.features.keys())}")

    if len(dataset) == 0:
        print("\n  Dataset is empty! Nothing to validate.")
        return

    # Episode lengths.
    try:
        ep_lens = []
        for ep_idx in range(dataset.num_episodes):
            start = dataset.episode_data_index["from"][ep_idx].item()
            end = dataset.episode_data_index["to"][ep_idx].item()
            ep_lens.append(end - start)
        print(f"\n  Episode lengths:")
        print(f"    min={min(ep_lens)}, max={max(ep_lens)}, "
              f"mean={np.mean(ep_lens):.1f}, std={np.std(ep_lens):.1f}")
    except Exception as e:
        print(f"\n  Could not compute episode lengths: {e}")
        ep_lens = []

    # Sample random frames.
    n = min(args.n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    issues = []

    # Load sample frames.
    sample_frames = [dataset[int(i)] for i in indices]

    # Check each feature.
    feature_keys = [k for k in sample_frames[0].keys() if k not in ("task_index", "index", "episode_index", "frame_index", "timestamp")]
    for key in sorted(feature_keys):
        vals = [_to_numpy(f[key]) for f in sample_frames if key in f]
        if not vals:
            continue
        v0 = vals[0]
        if not isinstance(v0, np.ndarray):
            continue

        print(f"\n  {key}:")
        print(f"    shape: {v0.shape}")
        print(f"    dtype: {v0.dtype}")

        stacked = np.stack(vals)
        lo, hi = stacked.min(), stacked.max()
        mean = stacked.mean()
        print(f"    range: [{lo}, {hi}]")
        print(f"    mean:  {mean:.4f}")

        # Image checks.
        if "image" in key:
            if v0.ndim != 3:
                issues.append(f"{key}: expected 3D array, got {v0.ndim}D")
            if lo == hi:
                issues.append(f"{key}: constant image (all pixels = {lo})")
            if v0.ndim == 3 and v0.shape[0] == 3:
                print(f"    format: CHW (channels first)")
            elif v0.ndim == 3 and v0.shape[2] == 3:
                print(f"    format: HWC (channels last)")

        # Gripper checks.
        elif key == "gripper_position":
            if lo < -0.1 or hi > 1.1:
                issues.append(f"{key}: values outside [0,1]: [{lo:.3f}, {hi:.3f}]")

        # Action checks.
        elif key == "actions":
            if np.any(np.isnan(stacked)):
                issues.append(f"{key}: contains NaN values")
            if np.any(np.isinf(stacked)):
                issues.append(f"{key}: contains Inf values")
            print(f"    per-dimension stats (over {n} samples):")
            dim_labels = [
                "J1_vel", "J2_vel", "J3_vel", "J4_vel",
                "J5_vel", "J6_vel", "J7_pad", "gripper",
            ]
            for d in range(min(8, stacked.shape[-1])):
                dm = stacked[..., d]
                label = dim_labels[d] if d < len(dim_labels) else f"dim{d}"
                print(f"      {label:8s}: [{dm.min():+8.3f}, {dm.max():+8.3f}] "
                      f"mean={dm.mean():+7.3f} std={dm.std():.3f}")

        # Joint position checks.
        elif key == "joint_position":
            if np.any(np.isnan(stacked)):
                issues.append(f"{key}: contains NaN values")

    # Save sample images.
    if args.save_samples:
        out_dir = pathlib.Path(args.save_samples)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_save = min(5, len(sample_frames))
        for i in range(n_save):
            frame = sample_frames[i]
            for img_key in ["exterior_image_1_left", "wrist_image_left"]:
                if img_key not in frame:
                    continue
                img = _to_numpy(frame[img_key])
                # Handle CHW → HWC.
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                # Handle float [0,1] → uint8.
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                path = out_dir / f"sample_{i}_{img_key}.png"
                Image.fromarray(img).save(path)
        print(f"\n  Sample images saved to {out_dir}/")

    # Summary.
    print(f"\n{'=' * 60}")
    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  All checks passed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
