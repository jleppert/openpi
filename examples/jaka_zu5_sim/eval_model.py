"""Detailed evaluation of a trained RL model for the JAKA Zu5 pick-cube task.

Runs many episodes and collects per-step diagnostics to understand failure modes.
Use --save-videos N to record the first N episodes as MP4 files.
"""

import dataclasses
import logging
import pathlib
import sys

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gym_env  # noqa: E402


@dataclasses.dataclass
class EpisodeStats:
    success: bool = False
    total_steps: int = 0
    total_reward: float = 0.0
    # Cube initial position
    cube_init_x: float = 0.0
    cube_init_y: float = 0.0
    # Distances
    min_dist: float = float("inf")
    final_dist: float = 0.0
    # Termination
    terminated_by: str = "truncated"  # "success", "grasp_fail", "truncated"
    # Per-step traces (sampled)
    dist_trace: list = dataclasses.field(default_factory=list)
    action_trace: list = dataclasses.field(default_factory=list)


def _add_overlay(frame, step, info):
    """Burn a simple text overlay into the frame (no extra dependencies)."""
    h, w = frame.shape[:2]
    # Black bar at top
    frame[:20, :] = 0
    dist = info["dist"]
    # Green bar proportional to closeness (closer = wider bar)
    closeness = max(0.0, 1.0 - dist / 0.3)
    bar_w = min(int(closeness * w), w)
    if info.get("is_success", False):
        frame[:20, :bar_w] = [0, 200, 0]
    else:
        frame[:20, :bar_w] = [200, 200, 0]
    return frame


def run_eval(model_path: str, n_episodes: int = 200, seed: int = 42,
             save_videos: int = 0, video_dir: str = "data/jaka_zu5_sim/eval_videos",
             verbose: bool = True):
    logging.info("Loading model from %s", model_path)

    # Use render_mode if we need videos.
    render_mode = "rgb_array" if save_videos > 0 else None
    env = gym_env.JakaZu5PickCubeEnv(render_mode=render_mode)

    model = SAC.load(model_path)
    rng = np.random.default_rng(seed)

    if save_videos > 0:
        import imageio
        video_path = pathlib.Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        logging.info("Will save %d episode videos to %s", save_videos, video_path)

    episodes: list[EpisodeStats] = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        stats = EpisodeStats()

        # Record initial cube position from flat observation.
        # obs = [gripper_x, gripper_y, cube_x, cube_y]
        stats.cube_init_x = float(obs[2])
        stats.cube_init_y = float(obs[3])

        recording = save_videos > 0 and ep < save_videos
        frames = []

        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            stats.total_steps += 1
            stats.total_reward += reward

            dist = info["dist"]
            stats.min_dist = min(stats.min_dist, dist)
            stats.final_dist = dist

            # Record video frame (skip the terminating step — grasp frames cover it).
            if recording and not terminated:
                frame = env.render()
                if frame is not None:
                    frame = _add_overlay(frame.copy(), step, info)
                    frames.append(frame)

            # Sample traces every 5 steps
            if step % 5 == 0:
                stats.dist_trace.append(dist)
                stats.action_trace.append(action.copy())

            if terminated:
                # Collect frames rendered during the scripted grasp sequence.
                if recording and hasattr(env, '_grasp_frames'):
                    frames.extend(env._grasp_frames)
                if info.get("is_success", False):
                    stats.success = True
                    stats.terminated_by = "success"
                else:
                    # Grasp triggered but failed (scripted grasp returned False)
                    stats.terminated_by = "grasp_fail"
                break

            if truncated:
                stats.terminated_by = "truncated"
                break

        episodes.append(stats)

        # Save video
        if recording and frames:
            label = stats.terminated_by
            fname = video_path / f"ep{ep:03d}_{label}_r{stats.total_reward:.0f}.mp4"
            imageio.mimwrite(str(fname), frames, fps=15, quality=8)
            logging.info("  Saved %s (%d frames)", fname.name, len(frames))

        if verbose and (ep + 1) % 50 == 0:
            recent = episodes[-50:]
            sr = sum(1 for e in recent if e.success) / len(recent)
            logging.info("  Episode %d/%d — recent 50 success rate: %.0f%%", ep + 1, n_episodes, sr * 100)

    env.close()
    return episodes


def analyze(episodes: list[EpisodeStats]):
    n = len(episodes)
    successes = [e for e in episodes if e.success]
    failures = [e for e in episodes if not e.success]
    grasp_fails = [e for e in episodes if e.terminated_by == "grasp_fail"]
    truncated = [e for e in episodes if e.terminated_by == "truncated"]

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {n} episodes")
    print(f"{'='*60}")

    print(f"\n## Overall")
    print(f"  Success rate:    {len(successes)}/{n} = {100*len(successes)/n:.1f}%")
    print(f"  Grasp fail rate: {len(grasp_fails)}/{n} = {100*len(grasp_fails)/n:.1f}%")
    print(f"  Timeout rate:    {len(truncated)}/{n} = {100*len(truncated)/n:.1f}%")
    print(f"  Mean reward:     {np.mean([e.total_reward for e in episodes]):.1f}")

    # Success analysis
    if successes:
        print(f"\n## Successful Episodes ({len(successes)})")
        steps = [e.total_steps for e in successes]
        rewards = [e.total_reward for e in successes]
        print(f"  Steps to reach:  mean={np.mean(steps):.0f}, min={min(steps)}, max={max(steps)}, std={np.std(steps):.0f}")
        print(f"  Reward:          mean={np.mean(rewards):.1f}, min={min(rewards):.1f}, max={max(rewards):.1f}")
        print(f"  Min XY distance: mean={np.mean([e.min_dist for e in successes]):.4f}")

    # Failure analysis
    if failures:
        print(f"\n## Failed Episodes ({len(failures)})")
        print(f"  Grasp triggered but failed: {len(grasp_fails)}")
        print(f"  Never reached cube (timeout): {len(truncated)}")

        if truncated:
            print(f"\n  Timeout episodes:")
            print(f"    Min distance:  mean={np.mean([e.min_dist for e in truncated]):.3f}, min={min(e.min_dist for e in truncated):.3f}")
            print(f"    Final distance: mean={np.mean([e.final_dist for e in truncated]):.3f}")

    # Spatial analysis — cube position vs outcome
    print(f"\n## Spatial Analysis (cube_x, cube_y → success)")
    print(f"  Cube X range: [{min(e.cube_init_x for e in episodes):.2f}, {max(e.cube_init_x for e in episodes):.2f}]")
    print(f"  Cube Y range: [{min(e.cube_init_y for e in episodes):.2f}, {max(e.cube_init_y for e in episodes):.2f}]")

    # Grid analysis
    x_bins = np.linspace(0.62, 0.78, 5)
    y_bins = np.linspace(-0.08, 0.08, 5)
    print(f"\n  Success rate by cube position (X=columns, Y=rows):")
    print(f"  {'':>8s}", end="")
    for j in range(len(x_bins) - 1):
        print(f"  x[{x_bins[j]:.2f},{x_bins[j+1]:.2f}]", end="")
    print()

    for i in range(len(y_bins) - 1):
        label = f"y[{y_bins[i]:+.2f},{y_bins[i+1]:+.2f}]"
        print(f"  {label:>20s}", end="")
        for j in range(len(x_bins) - 1):
            in_cell = [e for e in episodes
                       if x_bins[j] <= e.cube_init_x < x_bins[j+1]
                       and y_bins[i] <= e.cube_init_y < y_bins[i+1]]
            if in_cell:
                sr = sum(1 for e in in_cell if e.success) / len(in_cell)
                print(f"  {100*sr:5.0f}% ({len(in_cell):2d})", end="")
            else:
                print(f"    --- (0)", end="")
        print()

    # Timing distribution for successes
    if successes:
        print(f"\n## Success Timing Distribution")
        bins = [0, 5, 10, 15, 25, 50, 100]
        for i in range(len(bins) - 1):
            count = sum(1 for e in successes if bins[i] <= e.total_steps < bins[i+1])
            bar = "#" * min(count, 60)
            print(f"    {bins[i]:3d}-{bins[i+1]:3d} steps: {count:3d} {bar}")

    # Distance profile for failed episodes
    if truncated:
        print(f"\n## Average Distance Profile (timed-out episodes, n={len(truncated)})")
        max_trace_len = max(len(e.dist_trace) for e in truncated)
        for t_idx in range(0, min(max_trace_len, 40), 2):  # every 10 steps
            dists = [e.dist_trace[t_idx] for e in truncated if t_idx < len(e.dist_trace)]
            step = t_idx * 5
            n_alive = len(dists)
            print(f"    Step {step:3d}: dist={np.mean(dists):.3f} +/- {np.std(dists):.3f}, n={n_alive}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/jaka_zu5_sim/rl/best_success_model/best_success_model")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-videos", type=int, default=0, help="Save first N episodes as MP4 videos")
    parser.add_argument("--video-dir", type=str, default="data/jaka_zu5_sim/eval_videos", help="Directory for video output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    episodes = run_eval(args.model, n_episodes=args.episodes, seed=args.seed,
                        save_videos=args.save_videos, video_dir=args.video_dir)
    analyze(episodes)
