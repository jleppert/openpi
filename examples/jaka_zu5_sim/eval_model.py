"""Detailed evaluation of a trained RL model for the JAKA Zu5 pick-cube task.

Runs many episodes and collects per-step diagnostics to understand failure modes.
"""

import dataclasses
import logging
import pathlib
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gym_env  # noqa: E402

# Cube starts at TABLE_Z + 0.03 = 0.40. Actual lifting = cube_z - initial_z.
_TABLE_Z = 0.37
_CUBE_INIT_Z = _TABLE_Z + 0.03  # 0.40
_LIFT_THRESHOLD = _TABLE_Z + 0.10  # 0.47


@dataclasses.dataclass
class EpisodeStats:
    success: bool = False
    total_steps: int = 0
    total_reward: float = 0.0
    # Cube initial position
    cube_init_x: float = 0.0
    cube_init_y: float = 0.0
    # Timing
    first_grasp_step: int | None = None
    total_grasp_steps: int = 0
    first_real_lift_step: int | None = None  # cube > 2cm above INITIAL pos
    max_lift_height: float = 0.0  # above initial pos, not table
    max_hold_count: int = 0
    # Distances
    min_dist: float = float("inf")
    final_dist: float = 0.0
    # Termination
    terminated_by: str = "truncated"  # "success", "drop", "truncated"
    # Cube position at end
    final_cube_z: float = 0.0
    # Actions
    gripper_close_steps: int = 0
    # Per-step traces (sampled)
    dist_trace: list = dataclasses.field(default_factory=list)
    cube_z_trace: list = dataclasses.field(default_factory=list)
    grasp_trace: list = dataclasses.field(default_factory=list)
    action_trace: list = dataclasses.field(default_factory=list)


def run_eval(model_path: str, n_episodes: int = 200, seed: int = 42, verbose: bool = True):
    logging.info("Loading model from %s", model_path)
    model = PPO.load(model_path)

    env = gym_env.JakaZu5PickCubeEnv()
    rng = np.random.default_rng(seed)

    episodes: list[EpisodeStats] = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        stats = EpisodeStats()

        # Record initial cube position from observation
        # obs layout: [joint_pos(6), gripper(1), cube_xyz(3), dist(1)]
        stats.cube_init_x = float(obs[7])
        stats.cube_init_y = float(obs[8])
        initial_cube_z = float(obs[9])

        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            stats.total_steps += 1
            stats.total_reward += reward

            dist = info["dist"]
            cube_z = info["cube_z"]
            has_grasp = info["has_grasp"]
            hold_count = info["hold_count"]

            stats.min_dist = min(stats.min_dist, dist)
            stats.final_dist = dist
            stats.final_cube_z = cube_z
            stats.max_hold_count = max(stats.max_hold_count, hold_count)

            # Track gripper actions
            if action[3] > 0:  # close
                stats.gripper_close_steps += 1

            # Track grasp timing
            if has_grasp:
                stats.total_grasp_steps += 1
                if stats.first_grasp_step is None:
                    stats.first_grasp_step = step

            # Track real lift (above initial cube position)
            real_lift = max(0.0, cube_z - initial_cube_z)
            if real_lift > stats.max_lift_height:
                stats.max_lift_height = real_lift
            if real_lift > 0.02 and stats.first_real_lift_step is None:
                stats.first_real_lift_step = step

            # Sample traces every 5 steps
            if step % 5 == 0:
                stats.dist_trace.append(dist)
                stats.cube_z_trace.append(cube_z)
                stats.grasp_trace.append(has_grasp)
                stats.action_trace.append(action.copy())

            if terminated:
                if info.get("success", False):
                    stats.success = True
                    stats.terminated_by = "success"
                elif cube_z < _TABLE_Z - 0.05:
                    stats.terminated_by = "drop"
                break

            if truncated:
                stats.terminated_by = "truncated"
                break

        episodes.append(stats)

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
    drops = [e for e in episodes if e.terminated_by == "drop"]
    truncated = [e for e in episodes if e.terminated_by == "truncated"]

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {n} episodes")
    print(f"{'='*60}")

    print(f"\n## Overall")
    print(f"  Success rate:  {len(successes)}/{n} = {100*len(successes)/n:.1f}%")
    print(f"  Drop rate:     {len(drops)}/{n} = {100*len(drops)/n:.1f}%")
    print(f"  Timeout rate:  {len(truncated)}/{n} = {100*len(truncated)/n:.1f}%")
    print(f"  Mean reward:   {np.mean([e.total_reward for e in episodes]):.1f}")

    # Success analysis
    if successes:
        print(f"\n## Successful Episodes ({len(successes)})")
        steps = [e.total_steps for e in successes]
        grasp_steps = [e.first_grasp_step for e in successes if e.first_grasp_step is not None]
        lift_steps = [e.first_real_lift_step for e in successes if e.first_real_lift_step is not None]
        print(f"  Steps to complete:  mean={np.mean(steps):.0f}, min={min(steps)}, max={max(steps)}, std={np.std(steps):.0f}")
        if grasp_steps:
            print(f"  Steps to 1st grasp: mean={np.mean(grasp_steps):.0f}, min={min(grasp_steps)}, max={max(grasp_steps)}")
        if lift_steps:
            print(f"  Steps to 1st lift:  mean={np.mean(lift_steps):.0f}, min={min(lift_steps)}, max={max(lift_steps)}")
        print(f"  Total grasp steps:  mean={np.mean([e.total_grasp_steps for e in successes]):.0f}")
        print(f"  Max lift (above initial): mean={np.mean([e.max_lift_height for e in successes]):.3f}")

    # Failure analysis — mutually exclusive categories
    if failures:
        print(f"\n## Failed Episodes ({len(failures)})")

        # Sequential, mutually exclusive classification
        cat_drop_no_grasp = []     # Knocked cube off without grasping
        cat_drop_after_grasp = []  # Grasped then dropped
        cat_no_reach = []          # Never got close (dist > 0.05)
        cat_reached_no_grasp = []  # Got close but couldn't grasp
        cat_grasped_no_lift = []   # Grasped but couldn't lift enough
        cat_lifted_timeout = []    # Lifted to threshold height but couldn't hold

        for e in failures:
            if e.terminated_by == "drop":
                if e.first_grasp_step is not None:
                    cat_drop_after_grasp.append(e)
                else:
                    cat_drop_no_grasp.append(e)
            elif e.first_grasp_step is None:
                if e.min_dist > 0.05:
                    cat_no_reach.append(e)
                else:
                    cat_reached_no_grasp.append(e)
            elif e.max_lift_height < 0.05:  # 5cm above initial
                cat_grasped_no_lift.append(e)
            else:
                cat_lifted_timeout.append(e)

        total_categorized = (len(cat_drop_no_grasp) + len(cat_drop_after_grasp) +
                            len(cat_no_reach) + len(cat_reached_no_grasp) +
                            len(cat_grasped_no_lift) + len(cat_lifted_timeout))

        print(f"\n  Failure mode breakdown (mutually exclusive):")
        print(f"    Knocked cube off (no grasp):    {len(cat_drop_no_grasp):3d} ({100*len(cat_drop_no_grasp)/n:.0f}% of all)")
        print(f"    Dropped after grasping:         {len(cat_drop_after_grasp):3d} ({100*len(cat_drop_after_grasp)/n:.0f}% of all)")
        print(f"    Never reached cube (>5cm):      {len(cat_no_reach):3d} ({100*len(cat_no_reach)/n:.0f}% of all)")
        print(f"    Reached but couldn't grasp:     {len(cat_reached_no_grasp):3d} ({100*len(cat_reached_no_grasp)/n:.0f}% of all)")
        print(f"    Grasped but couldn't lift 5cm:  {len(cat_grasped_no_lift):3d} ({100*len(cat_grasped_no_lift)/n:.0f}% of all)")
        print(f"    Lifted but timed out:           {len(cat_lifted_timeout):3d} ({100*len(cat_lifted_timeout)/n:.0f}% of all)")
        assert total_categorized == len(failures), f"Categorization mismatch: {total_categorized} != {len(failures)}"

        if failures:
            print(f"\n  Failed episode stats:")
            print(f"    Min distance:     mean={np.mean([e.min_dist for e in failures]):.3f}, min={min(e.min_dist for e in failures):.3f}")
            print(f"    Max real lift:    mean={np.mean([e.max_lift_height for e in failures]):.3f}, max={max(e.max_lift_height for e in failures):.3f}")
            if any(e.first_grasp_step is not None for e in failures):
                grasping_failures = [e for e in failures if e.first_grasp_step is not None]
                print(f"    Grasp steps (when grasped): mean={np.mean([e.total_grasp_steps for e in grasping_failures]):.0f}")

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

    # Drops analysis
    if drops:
        print(f"\n## Drop Events ({len(drops)})")
        print(f"  Drop step:     mean={np.mean([e.total_steps for e in drops]):.0f}, min={min(e.total_steps for e in drops)}, max={max(e.total_steps for e in drops)}")
        early_drops = [e for e in drops if e.total_steps <= 20]
        print(f"  Early drops (≤20 steps): {len(early_drops)}/{len(drops)}")
        if drops:
            print(f"  Drop cube X: mean={np.mean([e.cube_init_x for e in drops]):.3f}, std={np.std([e.cube_init_x for e in drops]):.3f}")
            print(f"  Drop cube Y: mean={np.mean([e.cube_init_y for e in drops]):.3f}, std={np.std([e.cube_init_y for e in drops]):.3f}")

    # Timing distribution for successes
    if successes:
        print(f"\n## Success Timing Distribution")
        bins = [0, 25, 30, 35, 40, 50, 75, 200]
        for i in range(len(bins) - 1):
            count = sum(1 for e in successes if bins[i] <= e.total_steps < bins[i+1])
            bar = "#" * min(count, 60)
            print(f"    {bins[i]:3d}-{bins[i+1]:3d} steps: {count:3d} {bar}")

    # Distance profile for failed episodes
    if failures:
        print(f"\n## Average Distance Profile (failed episodes, n={len(failures)})")
        max_trace_len = max(len(e.dist_trace) for e in failures)
        for t_idx in range(0, min(max_trace_len, 40), 2):  # every 10 steps
            dists = [e.dist_trace[t_idx] for e in failures if t_idx < len(e.dist_trace)]
            grasps = [e.grasp_trace[t_idx] for e in failures if t_idx < len(e.grasp_trace)]
            step = t_idx * 5
            grasp_pct = 100 * sum(grasps) / len(grasps) if grasps else 0
            n_alive = len(dists)
            print(f"    Step {step:3d}: dist={np.mean(dists):.3f} ±{np.std(dists):.3f}, grasping={grasp_pct:.0f}%, alive={n_alive}")

    # Gripper behavior
    print(f"\n## Gripper Behavior")
    print(f"  Success: gripper closed {np.mean([e.gripper_close_steps/max(e.total_steps,1) for e in successes])*100:.0f}% of steps" if successes else "")
    print(f"  Failure: gripper closed {np.mean([e.gripper_close_steps/max(e.total_steps,1) for e in failures])*100:.0f}% of steps" if failures else "")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/jaka_zu5_sim/rl/best_success_model/best_success_model")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    episodes = run_eval(args.model, n_episodes=args.episodes, seed=args.seed)
    analyze(episodes)
