"""Evaluate a VLA checkpoint on the JAKA Zu5 pick-cube task in MuJoCo simulation.

Supports two modes:
  - Single checkpoint evaluation (default)
  - Watch mode (--watch): polls checkpoint directory for new checkpoints

Usage:
    # Single checkpoint eval (quick test)
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 MUJOCO_GL=egl \
      uv run python examples/jaka_zu5_sim/eval_vla.py \
        --config pi0_fast_jaka_zu5_pick_cube \
        --checkpoint-dir checkpoints/pi0_fast_jaka_zu5_pick_cube/jaka_zu5_pick_cube_v1 \
        --n-episodes 3

    # Watch mode (run alongside training on a separate GPU)
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 MUJOCO_GL=egl \
      uv run python examples/jaka_zu5_sim/eval_vla.py \
        --config pi0_fast_jaka_zu5_pick_cube \
        --checkpoint-dir checkpoints/pi0_fast_jaka_zu5_pick_cube/jaka_zu5_pick_cube_v1 \
        --watch --wandb --n-episodes 50
"""

import dataclasses
import gc
import logging
import pathlib
import re
import sys
import time

import imageio
import mujoco
import mujoco.renderer
import numpy as np
from openpi_client import image_tools
import tyro

# Ensure local imports work regardless of working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import camera_config as cam_cfg

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
_MJCF_PATH = _ASSETS_DIR / "jaka_zu5.xml"

# Joint/actuator names (must match MJCF).
_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATOR_NAMES = [f"act_joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT = "finger_left"
_GRIPPER_ACTUATOR = "act_gripper"
_GRIPPER_MAX_OPEN = 0.04

# Physics/control.
_PHYSICS_STEPS = 33  # per control step (~15 Hz at 0.002s timestep)
_IMAGE_SIZE = 224

# Task parameters.
_TABLE_Z = 0.37
_CUBE_X_RANGE = (0.55, 0.82)
_CUBE_Y_RANGE = (-0.35, 0.35)
_WS_X_RANGE = (0.45, 0.85)
_WS_Y_RANGE = (-0.40, 0.40)
_APPROACH_Z = 0.52
_LIFT_THRESHOLD = 0.50


@dataclasses.dataclass
class Args:
    """Arguments for VLA evaluation."""

    # Training config name (e.g. pi0_fast_jaka_zu5_pick_cube).
    config: str = "pi0_fast_jaka_zu5_pick_cube"
    # Path to checkpoint directory.
    checkpoint_dir: str = "checkpoints/pi0_fast_jaka_zu5_pick_cube/jaka_zu5_pick_cube_v1"
    # Enable watcher mode (poll for new checkpoints).
    watch: bool = False
    # Number of eval episodes per checkpoint.
    n_episodes: int = 50
    # Max control steps per episode (~20s at 15Hz).
    max_steps_per_episode: int = 300
    # Enable W&B logging (connects to same run via wandb_id.txt).
    wandb: bool = False
    # Random seed.
    seed: int = 42
    # Polling interval in seconds for watch mode.
    poll_interval: int = 60
    # Output directory for videos and action logs.
    output_dir: str = "eval_output"
    # Disable wrist camera (send black image) to test if it hurts performance.
    disable_wrist_cam: bool = False


class SimEvaluator:
    """Runs VLA policy in MuJoCo and checks pick-cube success."""

    def __init__(self, seed: int = 42):
        self._model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=480, width=640)

        # Cache IDs.
        self._jids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES]
        self._aids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES]
        self._grip_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT)
        self._grip_aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR)
        self._cube_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._cube_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        self._grip_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self._ext_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "external_cam")
        self._wrist_cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        self._rng = np.random.default_rng(seed)
        self._dt = self._model.opt.timestep * _PHYSICS_STEPS

    def reset(self):
        """Reset sim with randomized cube and arm start positions (matches collect_demos.py)."""
        mujoco.mj_resetData(self._model, self._data)

        # Randomize cube on table.
        cx = self._rng.uniform(*_CUBE_X_RANGE)
        cy = self._rng.uniform(*_CUBE_Y_RANGE)
        adr = self._model.jnt_qposadr[self._cube_jid]
        self._data.qpos[adr : adr + 3] = [cx, cy, _TABLE_Z + 0.03]
        self._data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

        # Nominal home pose.
        j2, j3 = -1.48, 1.8
        home = [0.0, j2, j3, 0.0, -(j2 + j3), 0.0]
        for i, (jid, val) in enumerate(zip(self._jids, home)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._aids[i]] = val
        mujoco.mj_forward(self._model, self._data)

        # Randomize starting position within workspace.
        sx = self._rng.uniform(*_WS_X_RANGE)
        sy = self._rng.uniform(*_WS_Y_RANGE)
        start_jt = self._solve_ik(np.array([sx, sy, _APPROACH_Z]))
        for i, (jid, val) in enumerate(zip(self._jids, start_jt)):
            self._data.qpos[self._model.jnt_qposadr[jid]] = val
            self._data.ctrl[self._aids[i]] = val

        # Open gripper.
        self._data.qpos[self._model.jnt_qposadr[self._grip_jid]] = _GRIPPER_MAX_OPEN
        self._data.ctrl[self._grip_aid] = _GRIPPER_MAX_OPEN
        mujoco.mj_forward(self._model, self._data)

    def get_observation(self) -> dict:
        """Produce DROID-format observation (matches env.py:get_observation)."""
        ext_img = self._render_camera(self._ext_cam_id)
        wrist_img = self._render_camera(self._wrist_cam_id)

        ext_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(ext_img, _IMAGE_SIZE, _IMAGE_SIZE))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, _IMAGE_SIZE, _IMAGE_SIZE))

        # EXPERIMENT: disable wrist camera to test if it hurts more than helps.
        if getattr(self, "_disable_wrist_cam", False):
            wrist_img = np.zeros_like(wrist_img)

        joint_pos = np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._jids])
        joint_pos_3 = joint_pos[:3]  # Only active joints: J1, J2, J3

        gripper_raw = self._data.qpos[self._model.jnt_qposadr[self._grip_jid]]
        gripper_pos = np.array([gripper_raw / _GRIPPER_MAX_OPEN])

        # Keys must match the RepackTransform source keys in the training config's
        # data_transforms (LeRobot dataset format → DROID observation format).
        return {
            "exterior_image_1_left": ext_img,
            "wrist_image_left": wrist_img,
            "joint_position": joint_pos_3.astype(np.float32),
            "gripper_position": gripper_pos.astype(np.float32),
            "task": "pick up the red cube",
            # Dummy actions — required by RepackTransform but unused during inference.
            "actions": np.zeros(4, dtype=np.float32),
        }

    def apply_single_action(self, actions: np.ndarray):
        """Apply a single action vector (4,) to the sim.

        Actions are [J1_vel, J2_vel, J3_vel, gripper]. J4 and J6 are set to 0.
        J5 is computed analytically as -(J2+J3) to keep the gripper pointing down.
        """
        j1_vel, j2_vel, j3_vel = actions[0], actions[1], actions[2]
        gripper_cmd = actions[3]

        # Expand to 6-DOF: [J1, J2, J3, 0 (J4), J5_comp, 0 (J6)]
        joint_vel = np.array([j1_vel, j2_vel, j3_vel, 0.0, -(j2_vel + j3_vel), 0.0])

        dt = self._model.opt.timestep * _PHYSICS_STEPS
        current_pos = np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._jids])
        target_pos = np.clip(
            current_pos + joint_vel * dt,
            [self._model.jnt_range[jid, 0] for jid in self._jids],
            [self._model.jnt_range[jid, 1] for jid in self._jids],
        )

        for aid, pos in zip(self._aids, target_pos):
            self._data.ctrl[aid] = pos
        self._data.ctrl[self._grip_aid] = _GRIPPER_MAX_OPEN if gripper_cmd > 0.5 else 0.0

        for _ in range(_PHYSICS_STEPS):
            mujoco.mj_step(self._model, self._data)

    def check_success(self) -> bool:
        """Check if cube is lifted above threshold."""
        return bool(self._data.xpos[self._cube_bid][2] > _LIFT_THRESHOLD)

    def gripper_cube_distance(self) -> float:
        """Euclidean distance between gripper and cube."""
        return float(np.linalg.norm(self._data.xpos[self._grip_bid] - self._data.xpos[self._cube_bid]))

    def render_frame(self) -> np.ndarray:
        """Render external camera at full 480x640 resolution for video recording."""
        return self._render_camera(self._ext_cam_id).copy()

    def get_joint_state(self) -> tuple[np.ndarray, float]:
        """Return current (joint_positions (6,), normalized_gripper)."""
        joint_pos = np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._jids])
        gripper_raw = self._data.qpos[self._model.jnt_qposadr[self._grip_jid]]
        return joint_pos, gripper_raw / _GRIPPER_MAX_OPEN

    def _render_camera(self, cam_id: int) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=cam_id)
        return self._renderer.render()

    def _solve_ik(self, target_pos: np.ndarray) -> np.ndarray:
        """Jacobian-based IK for J1-J3 (matches collect_demos.py)."""
        qpos_save = self._data.qpos.copy()
        qvel_save = self._data.qvel.copy()

        nv = self._model.nv
        jacp = np.zeros((3, nv))
        controlled = [0, 1, 2]

        for _ in range(20):
            mujoco.mj_forward(self._model, self._data)
            err = target_pos - self._data.xpos[self._grip_bid]
            if np.linalg.norm(err) < 1e-3:
                break

            jacp[:] = 0
            mujoco.mj_jacBody(self._model, self._data, jacp, None, self._grip_bid)
            dof_idx = [self._model.jnt_dofadr[self._jids[j]] for j in controlled]
            J = jacp[:, dof_idx]
            JJT = J @ J.T + (0.01**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)

            for idx, cj in enumerate(controlled):
                jid = self._jids[cj]
                adr = self._model.jnt_qposadr[jid]
                lo, hi = self._model.jnt_range[jid]
                self._data.qpos[adr] = np.clip(self._data.qpos[adr] + dq[idx], lo, hi)

            j2 = self._data.qpos[self._model.jnt_qposadr[self._jids[1]]]
            j3 = self._data.qpos[self._model.jnt_qposadr[self._jids[2]]]
            j5_id = self._jids[4]
            j5_lo, j5_hi = self._model.jnt_range[j5_id]
            self._data.qpos[self._model.jnt_qposadr[self._jids[3]]] = 0.0
            self._data.qpos[self._model.jnt_qposadr[j5_id]] = np.clip(-(j2 + j3), j5_lo, j5_hi)
            self._data.qpos[self._model.jnt_qposadr[self._jids[5]]] = 0.0

        result = np.array([self._data.qpos[self._model.jnt_qposadr[j]] for j in self._jids])
        self._data.qpos[:] = qpos_save
        self._data.qvel[:] = qvel_save
        mujoco.mj_forward(self._model, self._data)
        return result


def evaluate_checkpoint(
    policy,
    sim: SimEvaluator,
    n_episodes: int,
    max_steps: int,
    output_dir: pathlib.Path | None = None,
    ckpt_step: int | None = None,
    use_wandb: bool = False,
) -> dict:
    """Run n_episodes and return metrics.

    If output_dir and ckpt_step are provided, saves per-episode:
      - MP4 video from external camera
      - NPZ with action_chunks, joint_positions, gripper_positions, success
    """
    successes = 0
    episode_lengths = []
    final_dists = []

    # Create output directory for this checkpoint.
    save_dir = None
    if output_dir is not None and ckpt_step is not None:
        save_dir = output_dir / str(ckpt_step)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Warm up JIT with a dummy inference call.
    logging.info("Warming up policy (JIT compilation, may take a few minutes)...")
    sim.reset()
    t0 = time.time()
    obs = sim.get_observation()
    _ = policy.infer(obs)
    logging.info("JIT warmup done in %.1fs. Starting evaluation.", time.time() - t0)

    first_video_path = None

    for ep in range(n_episodes):
        sim.reset()
        success = False
        ep_start = time.time()
        step = 0

        # Per-episode recording buffers.
        frames: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        joint_positions: list[np.ndarray] = []
        gripper_positions: list[float] = []

        # Record initial frame and state.
        if save_dir is not None:
            frames.append(sim.render_frame())
            jp, gp = sim.get_joint_state()
            joint_positions.append(jp)
            gripper_positions.append(gp)

        while step < max_steps:
            obs = sim.get_observation()
            action = policy.infer(obs)

            # Policy returns action chunk (action_horizon, action_dim).
            raw = np.asarray(action["actions"])
            chunk = raw if raw.ndim == 2 else raw[np.newaxis, :]

            if save_dir is not None:
                action_chunks.append(chunk.copy())

            # Prevent JAX/numpy memory accumulation.
            del obs, action

            # Execute all actions in the chunk.
            for a in chunk:
                sim.apply_single_action(a)
                step += 1

                if save_dir is not None:
                    frames.append(sim.render_frame())
                    jp, gp = sim.get_joint_state()
                    joint_positions.append(jp)
                    gripper_positions.append(gp)

                if sim.check_success():
                    success = True
                    break
                if step >= max_steps:
                    break

            if success:
                episode_lengths.append(step)
                break

        if success:
            successes += 1
        final_dists.append(sim.gripper_cube_distance())

        # Save episode data.
        if save_dir is not None:
            video_path = save_dir / f"episode_{ep}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=15, quality=8)
            if first_video_path is None:
                first_video_path = video_path

            np.savez_compressed(
                save_dir / f"episode_{ep}.npz",
                action_chunks=np.array(action_chunks, dtype=object),
                joint_positions=np.array(joint_positions),
                gripper_positions=np.array(gripper_positions),
                success=success,
            )
            logging.info("Saved %s (%d frames)", video_path, len(frames))

        # Clean up between episodes to prevent memory accumulation.
        del frames, action_chunks, joint_positions, gripper_positions
        gc.collect()

        ep_time = time.time() - ep_start
        logging.info(
            "Episode %d/%d: %s in %d steps (%.1fs, %.0f Hz), dist=%.3f",
            ep + 1, n_episodes, "SUCCESS" if success else "fail",
            step, ep_time, step / ep_time, final_dists[-1],
        )

    success_rate = successes / n_episodes
    mean_ep_len = float(np.mean(episode_lengths)) if episode_lengths else float(max_steps)
    mean_dist = float(np.mean(final_dists))

    metrics = {
        "eval/success_rate": success_rate,
        "eval/mean_episode_length": mean_ep_len,
        "eval/mean_cube_dist": mean_dist,
    }

    # Log first episode video to W&B.
    if use_wandb and first_video_path is not None:
        import wandb

        metrics["eval/video"] = wandb.Video(str(first_video_path), fps=15, format="mp4")

    return metrics


def get_checkpoint_steps(checkpoint_dir: pathlib.Path) -> list[int]:
    """Return sorted list of completed checkpoint step numbers."""
    from orbax.checkpoint.utils import is_checkpoint_finalized, is_tmp_checkpoint

    steps = []
    if not checkpoint_dir.exists():
        return steps

    for p in checkpoint_dir.iterdir():
        if not p.is_dir():
            continue
        # Skip tmp (in-progress) checkpoints.
        if is_tmp_checkpoint(p):
            continue
        # Only consider numeric directory names.
        if not re.fullmatch(r"\d+", p.name):
            continue
        # Verify finalized.
        if is_checkpoint_finalized(p):
            steps.append(int(p.name))

    return sorted(steps)


def load_policy(config_name: str, checkpoint_dir: pathlib.Path, step: int):
    """Load a policy from a specific checkpoint step."""
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    train_config = get_config(config_name)
    step_dir = checkpoint_dir / str(step)
    logging.info("Loading checkpoint step %d from %s", step, step_dir)
    policy = create_trained_policy(train_config, step_dir)
    return policy


def main(args: Args) -> None:
    import jax
    import etils.epath as epath

    # Enable persistent JAX compilation cache (same as train.py).
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)

    # W&B setup.
    if args.wandb:
        import wandb

        wandb_id_file = checkpoint_dir / "wandb_id.txt"
        if wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb.init(id=run_id, resume="must", project="openpi")
            logging.info("Connected to W&B run %s", run_id)
        else:
            logging.warning("wandb_id.txt not found in %s, starting new W&B run", checkpoint_dir)
            wandb.init(project="openpi", name=f"eval-{checkpoint_dir.name}")

    sim = SimEvaluator(seed=args.seed)
    sim._disable_wrist_cam = args.disable_wrist_cam
    if args.disable_wrist_cam:
        logging.info("Wrist camera DISABLED (sending black images)")

    if not args.watch:
        # Single evaluation: evaluate all checkpoints.
        steps = get_checkpoint_steps(checkpoint_dir)
        if not steps:
            logging.error("No completed checkpoints found in %s", checkpoint_dir)
            return
        steps = steps[::-1]  # Evaluate newest checkpoints first.
        logging.info("Found %d checkpoints: %s", len(steps), steps)

        for i, step in enumerate(steps):
            logging.info("Evaluating checkpoint %d/%d: step %d", i + 1, len(steps), step)
            policy = load_policy(args.config, checkpoint_dir, step)
            metrics = evaluate_checkpoint(
                policy, sim, args.n_episodes, args.max_steps_per_episode,
                output_dir=output_dir, ckpt_step=step, use_wandb=args.wandb,
            )
            del policy
            gc.collect()

            print(f"\n=== Step {step} ===")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")

            if args.wandb:
                import wandb

                wandb.log(metrics, step=step)
    else:
        # Watch mode: poll for new checkpoints.
        evaluated: set[int] = set()
        logging.info("Watching %s for new checkpoints (poll every %ds)...", checkpoint_dir, args.poll_interval)

        current_policy = None
        current_step = None

        while True:
            steps = get_checkpoint_steps(checkpoint_dir)
            new_steps = sorted(set(steps) - evaluated)

            if not new_steps:
                logging.info("No new checkpoints. Waiting %ds...", args.poll_interval)
                time.sleep(args.poll_interval)
                continue

            for step in new_steps:
                logging.info("New checkpoint detected: step %d", step)

                # Load new policy (reloads model each time to pick up new weights).
                del current_policy
                current_policy = load_policy(args.config, checkpoint_dir, step)
                current_step = step

                metrics = evaluate_checkpoint(
                    current_policy, sim, args.n_episodes, args.max_steps_per_episode,
                    output_dir=output_dir, ckpt_step=step, use_wandb=args.wandb,
                )
                evaluated.add(step)

                print(f"\n=== Step {step} ===")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")

                if args.wandb:
                    import wandb

                    wandb.log(metrics, step=step)

            logging.info("Waiting %ds for next poll...", args.poll_interval)
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = tyro.cli(Args)
    main(args)
