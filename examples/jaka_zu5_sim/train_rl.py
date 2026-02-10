"""SAC training script for the JAKA Zu5 pick-cube task."""

import dataclasses
import logging
import pathlib

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer
import tyro


class BestSuccessRateCallback(BaseCallback):
    """Saves the model with the highest evaluation success rate."""

    def __init__(self, eval_env, eval_freq, n_eval_episodes, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = pathlib.Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_success_rate = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        successes = []
        # Run eval episodes across vectorized envs.
        n_envs = self.eval_env.num_envs
        obs = self.eval_env.reset()
        episode_counts = 0
        # Track per-env episode completion.
        episode_success = [False] * n_envs
        episode_done = [False] * n_envs

        while episode_counts < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            for i in range(n_envs):
                if dones[i] and not episode_done[i]:
                    successes.append(infos[i].get("is_success", False))
                    episode_counts += 1
                    if episode_counts >= self.n_eval_episodes:
                        break

        success_rate = np.mean(successes) if successes else 0.0
        self.logger.record("eval/success_rate_custom", success_rate)

        if self.verbose:
            logging.info(
                "Eval timesteps=%d, success_rate=%.1f%% (%d/%d)",
                self.num_timesteps, 100 * success_rate,
                sum(successes), len(successes),
            )

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            save_file = self.save_path / "best_success_model"
            self.model.save(str(save_file))
            if self.verbose:
                logging.info(
                    "New best success rate: %.1f%% — saved to %s",
                    100 * success_rate, save_file,
                )

        return True


class VideoEvalCallback(BaseCallback):
    """Runs full eval_model analysis with video recording at regular intervals."""

    def __init__(self, out_dir, eval_freq, n_eval_episodes=50, save_videos=5, verbose=1):
        super().__init__(verbose)
        self.out_dir = pathlib.Path(out_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_videos = save_videos

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        import eval_model

        timesteps = self.num_timesteps

        # Save current model to a temp location for eval.
        snap_path = self.out_dir / "checkpoints" / f"eval_snap_{timesteps}"
        self.model.save(str(snap_path))

        video_dir = str(self.out_dir / "eval_videos" / f"step_{timesteps}")
        logging.info("=== VideoEval at %d steps: %d episodes, %d videos → %s ===",
                      timesteps, self.n_eval_episodes, self.save_videos, video_dir)

        episodes = eval_model.run_eval(
            model_path=str(snap_path),
            n_episodes=self.n_eval_episodes,
            save_videos=self.save_videos,
            video_dir=video_dir,
            verbose=False,
        )
        eval_model.analyze(episodes)

        # Log summary to tensorboard.
        n_success = sum(1 for e in episodes if e.success)
        n_drops = sum(1 for e in episodes if e.terminated_by == "drop")
        self.logger.record("video_eval/success_rate", n_success / len(episodes))
        self.logger.record("video_eval/drop_rate", n_drops / len(episodes))
        self.logger.record("video_eval/mean_reward",
                           np.mean([e.total_reward for e in episodes]))

        return True


@dataclasses.dataclass
class Args:
    # Output directory for checkpoints and logs.
    out_dir: pathlib.Path = pathlib.Path("data/jaka_zu5_sim/rl")

    # Training parameters.
    total_timesteps: int = 1_000_000
    n_envs: int = 32
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 50_000

    # SAC hyperparameters (tuned for HER, based on rl-baselines3-zoo FetchPickAndPlace).
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    tau: float = 0.05
    gamma: float = 0.95
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 4

    # Network architecture.
    net_arch: int = 512
    n_layers: int = 3

    # Video evaluation during training.
    video_eval_freq: int = 50_000
    video_eval_episodes: int = 50
    video_eval_videos: int = 5

    # Resume from checkpoint.
    resume: str | None = None


def _make_env():
    import gym_env  # noqa: F811
    return gym_env.JakaZu5PickCubeEnv()


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating %d parallel environments...", args.n_envs)

    # Training envs.
    train_envs = make_vec_env(
        _make_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Eval envs (fewer, for periodic evaluation).
    eval_envs = make_vec_env(
        _make_env,
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
    )

    # Separate eval envs for success rate callback.
    success_eval_envs = make_vec_env(
        _make_env,
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
    )

    net_arch = [args.net_arch] * args.n_layers

    if args.resume:
        logging.info("Resuming from %s", args.resume)
        model = SAC.load(args.resume, env=train_envs)
    else:
        model = SAC(
            "MultiInputPolicy",
            train_envs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_entropy=-2.0,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=str(log_dir),
            device="auto",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(args.out_dir / "checkpoints"),
        name_prefix="sac_jaka_pick_cube",
    )

    eval_cb = EvalCallback(
        eval_envs,
        best_model_save_path=str(args.out_dir / "best_reward_model"),
        log_path=str(args.out_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    success_cb = BestSuccessRateCallback(
        eval_env=success_eval_envs,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        save_path=str(args.out_dir / "best_success_model"),
    )

    video_eval_cb = VideoEvalCallback(
        out_dir=args.out_dir,
        eval_freq=max(args.video_eval_freq // args.n_envs, 1),
        n_eval_episodes=args.video_eval_episodes,
        save_videos=args.video_eval_videos,
    )

    logging.info("Starting SAC training for %d timesteps...", args.total_timesteps)
    logging.info("  learning_rate=%.0e, buffer_size=%d, batch_size=%d", args.learning_rate, args.buffer_size, args.batch_size)
    logging.info("  video eval every %d steps: %d episodes, %d videos", args.video_eval_freq, args.video_eval_episodes, args.video_eval_videos)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, success_cb, video_eval_cb],
        progress_bar=True,
    )

    final_path = args.out_dir / "final_model"
    model.save(str(final_path))
    logging.info("Training complete. Final model saved to %s", final_path)
    logging.info("Best success rate: %.1f%%", 100 * success_cb.best_success_rate)

    train_envs.close()
    eval_envs.close()
    success_eval_envs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    tyro.cli(main)
