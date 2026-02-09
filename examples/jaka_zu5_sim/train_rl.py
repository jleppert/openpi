"""PPO training script for the JAKA Zu5 pick-cube task."""

import dataclasses
import logging
import pathlib

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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


@dataclasses.dataclass
class Args:
    # Output directory for checkpoints and logs.
    out_dir: pathlib.Path = pathlib.Path("data/jaka_zu5_sim/rl")

    # Training parameters.
    total_timesteps: int = 3_000_000
    n_envs: int = 32
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 50_000

    # PPO hyperparameters — proven settings that reached 70% success.
    # BestSuccessRateCallback captures the peak before any policy collapse.
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    target_kl: float | None = None

    # Network architecture.
    net_arch_pi: int = 256
    net_arch_vf: int = 256
    n_layers: int = 2

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

    net_arch = dict(
        pi=[args.net_arch_pi] * args.n_layers,
        vf=[args.net_arch_vf] * args.n_layers,
    )

    if args.resume:
        logging.info("Resuming from %s", args.resume)
        model = PPO.load(args.resume, env=train_envs)
    else:
        model = PPO(
            "MlpPolicy",
            train_envs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=str(log_dir),
            device="cpu",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(args.out_dir / "checkpoints"),
        name_prefix="ppo_jaka_pick_cube",
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

    logging.info("Starting PPO training for %d timesteps...", args.total_timesteps)
    logging.info("  learning_rate=%.0e, clip_range=%.2f", args.learning_rate, args.clip_range)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, success_cb],
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
