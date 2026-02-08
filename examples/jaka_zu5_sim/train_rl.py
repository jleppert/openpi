"""PPO training script for the JAKA Zu5 pick-cube task."""

import dataclasses
import logging
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import tyro


@dataclasses.dataclass
class Args:
    # Output directory for checkpoints and logs.
    out_dir: pathlib.Path = pathlib.Path("data/jaka_zu5_sim/rl")

    # Training parameters.
    total_timesteps: int = 1_000_000
    n_envs: int = 16
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 50_000

    # PPO hyperparameters.
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

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
        best_model_save_path=str(args.out_dir / "best_model"),
        log_path=str(args.out_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    logging.info("Starting PPO training for %d timesteps...", args.total_timesteps)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = args.out_dir / "final_model"
    model.save(str(final_path))
    logging.info("Training complete. Final model saved to %s", final_path)

    train_envs.close()
    eval_envs.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    tyro.cli(main)
