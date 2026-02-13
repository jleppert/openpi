#!/bin/bash

MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py --args.total-timesteps 100000 --args.n-envs 4 --args.eval-freq 2500 --args.checkpoint-freq 10000 --args.video-eval-freq 50000 --args.out-dir data/jaka_zu5_sim/rl_test4

#MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py --args.total-timesteps 10000 --args.n-envs 4 --args.eval-freq 2500 --args.checkpoint-freq 5000 --args.video-eval-freq 100000 --args.out-dir data/jaka_zu5_sim/rl_test

#MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py --args.out-dir data/jaka_zu5_sim/rl_sac

#MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py --args.n-envs 32 --args.total-timesteps 3000000
