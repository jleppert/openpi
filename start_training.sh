#!/bin/bash

MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py --args.n-envs 32 --args.total-timesteps 3000000
