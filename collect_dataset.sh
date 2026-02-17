#!/bin/bash

MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.n-episodes 1000 \
    --args.output-dir data/jaka_zu5_sim/datasets/jaka_zu5_pick_cube_v4
