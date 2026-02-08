#!/bin/bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi0_fast_droid \
  --policy.dir=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid
