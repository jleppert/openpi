#!/bin/bash
# Train pi0-FAST on JAKA Zu5 pick-cube, then evaluate all checkpoints.
#
# Training: ~9-10 hours (30k steps), saves checkpoints every 5k steps.
# Eval: runs after training on same GPU (model needs ~22GB, only fits on 3090).
#
# CUDA device mapping on this machine:
#   CUDA 0 = RTX 3090 (24GB)  <-- used for both training and eval
#   CUDA 1 = RTX 2080 Ti (11GB)
#   CUDA 2 = RTX 2080 Ti (11GB)
#   CUDA 3 = RTX 3060 (12GB)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl

EXP_NAME="jaka_zu5_pick_cube_v1"
CONFIG="pi0_fast_jaka_zu5_pick_cube"
CKPT_DIR="checkpoints/${CONFIG}/${EXP_NAME}"

echo "=== Starting training ==="
uv run python scripts/train.py \
  "$CONFIG" \
  --exp-name="$EXP_NAME" \
  --overwrite

echo "=== Training complete. Evaluating all checkpoints ==="
uv run python examples/jaka_zu5_sim/eval_vla.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CKPT_DIR" \
  --n-episodes 50 \
  --wandb
