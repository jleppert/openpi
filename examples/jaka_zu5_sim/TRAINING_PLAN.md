# JAKA Zu5 RL Training + VLA Fine-Tuning Plan

End-to-end pipeline: RL learns a pick-cube task in simulation, generates demonstrations,
then LoRA fine-tunes the pi0_fast VLA to produce a vision-language-action model for the
JAKA Zu5.

## Pipeline Overview

```
Phase 1: RL Training (CPU only)
  gym_env.py + train_rl.py → PPO learns top-down cube pick-up (~1-2 hours)

Phase 2: Demo Collection (CPU + EGL)
  collect_demos.py → 500 episodes with camera rendering → LeRobot dataset (~30-60 min)

Phase 3: LoRA Fine-Tuning (RTX 3090, 24GB)
  LeRobot dataset + pi0_fast_droid checkpoint → fine-tuned model (~4-12 hours)

Phase 4: Evaluation
  Fine-tuned model → policy server → JAKA Zu5 sim → verify task completion
```

---

## Phase 1: RL Training

### Implementation: `gym_env.py` + `train_rl.py`

Both files are implemented and tested. A 50K-step smoke test completed in ~90 seconds
(730 FPS with 8 parallel envs on CPU), with mean reward improving from -171 to -151.

### Task Definition

**Task:** Pick up the red cube.

**Success condition:** Cube center is 10+ cm above the table surface and held in the
gripper for 15 consecutive steps (~1 second at 15 Hz).

**Episode structure:**
- Max 200 steps (~13.3s at 15 Hz control)
- Early termination on success (cube held above threshold)
- Early termination if cube falls off table (z < table - 0.05)
- Arm starts from home pose with small random perturbations

### Observation Space (11D vector, no images)

RL trains on low-dimensional state, not pixels. Images are only captured during demo
collection in Phase 2.

| Dimension | Description | Range |
|-----------|-------------|-------|
| 0-5 | Joint positions (6 DOF) | Joint limits |
| 6 | Gripper opening (normalized) | [0, 1] |
| 7-9 | Cube position (x, y, z) | Workspace bounds |
| 10 | Gripper-to-cube distance | [0, ~1.5] |

### Action Space (4D continuous, top-down constrained)

| Dimension | Description | Range |
|-----------|-------------|-------|
| 0 | J1 velocity command | [-1, 1] |
| 1 | J2 velocity command | [-1, 1] |
| 2 | J3 velocity command | [-1, 1] |
| 3 | Gripper command (>0 = close, <0 = open) | [-1, 1] |

Wrist joints (J4, J5, J6) are locked to `[0.0, 0.5, 0.0]` to maintain a downward-pointing
orientation. This reduces the search space from 7D to 4D, making training much faster.

### Reward Function

Dense shaped reward to guide learning through reach-grasp-lift phases:

```
R(s, a) = R_reach + R_grasp + R_lift + R_success + R_penalty
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| R_reach | `-2.0 * distance(gripper, cube)` | Guide gripper toward cube |
| R_grasp | `+1.0` if both fingers contact cube | Reward successful grasp |
| R_lift | `+5.0 * lift_height` (only if grasped) | Reward lifting after grasp |
| R_success | `+10.0` on task completion | Sparse bonus for full success |
| R_penalty | `-5.0` if cube falls off table | Penalize dropping the cube |

### Domain Randomization

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Cube start X | [0.35, 0.55] m | Vary reach distance |
| Cube start Y | [-0.10, 0.10] m | Vary lateral position |
| Arm home pose | +/- 0.05 rad per controlled joint | Vary initial configuration |

The green cylinder is moved below the floor during RL training.

### RL Algorithm: PPO

Stable-Baselines3 PPO with MLP policy:
- 2 hidden layers x 256 units
- 16 parallel vectorized environments (`SubprocVecEnv`)
- Learning rate: 3e-4, batch size: 64, n_steps: 2048, n_epochs: 10
- Runs on **CPU** (`device="cpu"`)
- Evaluation: 20 episodes every 10K timesteps
- Checkpoints every 50K timesteps

### How to Run

```bash
cd ~/openpi

# Full training run (1M timesteps, ~1-2 hours)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py

# With custom parameters
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py \
    --args.total-timesteps 2000000 \
    --args.n-envs 16 \
    --args.learning-rate 3e-4

# Resume from a checkpoint
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py \
    --args.resume data/jaka_zu5_sim/rl/checkpoints/ppo_jaka_pick_cube_500000_steps.zip

# Quick test run (50K steps, ~90 seconds)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py \
    --args.total-timesteps 50000 \
    --args.n-envs 8
```

### Output

All outputs go to `data/jaka_zu5_sim/rl/`:

```
data/jaka_zu5_sim/rl/
├── best_model/best_model.zip    # Best model by eval reward
├── checkpoints/                 # Periodic checkpoints
│   ├── ppo_jaka_pick_cube_50000_steps.zip
│   ├── ppo_jaka_pick_cube_100000_steps.zip
│   └── ...
├── eval_logs/evaluations.npz    # Eval metrics over training
├── logs/PPO_1/                  # Tensorboard logs
└── final_model.zip              # Model at end of training
```

### Monitor Training Progress

```bash
# Tensorboard (in a separate terminal)
uv run tensorboard --logdir data/jaka_zu5_sim/rl/logs

# Check eval results from the saved npz
uv run python -c "
import numpy as np
d = np.load('data/jaka_zu5_sim/rl/eval_logs/evaluations.npz')
for ts, r in zip(d['timesteps'], d['results'].mean(axis=1)):
    print(f'Step {ts:>8d}: mean_reward = {r:.1f}')
"
```

### Success Criteria for Phase 1

Before proceeding to demo collection:
- Success rate > 80% over 100 evaluation episodes
- Mean episode length < 150 steps (indicates efficient behavior)
- Grasps are stable (cube doesn't slip during lift)

---

## Phase 2: Demo Collection

### Implementation: `collect_demos.py`

Rolls out the trained RL policy while recording full DROID-format observations with camera
rendering. Only successful episodes are saved by default.

### What Gets Recorded

At each step the script records:
- **External camera image**: 640x480 rendered, resized+padded to 224x224 uint8 HWC
- **Wrist camera image**: Same processing
- **Joint positions**: 6D from MuJoCo, zero-padded to 7D for DROID format
- **Gripper position**: Normalized [0, 1]
- **Action**: 4D RL action mapped to 8D DROID format (see below)
- **Task string**: `"pick up the red cube"`

### Action Mapping (4D RL → 8D DROID)

| DROID Dim | Source |
|-----------|--------|
| 0-2 | J1-J3 velocity from RL policy |
| 3-5 | 0.0 (wrist joints locked) |
| 6 | 0.0 (padded 7th DOF) |
| 7 | Inverted gripper: RL >0=close → DROID 0.0; RL <0=open → DROID 1.0 |

### How to Run

```bash
cd ~/openpi

# Collect 500 successful demos using best model
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.model-path data/jaka_zu5_sim/rl/best_model/best_model.zip

# Collect 200 demos (faster, for initial testing)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.model-path data/jaka_zu5_sim/rl/best_model/best_model.zip \
    --args.n-episodes 200

# Save all episodes (not just successful ones)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.model-path data/jaka_zu5_sim/rl/best_model/best_model.zip \
    --args.only-successful false

# Use the final model instead
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.model-path data/jaka_zu5_sim/rl/final_model.zip
```

### Output

The dataset is saved in LeRobot format to `~/.cache/lerobot/levelhq/jaka_zu5_pick_cube/`.

```python
# Dataset structure
LeRobotDataset(
    repo_id="levelhq/jaka_zu5_pick_cube",
    robot_type="jaka_zu5",
    fps=15,
    features={
        "exterior_image_1_left": image (224, 224, 3),
        "wrist_image_left":      image (224, 224, 3),
        "joint_position":        float32 (7,),
        "gripper_position":      float32 (1,),
        "actions":               float32 (8,),
    },
)
```

### Success Criteria for Phase 2

- 400-500 successful episodes saved (80%+ success rate)
- Images contain meaningful content (arm visible, cube visible)
- Actions are smooth (no jittering from noisy RL policy)

---

## Phase 3: LoRA Fine-Tuning

### Not Yet Implemented

This phase requires adding a training config to `src/openpi/training/config.py` and
running the OpenPI training pipeline on the collected dataset.

### Training Config

Add to `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi0_fast_jaka_zu5_pick_cube",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=8,
        action_horizon=10,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    ),
    data=SimpleDataConfig(
        repo_id="levelhq/jaka_zu5_pick_cube",
        assets=AssetsConfig(asset_id="jaka_zu5"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid/params"
    ),
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=8,
        action_horizon=10,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    batch_size=2,
    num_train_steps=30_000,
)
```

Key choices:
- **`gemma_2b_lora`**: LoRA adapters on the PaLI-Gemma backbone, fits in 24GB VRAM
- **`batch_size=2`**: Minimizes memory footprint
- **`ema_decay=None`**: Disabled for LoRA (not enough parameters to benefit)
- **Base checkpoint**: `pi0_fast_droid` to leverage DROID pre-training

### How to Run (Once Config Is Added)

```bash
cd ~/openpi

# Step 1: Compute normalization statistics over the demo dataset
uv run python scripts/compute_norm_stats.py \
    --config-name=pi0_fast_jaka_zu5_pick_cube

# Step 2: Train with LoRA fine-tuning on RTX 3090
CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py \
    --config-name=pi0_fast_jaka_zu5_pick_cube
```

### Hardware

- RTX 3090 (24GB) with LoRA + batch_size=2
- Estimated training time: 4-12 hours for 30K steps
- Monitor loss convergence; may stop early if loss plateaus

---

## Phase 4: Evaluation

### How to Run (Once Fine-Tuned Model Exists)

```bash
cd ~/openpi

# Terminal 1: Serve the fine-tuned model
CUDA_VISIBLE_DEVICES=3 uv run python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_fast_jaka_zu5_pick_cube \
    --policy.dir=<path-to-finetuned-checkpoint>

# Terminal 2: Run the simulation (headless, records video)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/main.py

# Terminal 2 alternative: With live MuJoCo viewer
DISPLAY=:0 uv run python examples/jaka_zu5_sim/main.py --args.display
```

### Evaluation Metrics

Run 50 episodes with randomized cube positions:

| Metric | Target |
|--------|--------|
| Success rate | > 50% (good for first attempt) |
| Mean time to grasp | < 5s |
| Grasp stability | Cube doesn't slip after initial lift |

A 50%+ success rate on the first round validates the full pipeline. Subsequent rounds
can improve by collecting more demos, tuning rewards, or increasing training steps.

---

## Quick Reference: Full Pipeline Commands

```bash
cd ~/openpi

# Phase 1: Train RL policy (~1-2 hours, CPU)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/train_rl.py

# Phase 2: Collect demos (~30-60 min, CPU + EGL)
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/collect_demos.py \
    --args.model-path data/jaka_zu5_sim/rl/best_model/best_model.zip

# Phase 3: Compute norm stats, then fine-tune (4-12 hours, RTX 3090)
uv run python scripts/compute_norm_stats.py --config-name=pi0_fast_jaka_zu5_pick_cube
CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py --config-name=pi0_fast_jaka_zu5_pick_cube

# Phase 4: Serve and evaluate
CUDA_VISIBLE_DEVICES=3 uv run python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_fast_jaka_zu5_pick_cube \
    --policy.dir=<checkpoint-path>
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/main.py
```

---

## File Reference

| File | Status | Description |
|------|--------|-------------|
| `gym_env.py` | Implemented | Gymnasium environment: 11D obs, 4D action, shaped rewards, domain randomization |
| `train_rl.py` | Implemented | PPO training with SB3: 16 parallel envs, eval callbacks, checkpoints, tensorboard |
| `collect_demos.py` | Implemented | Rolls out trained policy with camera rendering, saves LeRobot dataset |
| `evaluate.py` | Not yet needed | Can be added later for batch evaluation metrics |

---

## Timeline Estimate

| Phase | Time | Hardware |
|-------|------|----------|
| Phase 1: RL training | 1-2 hours | CPU (16 cores) |
| Phase 2: Demo collection | 30-60 min | CPU + EGL |
| Phase 3: LoRA fine-tuning | 4-12 hours | RTX 3090 |
| Phase 4: Evaluation | 30 min | RTX 3090 |
| **Total** | **~1 day** | |

## Future Tasks (After Validation)

- **More objects:** Add cylinder, bowl, other shapes
- **Multi-step tasks:** "Pick up the red cube and place it on the green plate"
- **Unconstrained grasps:** Remove top-down constraint, allow angled approaches
- **Real hardware:** Swap sim env for JAKA SDK / ROS environment
- **Sim-to-real transfer:** Domain randomization on textures, lighting, camera noise
