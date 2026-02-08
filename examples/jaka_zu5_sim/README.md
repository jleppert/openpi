# JAKA Zu5 MuJoCo + OpenPI pi0_fast_droid Demo

End-to-end demonstration of a JAKA Zu5 robot arm in MuJoCo simulation driven by the
OpenPI `pi0_fast_droid` policy server. The MuJoCo environment renders camera images and
reads joint states, sends them to the policy server over WebSocket, receives action
predictions, and applies them to the simulated robot.

## Architecture

```
Terminal 1: OpenPI Policy Server (pi0_fast_droid, WebSocket :8000)
                    ^ observations        v actions [10, 8]
Terminal 2: MuJoCo Client (JAKA Zu5 sim)
              - env.py:   renders cameras, reads joints, applies actions
              - main.py:  runtime loop, connects to server
              - saver.py: records video of each episode
```

The client uses the DROID-native observation format so `DroidInputs` on the server
transforms them directly -- no server-side code changes are needed.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with >= 12 GB VRAM (tested on RTX 3060 / RTX 3090)
- CUDA 12.x drivers
- `uv` package manager (OpenPI uses uv for all dependency management)
- For headless machines: EGL support (`MUJOCO_GL=egl`)

## Quick Start

### 1. Clone and install OpenPI

```bash
cd /home/johnathan/devel/levelhq
git clone <openpi-repo-url> openpi   # if not already cloned
cd openpi
uv sync                               # installs all deps including openpi-client
```

### 2. Download the checkpoint

The `pi0_fast_droid` checkpoint (~10 GB) lives on Google Cloud Storage. If you have
`gcloud` credentials configured, the server will download it automatically. Otherwise,
download it manually with anonymous access:

```bash
cd /home/johnathan/devel/levelhq/openpi
uv run python -c "
import gcsfs, pathlib
fs = gcsfs.GCSFileSystem(token='anon')
dst = pathlib.Path.home() / '.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid'
dst.parent.mkdir(parents=True, exist_ok=True)
print('Downloading checkpoint (~10 GB)...')
fs.get('openpi-assets/checkpoints/pi0_fast_droid', str(dst), recursive=True)
print('Done!')
"
```

The checkpoint will be cached at `~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid/`.

### 3. Start the policy server (Terminal 1)

```bash
cd /home/johnathan/devel/levelhq/openpi

# Use CUDA_VISIBLE_DEVICES to select your GPU
CUDA_VISIBLE_DEVICES=3 uv run python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid
```

Wait for the server to print:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

The first inference request triggers JAX XLA compilation, which takes 1-2 minutes.
Subsequent requests are fast (~100ms).

### 4. Run the simulation client (Terminal 2)

**Headless** (saves video only, no GUI -- works on servers):

```bash
cd /home/johnathan/devel/levelhq/openpi

MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/main.py
```

**With live viewer** (shows the simulation on your monitor in real time):

```bash
cd /home/johnathan/devel/levelhq/openpi

DISPLAY=:0 uv run python examples/jaka_zu5_sim/main.py --args.display
```

> **Note:** The live viewer requires a physical monitor (or HDMI dummy plug) connected.
> Remote desktop software (NoMachine, VNC) without a physical display will fail with a
> GLFW monitor assertion error. If using remote desktop, enable the monitor first:
> `DISPLAY=:0 xrandr --output HDMI-2-0 --auto`

The client will:
1. Connect to the policy server on `ws://0.0.0.0:8000`
2. Run 600 steps (40 seconds at 15 Hz)
3. Save a video to `data/jaka_zu5_sim/videos/out_0.mp4`

#### CLI options

All options are prefixed with `--args.` (tyro convention):

| Flag | Default | Description |
|------|---------|-------------|
| `--args.prompt` | `"pick up the red cube"` | Language instruction sent to the policy |
| `--args.action-horizon` | `8` | Steps per action chunk (0.5s at 15 Hz) |
| `--args.host` | `0.0.0.0` | Policy server host |
| `--args.port` | `8000` | Policy server port |
| `--args.out-dir` | `data/jaka_zu5_sim/videos` | Video output directory |
| `--args.display` | `False` | Show live MuJoCo viewer window |

### 5. View the results

The output video is at `data/jaka_zu5_sim/videos/out_0.mp4`. Each subsequent run
increments the index (`out_1.mp4`, `out_2.mp4`, ...).

### 6. Interactive viewer (standalone, no policy server needed)

To explore the MuJoCo model interactively without running the policy server:

```bash
cd /home/johnathan/devel/levelhq/openpi

DISPLAY=:0 uv run python -m mujoco.viewer --mjcf=examples/jaka_zu5_sim/assets/jaka_zu5.xml
```

This opens an interactive window where you can:
- **Left-click + drag** to rotate the camera
- **Right-click + drag** to pan
- **Scroll** to zoom in/out
- **Double-click** a body to select it
- **Ctrl + right-click + drag** to apply forces to bodies
- **Space** to pause/unpause the simulation
- **Backspace** to reset to the initial state

This is useful for inspecting the robot model, verifying camera placements, and testing
the scene layout before running the full policy pipeline.

## File Reference

```
examples/jaka_zu5_sim/
  assets/
    jaka_zu5.xml      # MuJoCo MJCF model
  env.py              # Environment (renders cameras, reads joints, applies actions)
  main.py             # Entry point (wires Runtime, PolicyAgent, ActionChunkBroker)
  saver.py            # Video recorder subscriber
  requirements.txt    # Python dependencies
  README.md           # This file
```

### `assets/jaka_zu5.xml` -- MuJoCo Model

A procedurally-defined JAKA Zu5 6-DOF robot arm built from geometric primitives
(capsules, spheres, cylinders) matching the real kinematic structure. No external mesh
files are required.

Key dimensions are derived from the
[jaka_ros URDF](https://github.com/QiSheng918/jaka_ros/blob/master/jaka_description/urdf/jaka_description.urdf)
and manufacturer specs:

| Parameter | Value |
|-----------|-------|
| Upper arm (link2) | 360 mm |
| Forearm (link3) | 303 mm |
| Total reach | ~954 mm |
| DOF | 6 revolute + 1 prismatic gripper |

Joint limits from JAKA specifications:

| Joint | Range |
|-------|-------|
| J1, J5, J6 | +/- 360 deg |
| J2, J4 | -85 deg to +265 deg |
| J3 | +/- 175 deg |

The scene includes:
- Floor with checkerboard texture
- Table with red cube and green cylinder objects (free bodies)
- 2-finger parallel gripper with equality constraint for synchronized fingers
- `external_cam`: fixed 45-degree overhead view of the workspace
- `wrist_cam`: mounted on the gripper, looking down

### `env.py` -- JakaZu5SimEnvironment

Implements `openpi_client.runtime.environment.Environment` with 4 methods:

- **`reset()`** -- resets simulation to a home pose `[0, 0.5, -1.0, 0, 0.5, 0]`
- **`is_episode_complete()`** -- returns `True` after 600 steps (40s)
- **`get_observation()`** -- produces DROID-format observations:
  - `observation/exterior_image_1_left`: 224x224 uint8 HWC from `external_cam`
  - `observation/wrist_image_left`: 224x224 uint8 HWC from `wrist_cam`
  - `observation/joint_position`: float32 shape `[7]` (6 JAKA joints + zero-padded 7th)
  - `observation/gripper_position`: float32 shape `[1]` normalized to `[0, 1]`
  - `prompt`: language instruction string
- **`apply_action({"actions": ndarray[8]})`** -- interprets the 8D DROID action:
  - `actions[:6]`: joint velocity deltas, integrated into position targets
  - `actions[6]`: ignored (padded 7th DOF for Franka compatibility)
  - `actions[7]`: gripper command, binarized at 0.5 threshold
  - Runs 33 physics substeps per control step (0.002s timestep, ~15 Hz control)

### `main.py` -- Runtime Wiring

Composes the OpenPI client runtime:

```
Runtime (15 Hz)
  +-- JakaZu5SimEnvironment
  +-- PolicyAgent
  |     +-- ActionChunkBroker (horizon=8)
  |           +-- WebsocketClientPolicy (ws://host:port)
  +-- VideoSaver
```

`ActionChunkBroker` receives `[10, 8]` action chunks from the server and yields one
`[8]` action per step. After 8 steps it requests a new chunk.

### `saver.py` -- VideoSaver

Records `observation/exterior_image_1_left` from each step and writes an MP4 on episode
end. Images are already HWC uint8, so no transpose is needed (unlike the Aloha example
which uses CHW format).

## How It Works

### Observation Flow

```
MuJoCo sim
  |-- render external_cam (480x640) --> resize_with_pad(224,224) --> uint8 HWC
  |-- render wrist_cam (480x640)    --> resize_with_pad(224,224) --> uint8 HWC
  |-- read 6 joint positions        --> pad to 7 with zero
  |-- read gripper position          --> normalize to [0,1]
  |
  v
WebSocket --> Policy Server
               |-- DroidInputs transform:
               |     state = concat(joint_position[7], gripper_position[1]) -> [8]
               |     images = {base_0_rgb, base_1_rgb(zeros), wrist_0_rgb}
               |-- pi0_fast model inference (JAX/XLA on GPU)
               |-- DroidOutputs transform: actions[:, :8]
               |
               v
           actions [10, 8] --> WebSocket --> ActionChunkBroker
                                              |-- yields actions[i] for i in 0..7
                                              |
                                              v
                                          env.apply_action({"actions": [8]})
```

### 6-DOF to 8-DOF Mapping

The pi0_fast_droid model was trained on 7-DOF Franka Panda data (7 joints + 1 gripper = 8D actions).
The JAKA Zu5 is 6-DOF, so we handle the mismatch with zero-padding:

| Dimension | Franka (training) | JAKA (inference) |
|-----------|-------------------|------------------|
| 0-5 | Joint velocity 1-6 | Joint velocity 1-6 |
| 6 | Joint velocity 7 | Ignored (zero-padded in obs, discarded in action) |
| 7 | Gripper position | Gripper position |

## Known Limitations

- **Actions are not task-meaningful.** The model was trained on Franka Panda data, not
  JAKA Zu5. The arm will move but won't perform coherent tasks. Fine-tuning on JAKA
  data is required for real task execution.
- **Normalization mismatch.** DROID normalization statistics are computed over Franka
  joint ranges, which differ from JAKA ranges. This causes out-of-distribution inputs.
- **Geometric model only.** The MJCF uses capsule/sphere primitives, not real CAD meshes.
  Visual fidelity is low but sufficient for pipeline validation.
- **First inference is slow.** JAX XLA compilation on the first request takes 1-2 minutes.
  Subsequent requests are fast.
- **EGL cleanup warning.** A harmless `EGLError` prints at process exit on headless
  machines. This is a known MuJoCo/OpenGL teardown issue and can be safely ignored.

## Next Steps: Fine-Tuning for JAKA Zu5

To make the policy produce meaningful actions on the JAKA Zu5, you need to fine-tune
the pi0_fast model on JAKA-specific data. Here is the roadmap:

### 1. Collect Demonstration Data

Record teleoperation demonstrations on either the real JAKA Zu5 or in this MuJoCo
simulation. Each demonstration should capture:

- Two camera views (external + wrist) as 224x224 RGB images
- 6 joint positions + 1 gripper position at 15 Hz
- Language instruction for the task
- Joint velocity actions + gripper commands

Store in LeRobot HDF5 format (the format OpenPI training expects). See
`src/openpi/training/config.py` for the expected dataset structure.

### 2. Create a JAKA Training Config

Add a new config to `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi0_fast_jaka_zu5",
    model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="jaka_zu5"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
),
```

This reuses the `DroidInputs`/`DroidOutputs` transforms since the observation format
is identical. If the JAKA joint ranges differ significantly from Franka, you may want
to create custom input/output transforms with JAKA-specific normalization.

### 3. Compute Normalization Statistics

Generate norm stats for the JAKA dataset:

```bash
uv run python scripts/compute_norm_stats.py --config=pi0_fast_jaka_zu5 --data-dir=<path>
```

Place the output in `assets/pi0_fast_jaka_zu5/jaka_zu5/norm_stats.json`.

### 4. Fine-Tune

```bash
uv run python scripts/train.py \
    --config=pi0_fast_jaka_zu5 \
    --data-dir=<path-to-jaka-data> \
    --pretrained-dir=$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid
```

Fine-tuning from the DROID checkpoint (rather than training from scratch) leverages
the model's existing understanding of robot manipulation and should require fewer
demonstrations (50-100 demonstrations is a reasonable starting point).

### 5. Evaluate in Simulation

Serve the fine-tuned checkpoint and run this simulation:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_fast_jaka_zu5 \
    --policy.dir=<path-to-finetuned-checkpoint>

# Terminal 2
MUJOCO_GL=egl uv run python examples/jaka_zu5_sim/main.py \
    --args.prompt "pick up the red cube"
```

### 6. Deploy to Real Hardware

Once the fine-tuned model performs well in simulation:

1. Replace `env.py` with a real-robot environment that reads from JAKA SDK / ROS
2. The `main.py` runtime wiring stays the same -- only the `Environment` subclass changes
3. Consider adding ROS/Gazebo integration for a higher-fidelity sim-to-real bridge

### Key Fine-Tuning Considerations

- **Joint range normalization**: JAKA joint ranges (-360 to +360 deg for J1/J5/J6, -85 to
  +265 deg for J2/J4) differ from Franka. Custom norm stats are critical.
- **Action space**: DROID uses joint velocity actions. Ensure your demonstration data
  records velocity commands, not position targets.
- **7th DOF padding**: The zero-padded 7th joint in observations may confuse the model
  initially. Including this padding consistently in training data helps.
- **Camera placement**: Match the simulation camera positions to your real setup as closely
  as possible. The model is sensitive to viewpoint distribution shift.
- **Gripper mapping**: The JAKA gripper likely has different stroke/force characteristics
  than the Franka. Calibrate the gripper normalization to `[0, 1]` range.
