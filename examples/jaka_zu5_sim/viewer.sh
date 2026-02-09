#!/bin/bash
# Launch MuJoCo viewer with the home keyframe applied.
# Usage: DISPLAY=:0 ./examples/jaka_zu5_sim/viewer.sh

cd "$(dirname "$0")/../.."

DISPLAY="${DISPLAY:-:0}" uv run python -c "
import mujoco, mujoco.viewer
model = mujoco.MjModel.from_xml_path('examples/jaka_zu5_sim/assets/jaka_zu5.xml')
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.viewer.launch(model, data)
"
