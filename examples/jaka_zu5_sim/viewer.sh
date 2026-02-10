#!/bin/bash
# Launch interactive MuJoCo viewer with keyboard jog controls.
# Usage: DISPLAY=:0 ./examples/jaka_zu5_sim/viewer.sh

cd "$(dirname "$0")/../.."

DISPLAY="${DISPLAY:-:0}" uv run python examples/jaka_zu5_sim/viewer.py
