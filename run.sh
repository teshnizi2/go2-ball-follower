#!/usr/bin/env bash
# run.sh — launch the Go2 corridor simulation.
#
# Usage:
#   ./run.sh             — launch GUI with best known config
#   ./run.sh --tune      — run self_tune.py first, then launch GUI
#   ./run.sh --headless  — headless run (for testing)
#
# After a collision the sim prints [RESULT] FAIL; run with --tune
# to let the system improve itself automatically.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="$SCRIPT_DIR/.venv312/bin/python3.12"
MAIN="$SCRIPT_DIR/main.py"
CONFIG="$SCRIPT_DIR/best_config.json"
TUNE_SCRIPT="$SCRIPT_DIR/self_tune.py"

DO_TUNE=0
DO_HEADLESS=0

for arg in "$@"; do
  case "$arg" in
    --tune)     DO_TUNE=1 ;;
    --headless) DO_HEADLESS=1 ;;
  esac
done

# ── auto-tune if requested ────────────────────────────────────────────────────
if [ "$DO_TUNE" = "1" ]; then
  echo "[run.sh] Running self-tuner (3 seeds × 30 rows)..."
  "$PYTHON" "$TUNE_SCRIPT" --seeds 3 --rows 30
fi

# ── load best_config.json into env vars ──────────────────────────────────────
if [ -f "$CONFIG" ]; then
  A=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d['config']['ACTION_SCALE_MULT'])")
  S=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d['config']['ROBOT_SAFETY'])")
  D=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d['config']['DETOUR_MARGIN'])")
  L=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d['config']['OBS_BRAKE_LATERAL'])")
  echo "[run.sh] Loaded config: A=$A S=$S D=$D L=$L"
else
  # Validated 3/3 winner from grid search (fast+reliable variant)
  A=2.0; S=0.75; D=0.50; L=0.85
  echo "[run.sh] No saved config — using default: A=$A S=$S D=$D L=$L"
fi

export BALL_PATH_MODE=corridor   # always run in corridor/obstacle mode
export ACTION_SCALE_MULT="$A"
export ROBOT_SAFETY="$S"
export DETOUR_MARGIN="$D"
export OBS_BRAKE_LATERAL="$L"

# ── speed defaults (override via env before calling run.sh) ──────────────────
export SIM_SPEED="${SIM_SPEED:-3.0}"      # run ~3× faster than real-time
export RENDER_SKIP="${RENDER_SKIP:-4}"    # render every 4th step (~12 fps display)
export GO2_SHADOWS="${GO2_SHADOWS:-0}"    # shadows OFF for speed; set 1 for recordings

if [ "$DO_HEADLESS" = "1" ]; then
  export GO2_HEADLESS=1
fi

exec "$PYTHON" "$MAIN"
