#!/bin/bash
# Sweep ball speed multipliers (2x, 3x, 4x of original baselines) across modes.
# Usage: ./bench_speed_sweep.sh [seconds]
cd "$(dirname "$0")"
SECS="${1:-60}"
PY="./.venv312/bin/python"
rm -f /tmp/sim_sweep_*.log /tmp/sim_sweep_done
for mul in 2 3 4; do
  # Scale from current "1x" baselines: circle ω=0.065, square=0.10, fig8 ω=0.055
  C_W=$(python3 -c "print(0.065*${mul})")
  S_V=$(python3 -c "print(0.10*${mul})")
  F_W=$(python3 -c "print(0.055*${mul})")
  for mode in circle square figure8; do
    tag="${mode}_${mul}x"
    echo "=== $tag (circle_w=$C_W  square_v=$S_V  fig8_w=$F_W) ==="
    GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
      BALL_PATH_MODE="$mode" \
      BALL_CIRCLE_OMEGA="$C_W" BALL_SQUARE_SPEED="$S_V" BALL_FIG8_OMEGA="$F_W" \
      "$PY" main.py > "/tmp/sim_sweep_${tag}.log" 2>&1
  done
done
touch /tmp/sim_sweep_done
echo "=== sweep done ==="
