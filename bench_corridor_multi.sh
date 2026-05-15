#!/bin/bash
# Bench corridor across multiple random seeds.
cd "$(dirname "$0")"
SECS="${1:-180}"
SPEED="${2:-0.10}"
PY="./.venv312/bin/python"

for SEED in 1 2 3 4 5; do
  LOG="/tmp/sim_corridor_s${SEED}.log"
  rm -f "$LOG"
  echo "=== SEED=$SEED ==="
  GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
    BALL_PATH_MODE=corridor BALL_CORRIDOR_SPEED="$SPEED" \
    CORRIDOR_SEED="$SEED" \
    "$PY" main.py > "$LOG" 2>&1
  "$PY" analyze_corridor.py "$LOG"
  echo ""
done

touch /tmp/sim_corridor_multi_done
echo "=== multi-seed bench done ==="
