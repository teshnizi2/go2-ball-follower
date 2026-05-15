#!/bin/bash
# Bench corridor mode headless.
cd "$(dirname "$0")"
SECS="${1:-60}"
PY="./.venv312/bin/python"
rm -f /tmp/sim_corridor.log /tmp/sim_corridor_done

GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
  BALL_PATH_MODE=corridor BALL_CORRIDOR_SPEED="${2:-0.10}" \
  "$PY" main.py > /tmp/sim_corridor.log 2>&1

touch /tmp/sim_corridor_done
echo "=== corridor bench done ==="
