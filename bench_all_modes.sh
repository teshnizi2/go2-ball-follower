#!/bin/bash
# Run headless benchmarks for all 3 ball-path modes sequentially.
# Usage: ./bench_all_modes.sh [seconds]
cd "$(dirname "$0")"
SECS="${1:-60}"
PY="./.venv312/bin/python"
rm -f /tmp/sim_bench_*.log /tmp/sim_bench_done
for mode in circle square figure8; do
  echo "=== Running $mode for ${SECS}s ==="
  GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" BALL_PATH_MODE="$mode" \
    "$PY" main.py > "/tmp/sim_bench_${mode}.log" 2>&1
  # Write a small summary for each mode as we finish
  {
    echo "--- $mode summary ---"
    grep -E 'FELL|near-goal|Mean distance|vis_frac|Vx active|vx active|CSV saved|RESET' "/tmp/sim_bench_${mode}.log" | head -20
    echo
  } >> /tmp/sim_bench_summary.txt
done
touch /tmp/sim_bench_done
echo "=== all modes done ==="
