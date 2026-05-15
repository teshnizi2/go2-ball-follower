#!/bin/bash
cd "$(dirname "$0")"
chmod +x ./bench_speed_sweep.sh
rm -f /tmp/sim_sweep_*.log /tmp/sim_sweep_done /tmp/sweep_meta.log
nohup ./bench_speed_sweep.sh "${1:-45}" >/tmp/sweep_meta.log 2>&1 &
PID=$!
disown
echo "STARTED $PID"
