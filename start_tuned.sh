#!/bin/bash
cd "$(dirname "$0")"
chmod +x ./bench_tuned.sh
rm -f /tmp/sim_tuned_*.log /tmp/sim_tuned_done
nohup ./bench_tuned.sh "${1:-45}" >/tmp/tuned_meta.log 2>&1 &
disown
echo "STARTED $!"
