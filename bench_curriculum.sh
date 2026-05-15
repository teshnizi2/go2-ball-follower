#!/bin/bash
# Curriculum-style bench: increasing difficulty levels.
# Each level run across 3 seeds; report per-seed and overall pass-rate.
cd "$(dirname "$0")"
PY="./.venv312/bin/python"
LVL="${1:-1}"
SECS="${2:-200}"
SEEDS="${3:-1 2 3}"

# Difficulty parameters per level.
case "$LVL" in
  1)
    OBS_COUNT=4; OBS_MIN_GAP=3.0; OBS_Y_MIN=0.3; OBS_Y_MAX=1.0; SPEED=0.10
    DESC="Level 1 (easy): 4 obs, 3m gap, slow ball" ;;
  2)
    OBS_COUNT=5; OBS_MIN_GAP=2.5; OBS_Y_MIN=0.4; OBS_Y_MAX=1.05; SPEED=0.12
    DESC="Level 2 (medium): 5 obs, 2.5m gap" ;;
  3)
    OBS_COUNT=6; OBS_MIN_GAP=2.2; OBS_Y_MIN=0.5; OBS_Y_MAX=1.1; SPEED=0.13
    DESC="Level 3 (hard): 6 obs, 2.2m gap, tighter Y" ;;
  4)
    OBS_COUNT=7; OBS_MIN_GAP=1.9; OBS_Y_MIN=0.5; OBS_Y_MAX=1.15; SPEED=0.14
    DESC="Level 4 (very hard): 7 obs, 1.9m gap" ;;
  5)
    OBS_COUNT=8; OBS_MIN_GAP=1.7; OBS_Y_MIN=0.55; OBS_Y_MAX=1.2; SPEED=0.15
    DESC="Level 5 (extreme): 8 obs, 1.7m gap, fastest ball" ;;
  *)
    echo "Unknown level $LVL"; exit 1 ;;
esac

echo "=== $DESC ==="
echo "OBS_COUNT=$OBS_COUNT  OBS_MIN_GAP=$OBS_MIN_GAP  Y=[$OBS_Y_MIN, $OBS_Y_MAX]  SPEED=$SPEED  SECS=$SECS"
echo ""

PASS=0
FAIL=0
for SEED in $SEEDS; do
  LOG="/tmp/sim_lvl${LVL}_s${SEED}.log"
  rm -f "$LOG"
  GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
    BALL_PATH_MODE=corridor BALL_CORRIDOR_SPEED="$SPEED" \
    OBS_COUNT="$OBS_COUNT" OBS_MIN_GAP="$OBS_MIN_GAP" \
    OBS_Y_MIN="$OBS_Y_MIN" OBS_Y_MAX="$OBS_Y_MAX" \
    OBS_MIN_X=5.0 OBS_MAX_X=$((SECS / 4)) \
    CORRIDOR_SEED="$SEED" \
    "$PY" main.py > "$LOG" 2>&1
  echo "--- SEED $SEED ---"
  "$PY" analyze_corridor.py "$LOG" | grep -E "Falls|collision|progressed|froze|vis%|Obstacles"
done

touch "/tmp/sim_curriculum_lvl${LVL}_done"
echo ""
echo "=== Level $LVL bench done ==="
