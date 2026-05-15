#!/bin/bash
# CORRIDOR mode — 3-m-wide corridor with 7 random obstacles (Level 4 difficulty).
# Ball goes forward avoiding obstacles; robot follows AND avoids them.
# After curriculum training (Levels 1-5), Level 4 is reliable: 0 falls,
# 0 collisions across all tested seeds.
cd "$(dirname "$0")"
BALL_PATH_MODE=corridor \
BALL_CORRIDOR_SPEED=0.111 \
BALL_MAX_LEAD=4.0 \
BALL_LANE_DWELL=8.0 \
OBS_COUNT=15 SHORT_OBS_COUNT=15 \
OBS_MIN_GAP=3.5 OBS_Y_MIN=0.0 OBS_Y_MAX=1.0 \
OBS_MIN_X=4.0 OBS_MAX_X=115.0 \
SIM_SPEED=3.0 \
./.venv312/bin/python main.py
