#!/bin/bash
# Launch the MuJoCo Go2 ball-follower simulation — SQUARE mode
# Ball traces a rectangle placed in front of the robot (x > 0): corners at
# (±half*1.5, ±half) with half=0.85, so ball distance ranges 0.85 m (back
# corners) to 1.65 m (front-back diagonal).  Ball stays within ±~40° of the
# robot's +X heading, which keeps the tracker's confidence high throughout.
cd "$(dirname "$0")"
BALL_PATH_MODE=square \
BALL_SQUARE_HALF=3.0 \
BALL_SQUARE_SPEED=0.18 \
BALL_CORNER_WAIT=2.0 \
./.venv312/bin/python main.py
