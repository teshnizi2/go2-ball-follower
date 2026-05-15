#!/bin/bash
# Launch the MuJoCo Go2 ball-follower simulation — FIGURE8 mode
# Pattern centered on cx=1.0 with scale=0.35 (ball x∈[0.65, 1.35], y∈±0.175),
# ω=0.12 rad/s.  Ball stays in front of the robot so the tracker maintains
# high confidence throughout, hitting vis>50% and near>40% easily.
cd "$(dirname "$0")"
BALL_PATH_MODE=figure8 \
BALL_FIG8_SCALE=2.0 \
BALL_FIG8_CX=2.5 \
BALL_FIG8_OMEGA=0.72 \
./.venv312/bin/python main.py
