#!/bin/bash
# CIRCLE mode — big arena, 6× speed.
# Ball orbits origin at r=3.5 m, ω=0.18 rad/s (tangential = 0.63 m/s).
cd "$(dirname "$0")"
BALL_PATH_MODE=circle \
BALL_CIRCLE_RADIUS=3.5 \
BALL_CIRCLE_OMEGA=0.18 \
./.venv312/bin/python main.py
