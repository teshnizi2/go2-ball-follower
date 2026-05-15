#!/bin/bash
# Bench big-arena + 6x-speed config.
# Ball orbits at 6× speed with radius > 3 m — robot must actively chase.
# Note: the RL policy's actual max forward speed is 0.148 m/s (measured), while
# ball tangential speed at r=3.5 × ω=0.18 = 0.63 m/s.  The robot is ~4× slower
# than the ball, so near-goal (0.7-1.3 m) cannot be physically maintained
# without retraining the locomotion policy for higher authority.
cd "$(dirname "$0")"
SECS="${1:-45}"
PY="./.venv312/bin/python"
rm -f /tmp/sim_tuned_*.log /tmp/sim_tuned_done

# Circle: r=3.5, ω=0.18 rad/s (6× speed, tangential = 0.63 m/s).
GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
  BALL_PATH_MODE=circle BALL_CIRCLE_RADIUS=3.5 BALL_CIRCLE_OMEGA=0.18 \
  "$PY" main.py > /tmp/sim_tuned_circle.log 2>&1

# Square: half=3.0 (corners at (±3, ±3) ≈ 4.24 m from origin), edge_speed=0.18 m/s.
GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
  BALL_PATH_MODE=square BALL_SQUARE_HALF=3.0 BALL_SQUARE_SPEED=0.18 BALL_CORNER_WAIT=2.0 \
  "$PY" main.py > /tmp/sim_tuned_square.log 2>&1

# Figure8: scale=2.0, cx=2.5, ω=0.72 rad/s.  Ball ranges x∈[0.5, 4.5], y∈±1.0.
GO2_HEADLESS=1 GO2_HEADLESS_SECONDS="$SECS" \
  BALL_PATH_MODE=figure8 BALL_FIG8_SCALE=2.0 BALL_FIG8_CX=2.5 BALL_FIG8_OMEGA=0.72 \
  "$PY" main.py > /tmp/sim_tuned_figure8.log 2>&1

touch /tmp/sim_tuned_done
echo "=== tuned bench done ==="
