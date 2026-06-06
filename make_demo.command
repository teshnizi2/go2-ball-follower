#!/bin/bash
# Reliable demo generator — renders the chase view headlessly (no window),
# so it ALWAYS produces a saved video showing the robot navigating the
# endless, always-visible obstacle corridor.  Edit SECONDS for a longer clip.
cd "$(dirname "$0")"
OUT="$HOME/Desktop/go2_demo.mp4"
RAW="/tmp/go2_demo_chase.mp4"
SECONDS_SIM="${1:-700}"     # sim seconds (~90 s of video per 700) — pass an arg to change
echo "===> Rendering ${SECONDS_SIM}s of sim → chase video (headless, fast)..."
GO2_HEADLESS=1 RECORD_CHASE="$RAW" RECORD_FPS=30 RENDER_SKIP=13 \
GO2_HEADLESS_SECONDS="$SECONDS_SIM" GO2_SHADOWS=0 \
BALL_PATH_MODE=corridor COLLISION_STOPS_SIM=0 CORRIDOR_ENDLESS=1 \
ACTION_SCALE_MULT=2.0 ROBOT_SAFETY=0.50 DETOUR_MARGIN=0.50 OBS_BRAKE_LATERAL=0.85 \
CORRIDOR_SEED="${CORRIDOR_SEED:-700}" OBS_ROWS=50 OBS_MIN_GAP=5.4 OBS_MIN_X=4.0 OBS_MAX_X=275.0 \
BALL_CORRIDOR_SPEED=0.111 BALL_MAX_LEAD=4.0 BALL_LANE_DWELL=15.0 \
PYTHONUNBUFFERED=1 ./.venv312/bin/python main.py
echo "===> Encoding → $OUT"
if ffmpeg -y -loglevel error -i "$RAW" -c:v libx264 -crf 20 -pix_fmt yuv420p "$OUT" && [ -s "$OUT" ]; then
  echo "===> Saved (H.264): $OUT"
else
  cp -f "$RAW" "$OUT"; echo "===> Saved (raw): $OUT"
fi
open "$OUT" 2>/dev/null || true
