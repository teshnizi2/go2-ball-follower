#!/bin/bash
# CORRIDOR recording — runs until you press Q in the sim window.
# RENDER_SKIP=33 → each video frame = 33 physics steps = 19.8× sim speed.
# No time limit.  Press Q to stop; the video is saved automatically.
cd "$(dirname "$0")"

OUT_RAW="/tmp/go2_corridor_recording.mp4"
OUT_FINAL="$HOME/Desktop/go2_demo_1min.mp4"
rm -f "$OUT_RAW" "$OUT_FINAL"

echo "===> Recording at ~8× sim speed — press Q in the window to stop."
PYTHONUNBUFFERED=1 \
GO2_GUI_VIEW=chase \
GO2_SHADOWS=1 \
RECORD_VIDEO="$OUT_RAW" \
RECORD_FPS=30 \
BALL_PATH_MODE=corridor \
BALL_CORRIDOR_SPEED=0.111 \
BALL_MAX_LEAD=4.0 \
BALL_LANE_DWELL=15.0 \
OBS_ROWS=50 \
OBS_MIN_GAP=5.4 \
OBS_MIN_X=4.0 OBS_MAX_X=275.0 \
SIM_SPEED=0.75 \
RENDER_SKIP=13 \
CORRIDOR_SEED=700 \
COLLISION_STOPS_SIM=0 \
CORRIDOR_ENDLESS=1 \
ACTION_SCALE_MULT=2.0 \
ROBOT_SAFETY=0.50 \
DETOUR_MARGIN=0.50 \
OBS_BRAKE_LATERAL=0.85 \
DETOUR_LOOKAHEAD=8.0 \
VY_LATERAL=1 \
MIN_PASSAGE=1.45 \
./.venv312/bin/python main.py

echo "===> Sim stopped. Re-encoding..."
RAW_DUR=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$OUT_RAW" 2>/dev/null)
echo "===> raw_dur=${RAW_DUR}s"
# Robust save: re-encode to H.264 for compatibility; if that fails for ANY
# reason, fall back to copying the raw clip so a video is ALWAYS produced
# (the old script printed "Saved" even when ffmpeg silently failed).
if ffmpeg -y -loglevel error -i "$OUT_RAW" \
        -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p "$OUT_FINAL" \
        && [ -s "$OUT_FINAL" ]; then
    echo "===> Saved (H.264): $OUT_FINAL"
else
    echo "===> Re-encode failed — saving the raw clip instead."
    cp -f "$OUT_RAW" "$OUT_FINAL"
    echo "===> Saved (raw mp4v): $OUT_FINAL"
fi
echo "===> Opening..."
open "$OUT_FINAL" 2>/dev/null || true
