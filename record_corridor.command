#!/bin/bash
# CORRIDOR — 3rd-person chase camera fullscreen recording.  Outputs 10x video.
cd "$(dirname "$0")"

OUT_RAW="/tmp/go2_corridor_recording.mp4"
OUT_10X="/tmp/go2_corridor_recording_10x.mp4"
DESKTOP_RAW="$HOME/Desktop/go2_corridor_recording.mp4"
DESKTOP_10X="$HOME/Desktop/go2_corridor_recording_10x.mp4"
rm -f "$OUT_RAW" "$OUT_10X"

WALL_DUR=3000   # 50-min wall — enough to traverse 50 rows × 2 = 100 obstacles
echo "===> Starting ${WALL_DUR}s recording (3rd-person chase view)..."
PYTHONUNBUFFERED=1 \
GO2_GUI_VIEW=chase \
RECORD_VIDEO="$OUT_RAW" \
RECORD_DURATION_S="$WALL_DUR" \
RECORD_FPS=30 \
BALL_PATH_MODE=corridor \
BALL_CORRIDOR_SPEED=0.111 \
BALL_MAX_LEAD=4.0 \
BALL_LANE_DWELL=8.0 \
OBS_ROWS=50 \
OBS_MIN_GAP=5.4 \
OBS_MIN_X=4.0 OBS_MAX_X=275.0 \
SIM_SPEED=3.0 \
RENDER_SKIP=8 \
./.venv312/bin/python main.py

echo "===> Recording done. Producing 10x video..."
TARGET_DUR=$(awk "BEGIN { printf \"%.4f\", ${WALL_DUR}/10.0 }")
RAW_DUR=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$OUT_RAW")
PTS_FACTOR=$(awk "BEGIN { printf \"%.6f\", ${TARGET_DUR}/${RAW_DUR} }")
echo "===> raw_dur=${RAW_DUR}s target_out=${TARGET_DUR}s pts_factor=${PTS_FACTOR}"
ffmpeg -y -loglevel error -i "$OUT_RAW" \
    -filter:v "setpts=${PTS_FACTOR}*PTS,fps=30" -an "$OUT_10X"

echo "===> Copying videos to Desktop..."
cp "$OUT_RAW" "$DESKTOP_RAW"
cp "$OUT_10X" "$DESKTOP_10X"
echo "===> Saved:"
echo "      $DESKTOP_RAW   (raw)"
echo "      $DESKTOP_10X   (10x compressed, ${TARGET_DUR}s)"
echo "===> Opening 10x in default player..."
open "$DESKTOP_10X"
