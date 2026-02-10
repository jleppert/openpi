#!/bin/bash
set -e

VID="${1:-data/jaka_zu5_sim/eval_videos_wide}"
GRID="${2:-4}"  # grid size (4 = 4x4)
FPS="${3:-5}"   # output fps (lower = slower playback)
OUT="data/jaka_zu5_sim/montage_${GRID}x${GRID}.mp4"

N=$((GRID * GRID))

# Pick N success videos, evenly spaced.
mapfile -t ALL < <(ls "$VID"/*success*.mp4 2>/dev/null | sort)
TOTAL=${#ALL[@]}

if [ "$TOTAL" -lt "$N" ]; then
    echo "Only $TOTAL success videos, need $N for ${GRID}x${GRID} grid"
    exit 1
fi

# Evenly sample N from TOTAL
INPUTS=""
SCALES=""
LABELS=""
LAYOUT=""
W=320
H=240

for i in $(seq 0 $((N - 1))); do
    idx=$(( i * TOTAL / N ))
    INPUTS="$INPUTS -i ${ALL[$idx]}"
    SCALES="${SCALES}[$i:v]scale=${W}:${H},setpts=PTS-STARTPTS[v$i];"
    LABELS="${LABELS}[v$i]"
    col=$((i % GRID))
    row=$((i / GRID))
    [ -n "$LAYOUT" ] && LAYOUT="${LAYOUT}|"
    LAYOUT="${LAYOUT}$((col * W))_$((row * H))"
done

ffmpeg -y $INPUTS \
    -filter_complex "${SCALES}${LABELS}xstack=inputs=${N}:layout=${LAYOUT}[out]" \
    -map "[out]" -c:v libx264 -crf 20 -r "$FPS" -pix_fmt yuv420p "$OUT"

echo "Done: $OUT (${GRID}x${GRID}, $N/${TOTAL} success videos from $VID, ${FPS}fps)"
