#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found. Please, use: sudo apt install ffmpeg"
    exit 1
fi

# Default output dir
OUTPUT_DIR="outputs/video_trials"
mkdir -p "$OUTPUT_DIR"

# If output filename provided, use it; else generate one
if [ -n "$1" ]; then
    OUTPUT_PATH="$1"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_PATH="${OUTPUT_DIR}/video_${TIMESTAMP}.mp4"
fi

# Make the video
ffmpeg -y -framerate 30 -i outputs/frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p "$OUTPUT_PATH"

echo "Video created: $OUTPUT_PATH"
