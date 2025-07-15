#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found. Please, use: sudo apt install ffmpeg"
    exit 1
fi

# Create folder if not exists
OUTPUT_DIR="outputs/video_trials"
mkdir -p "$OUTPUT_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Compose video filename
VIDEO_NAME="video_${TIMESTAMP}.mp4"

# Make the video
ffmpeg -framerate 30 -i outputs/frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p "${OUTPUT_DIR}/${VIDEO_NAME}.mp4"

echo "Video created: ${OUTPUT_DIR}/${VIDEO_NAME}"
