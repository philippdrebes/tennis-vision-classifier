#!/bin/bash

# Directory containing your video files
VIDEO_DIR="../video"

# Directory where you want to save the extracted frames
OUTPUT_DIR="../video/frames"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each video file in the video directory
for video in "$VIDEO_DIR"/*.mp4; do
  VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/$(basename "$video" .mp4)"

  # Check if VIDEO_OUTPUT_DIR already exists
  if [ -d "$VIDEO_OUTPUT_DIR" ]; then
    echo "Directory $VIDEO_OUTPUT_DIR already exists. Skipping..."
    continue # Skip to the next video file
  fi

  mkdir -p "$VIDEO_OUTPUT_DIR"
  # Use FFmpeg to extract one frame per second
  ffmpeg -i "$video" -vf "fps=1" "$VIDEO_OUTPUT_DIR/$(basename "$video" .mp4)_frame_%04d.jpg"
done

