#!/bin/bash

for file in palestinian_videos/*.webm; do
  if [ -f "$file" ]; then
    filename=$(basename "$file" .webm)
    ffmpeg -i "$file" -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 192k "palestinian_videos2/${filename}.mp4"
  fi
done
