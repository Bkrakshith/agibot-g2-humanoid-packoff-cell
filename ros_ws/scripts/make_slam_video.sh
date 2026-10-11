#!/usr/bin/env bash
# Compose output/renders/g2_slam_nav.mp4 from the recorded SLAM / Nav2 runs (run inside the container):
# free camera (record_cams, 6 fps) | map frames (map_frames.py, 6 fps), sped up, then concatenated.
# Usage: bash make_slam_video.sh [dir=/g2cell_output/slam] [out=/g2cell_output/renders/g2_slam_nav.mp4]
set -euo pipefail
D=${1:-/g2cell_output/slam}; OUT=${2:-/g2cell_output/renders/g2_slam_nav.mp4}
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
cd "$D"
seg() { # freecam.mp4 frames_dir speed label out
  printf '%s' "$4" > "$5.txt"
  ffmpeg -y -loglevel error -i "$1" -framerate 6 -i "$2/f_%05d.png" -filter_complex \
    "[0:v]scale=960:540,setpts=PTS/$3,drawtext=fontfile=$FONT:textfile=$5.txt:x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6[a];[1:v]setpts=PTS/$3[b];[a][b]hstack=shortest=1,fps=24,format=yuv420p" \
    -c:v libx264 -crf 22 -preset medium "$5"
  rm -f "$5.txt"
}
seg mapping_freecam_raw.mp4 frames_map 4.5 "1/2  SLAM mapping: slam_toolbox, scripted loop (x4.5)" seg1.mp4
seg nav2_freecam_raw.mp4 frames_nav2 2.5 "2/2  Nav2 NavigateToPose around the cell (x2.5)" seg2.mp4
ffmpeg -y -loglevel error -i seg1.mp4 -i seg2.mp4 -filter_complex "[0:v][1:v]concat=n=2:v=1[v]" -map "[v]" \
  -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart "$OUT"
rm -f seg1.mp4 seg2.mp4
ffprobe -v error -show_entries format=duration:stream=width,height,codec_name,pix_fmt -of compact "$OUT"
