#!/usr/bin/env bash
# Live RViz replay of an episode (solid = recorded joint_states, ghost = policy prediction).
#   tools/viz/replay_rviz.sh <episode.hdf5> [--checkpoint RUN_DIR|ckpt.pt] [--speed 1.0]
#                            [--ghost-offset 0.0] [--no-loop] [--no-rviz] [--npz-only]
# 1) export_replay.py (.venv-learn, CPU) -> output/viz/<episode>[__<run>].npz
# 2) replay.launch.py in a clean env (conda libs break ROS 2 Humble), ROS_DOMAIN_ID=7, DISPLAY :1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$HERE/../.." && pwd)
WS=$(cd "$PROJ/.." && pwd)
EP=""; CKPT=""; SPEED=1.0; GHOST_DY=0.0; LOOP=true; RVIZ=true; NPZ_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT=$2; shift 2;;
    --speed) SPEED=$2; shift 2;;
    --ghost-offset) GHOST_DY=$2; shift 2;;
    --no-loop) LOOP=false; shift;;
    --no-rviz) RVIZ=false; shift;;
    --npz-only) NPZ_ONLY=1; shift;;
    -h|--help) sed -n 2,6p "$0"; exit 0;;
    *) EP=$1; shift;;
  esac
done
[ -n "$EP" ] || { sed -n 2,6p "$0"; exit 2; }
EP=$(realpath "$EP")
STEM=$(basename "$EP" .hdf5)
OUT="$PROJ/output/viz"; mkdir -p "$OUT"
if [ -n "$CKPT" ]; then
  CKPT=$(realpath "$CKPT")
  NPZ="$OUT/${STEM}__$(basename "${CKPT%.pt}").npz"
  PY="$WS/.venv-learn/bin/python"
  CK_ARGS=(--checkpoint "$CKPT")
else
  NPZ="$OUT/${STEM}.npz"
  PY="$WS/.venv-learn/bin/python"; [ -x "$PY" ] || PY="$WS/.venv-tools/bin/python"
  CK_ARGS=()
fi
echo "[replay] exporting $EP -> $NPZ"
(cd "$PROJ" && CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/export_replay.py" "$EP" -o "$NPZ" "${CK_ARGS[@]}")
[ "$NPZ_ONLY" = 1 ] && exit 0
U=$(id -u)
exec env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin DISPLAY="${REPLAY_DISPLAY:-:1}" \
  XAUTHORITY="${REPLAY_XAUTHORITY:-/run/user/$U/gdm/Xauthority}" XDG_RUNTIME_DIR="/run/user/$U" \
  ROS_DOMAIN_ID=7 bash --noprofile --norc -c "source /opt/ros/humble/setup.bash && \
  source '$WS/g2_viz_ws/install/setup.bash' && \
  exec ros2 launch '$HERE/replay.launch.py' npz:='$NPZ' speed:=$SPEED loop:=$LOOP rviz:=$RVIZ ghost_offset_y:=$GHOST_DY"
