#!/usr/bin/env bash
# Record one teleop episode to MCAP, then convert it to HDF5 when you press Ctrl+C.
#
#   ros2 run g2cell_recorder record_episode.sh <name> [--scene S] [--seed N] [--rate HZ]
#        [--duration SEC] [--no-convert] [--dry-run]
#        [--rgb compressed|zstd|raw] [--depth compressedDepth|zstd|raw|off]
#        [--depth-rate HZ|full] [--free-cam] [--compression zstd_fast|zstd_small|none]
#        [--full-fidelity] [-- bag_to_hdf5 args]
#
# Defaults are the cheap profile (~1.4 MB/s instead of ~78 MB/s; see docs/OPTIMISATION.md):
#   head RGB over image_transport "compressed" (JPEG q95), head depth over "compressedDepth"
#   (lossless PNG of quantised inverse depth, <1 mm error below 3 m) throttled to 5 Hz by
#   throttle_relay, no free camera, joint_states/joint_command/tf at full rate, and MCAP
#   chunk compression (zstd_fast). --full-fidelity = --rgb raw --depth raw --depth-rate full
#   --free-cam (lossless; chunk zstd stays on unless --compression none).
# --duration stops by itself after SEC seconds (for scripted runs without a terminal).
# --dry-run prints TOPICS=, RELAY=, CMD= and exits without touching ROS or the disk.
# Env: EPISODES_DIR (default /g2cell_output/episodes), EXTRA_TOPICS (space-separated).
set -euo pipefail

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
die() { echo "record_episode: $*" >&2; exit 2; }

[[ $# -ge 1 && $1 != -* ]] || usage
NAME=$1; shift
SCENE="" SEED=-1 RATE=30 DURATION="" CONVERT=1 DRY=0 EXTRA=()
RGB=compressed DEPTH_TR=compressedDepth DEPTH_RATE=5 FREE_CAM=0 BAGCOMP=zstd_fast
while [[ $# -gt 0 ]]; do
  case $1 in
    --scene) SCENE=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --rate) RATE=$2; shift 2 ;;
    --duration) DURATION=$2; shift 2 ;;
    --no-convert) CONVERT=0; shift ;;
    --dry-run) DRY=1; shift ;;
    --rgb) RGB=$2; shift 2 ;;
    --depth) DEPTH_TR=$2; shift 2 ;;
    --depth-rate) DEPTH_RATE=$2; shift 2 ;;
    --free-cam) FREE_CAM=1; shift ;;
    --compression) BAGCOMP=$2; shift 2 ;;
    --full-fidelity) RGB=raw DEPTH_TR=raw DEPTH_RATE=full FREE_CAM=1; shift ;;
    -h|--help) usage 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done

case $RGB in compressed|zstd|raw) ;; *) die "--rgb must be compressed, zstd or raw" ;; esac
case $DEPTH_TR in compressedDepth|zstd|raw|off) ;;
  *) die "--depth must be compressedDepth, zstd, raw or off" ;; esac
case $BAGCOMP in zstd_fast|zstd_small|none) ;;
  *) die "--compression must be zstd_fast, zstd_small or none" ;; esac
[[ $DEPTH_RATE == full || $DEPTH_RATE =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "--depth-rate must be a number (Hz) or 'full'"

suffix() { [[ $1 == raw ]] && echo "" || echo "/$1"; }   # image_transport topic suffix
HEAD=/genie_sim/head_front_camera_rgb/image_raw
FREE=/genie_sim/free_camera_rgb/image_raw
DEPTH=/genie_sim/head_front_camera_depth/image_raw
RELAY_PREFIX=/g2cell/rec/head_front_camera_depth/

TOPICS=(/joint_states /joint_command /tf /tf_static "$HEAD$(suffix "$RGB")")
[[ $FREE_CAM -eq 1 ]] && TOPICS+=("$FREE$(suffix "$RGB")")
RELAY_IN="" RELAY_OUT="" RELAY_TYPE=""
if [[ $DEPTH_TR != off ]]; then
  DEPTH_TOPIC=$DEPTH$(suffix "$DEPTH_TR")
  if [[ $DEPTH_RATE == full ]]; then
    TOPICS+=("$DEPTH_TOPIC")
  else
    RELAY_IN=$DEPTH_TOPIC
    if [[ $DEPTH_TR == raw ]]; then
      RELAY_OUT=${RELAY_PREFIX}image_raw RELAY_TYPE=sensor_msgs/msg/Image
    else
      RELAY_OUT=$RELAY_PREFIX$DEPTH_TR RELAY_TYPE=sensor_msgs/msg/CompressedImage
    fi
    TOPICS+=("$RELAY_OUT")
  fi
fi
# shellcheck disable=SC2206
TOPICS+=(${EXTRA_TOPICS:-})

# throttle_relay declares rate_hz as a double: an integer override ("5") makes it exit at start
RATE_PARAM=$DEPTH_RATE
[[ $RATE_PARAM =~ ^[0-9]+$ ]] && RATE_PARAM=$RATE_PARAM.0
RELAY_CMD=(ros2 run g2cell_recorder throttle_relay --ros-args -p input:="$RELAY_IN"
  -p output:="$RELAY_OUT" -p rate_hz:="$RATE_PARAM" -p msg_type:="$RELAY_TYPE")

DIR=${EPISODES_DIR:-/g2cell_output/episodes}/$NAME
BAG=$DIR/bag
REC=(ros2 bag record --storage mcap -o "$BAG")
[[ $BAGCOMP != none ]] && REC+=(--storage-preset-profile "$BAGCOMP")
REC+=(--topics "${TOPICS[@]}")

if [[ $DRY -eq 1 ]]; then
  echo "TOPICS=${TOPICS[*]}"
  if [[ -n $RELAY_IN ]]; then
    echo "RELAY=$RELAY_IN -> $RELAY_OUT @ $DEPTH_RATE Hz ($RELAY_TYPE)"
    echo "RELAY_CMD=${RELAY_CMD[*]}"
  else
    echo "RELAY=none"
  fi
  echo "CMD=${REC[*]}"
  exit 0
fi

[[ -e $BAG ]] && { echo "$BAG already exists; pick another name" >&2; exit 1; }
mkdir -p "$DIR"

RELAY_PID=""
cleanup() { [[ -n $RELAY_PID ]] && kill "$RELAY_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [[ -n $RELAY_IN ]]; then
  "${RELAY_CMD[@]}" &
  RELAY_PID=$!
fi

echo "recording ${#TOPICS[@]} topics to $BAG  (Ctrl+C to stop)"
# Ctrl+C reaches ros2 bag (same process group), which closes the bag cleanly; the trap only
# keeps this script alive so the conversion still runs afterwards.
trap 'echo; echo "stopping recorder ..."' INT
if [[ -n $DURATION ]]; then
  timeout --foreground -s INT "$DURATION" "${REC[@]}" || true
else
  "${REC[@]}" || true
fi
trap - INT
cleanup; RELAY_PID=""

[[ -f $BAG/metadata.yaml ]] || { echo "no bag written at $BAG" >&2; exit 1; }
ros2 bag info "$BAG" || true
[[ $CONVERT -eq 1 ]] || exit 0

echo "converting to $DIR/$NAME.hdf5 ..."
if ! python3 -c "import h5py" 2>/dev/null; then
  echo "h5py is not installed in this python3; convert later with:" >&2
  echo "  bag_to_hdf5 $BAG -o $DIR/$NAME.hdf5 --scene '$SCENE' --seed $SEED --rate $RATE" >&2
  exit 3
fi
exec ros2 run g2cell_recorder bag_to_hdf5 "$BAG" -o "$DIR/$NAME.hdf5" \
  --scene "$SCENE" --seed "$SEED" --rate "$RATE" "${EXTRA[@]}"
