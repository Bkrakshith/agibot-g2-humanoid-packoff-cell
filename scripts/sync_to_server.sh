#!/usr/bin/env bash
# Copy the project and the asset folder from this workstation to the server (rsync over SSH,
# compressed, resumable). genie_sim is NOT copied: the server clones it at the pinned commit and
# pulls LFS itself (scripts/server_setup.sh -> bootstrap_host.sh).
#
#   scripts/sync_to_server.sh                 # push g2_packoff_cell + geniesim_assets to paperspace:~/g2/
#   scripts/sync_to_server.sh --dry-run       # list what would be sent, send nothing
#   scripts/sync_to_server.sh --delete        # also delete server files that are gone here (excluded paths are kept)
#   scripts/sync_to_server.sh --pull-output   # fetch the server's output/ (episodes, renders, ...) back here
#   scripts/sync_to_server.sh --push-dataset pick_v1   # output/episodes/pick_v1/ -> server (HDF5, episode.json,
#                                             # dataset card; not the MCAP bags or _crashed/ unless --with-bags)
#   scripts/sync_to_server.sh --host other --remote-dir work/g2
#
# Sent: g2_packoff_cell/ (everything but the excludes below) and geniesim_assets/ (incl. scenes/g2cell
# and every robot under robot/). Excluded: output/episodes, output/v2_work, ros_ws/{build,install,log},
# .venv*, __pycache__, .pytest_cache, *.mcap. About 0.5 GB on the first run, seconds after that.
#   HOST (default paperspace, the ssh alias)   REMOTE_DIR (default g2, relative to the remote home)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PARENT=$(cd "$ROOT/.." && pwd)
HOST=${HOST:-paperspace}
REMOTE_DIR=${REMOTE_DIR:-g2}
ASSETS_DIR=${ASSETS_DIR:-$PARENT/geniesim_assets}
DRY=0
DELETE=0
PULL=0
DATASET=""
WITH_BAGS=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--dry-run) DRY=1; shift ;;
    --delete) DELETE=1; shift ;;
    --pull-output) PULL=1; shift ;;
    --push-dataset) DATASET=${2:?--push-dataset needs a batch name}; shift 2 ;;
    --with-bags) WITH_BAGS=1; shift ;;
    --host) HOST=$2; shift 2 ;;
    --remote-dir) REMOTE_DIR=$2; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# -a archive, -z compress (rsync >= 3.2 negotiates zstd/lz4 when both ends have it),
# --partial resumes big files, -h/progress2 for one overall progress line
# shellcheck disable=SC2054  # the commas belong to --info
OPTS=(-az --partial -h --info=stats1,progress2)
# shellcheck disable=SC2054
[[ $DRY == 1 ]] && OPTS+=(--dry-run --itemize-changes --info=stats2,name1)
[[ $DELETE == 1 ]] && OPTS+=(--delete)

COMMON_EXCLUDES=(--exclude '.venv*/' --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.pyc')
PROJECT_EXCLUDES=(
  --exclude '/output/episodes/' --exclude '/output/v2_work/'
  --exclude '/output/cache*/' --exclude '/output/readiness/pilot/aside/'   # frame caches are rebuilt per machine
  --exclude '/ros_ws/build/' --exclude '/ros_ws/install/' --exclude '/ros_ws/log/'
  --exclude '*.mcap'
)

if [[ $PULL == 1 ]]; then
  echo "==> $HOST:$REMOTE_DIR/g2_packoff_cell/output/ -> $ROOT/output/"
  rsync "${OPTS[@]}" "$HOST:$REMOTE_DIR/g2_packoff_cell/output/" "$ROOT/output/"
  exit 0
fi

if [[ -n $DATASET ]]; then
  [[ $DATASET != */* && $DATASET != .* ]] || { echo "--push-dataset takes a batch name under output/episodes/" >&2; exit 2; }
  SRC=$ROOT/output/episodes/$DATASET
  [[ -d $SRC ]] || { echo "no dataset at $SRC" >&2; exit 1; }
  DEST=$REMOTE_DIR/g2_packoff_cell/output/episodes/$DATASET
  DS_EXCLUDES=(--exclude '_crashed/' --exclude '*.tmp')
  [[ $WITH_BAGS == 1 ]] || DS_EXCLUDES+=(--exclude 'bag/' --exclude '*.mcap')
  # shellcheck disable=SC2029  # DEST is meant to expand here
  [[ $DRY == 1 ]] || ssh "$HOST" "mkdir -p $DEST"
  echo "==> $SRC/ -> $HOST:$DEST/"
  rsync "${OPTS[@]}" "${DS_EXCLUDES[@]}" "$SRC/" "$HOST:$DEST/"
  [[ $DRY == 1 ]] && echo "(dry run: nothing was sent)"
  exit 0
fi

[[ -f $ASSETS_DIR/pyproject.toml ]] || { echo "no geniesim_assets at $ASSETS_DIR" >&2; exit 1; }
# shellcheck disable=SC2029  # REMOTE_DIR is meant to expand here
[[ $DRY == 1 ]] || ssh "$HOST" "mkdir -p $REMOTE_DIR/g2_packoff_cell $REMOTE_DIR/geniesim_assets"

echo "==> $ROOT/ -> $HOST:$REMOTE_DIR/g2_packoff_cell/"
rsync "${OPTS[@]}" "${COMMON_EXCLUDES[@]}" "${PROJECT_EXCLUDES[@]}" \
  "$ROOT/" "$HOST:$REMOTE_DIR/g2_packoff_cell/"

echo "==> $ASSETS_DIR/ -> $HOST:$REMOTE_DIR/geniesim_assets/"
rsync "${OPTS[@]}" "${COMMON_EXCLUDES[@]}" \
  "$ASSETS_DIR/" "$HOST:$REMOTE_DIR/geniesim_assets/"

[[ $DRY == 1 ]] && echo "(dry run: nothing was sent)"
echo "server: ~/$REMOTE_DIR/{g2_packoff_cell,geniesim_assets}; genie_sim is cloned there by scripts/server_setup.sh"

# genie_sim is cloned fresh on the server, but genie_sim_robot_model/urdf/ is GENERATED (gitignored,
# xacro output) and MoveIt's wbc.launch.py loads genie_<model>_<arm>_<gripper>.urdf from it. Without
# it move_group dies on start (found on the A6000, 2026-10-11). Copy it if the server clone exists.
URDF_SUB=source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/urdf
GENIESIM_LOCAL=${GENIESIM_LOCAL:-$(cd "$ROOT/../genie_sim" 2>/dev/null && pwd)}
if [[ -n ${GENIESIM_LOCAL:-} && -d $GENIESIM_LOCAL/$URDF_SUB ]]; then
  # shellcheck disable=SC2029
  if ssh "$HOST" "test -d $REMOTE_DIR/genie_sim/$(dirname "$URDF_SUB")"; then
    echo "==> generated URDFs -> $HOST:$REMOTE_DIR/genie_sim/$URDF_SUB/"
    rsync "${OPTS[@]}" "$GENIESIM_LOCAL/$URDF_SUB/" "$HOST:$REMOTE_DIR/genie_sim/$URDF_SUB/"
  else
    echo "note: $HOST:$REMOTE_DIR/genie_sim not cloned yet; rerun after server_setup.sh to copy generated URDFs" >&2
  fi
fi
