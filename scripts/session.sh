#!/usr/bin/env bash
# One entry point per task on the GPU server; each starts what that task needs and nothing more.
#
#   scripts/session.sh up                      container + built ROS workspaces; no sim (offline tests, editing)
#   scripts/session.sh collect [scene.yaml]    up + sim stack with PROFILE=collect (teleop data collection)
#   scripts/session.sh showcase [--mobile] [scene.yaml]
#                                              up + sim stack with PROFILE=showcase (renders/videos);
#                                              --mobile: mobile scene + MoveIt (drive-in / packing demo)
#   scripts/session.sh down [--poweroff]       stop the stack and the container; --poweroff also stops the machine
#   scripts/session.sh status                  container, tmux windows, GPU, disk, idle-shutdown, uptime
#
# PROFILE is passed straight to ros_ws/scripts/stage1_up.sh (which owns what each profile starts and
# renders); override it with PROFILE=... . Default scene: scene_g2cell.yaml (SCENE=... overrides).
# One stack at a time: collect/showcase refuse while a tmux session "g2cell" runs (down first).
#
# From the workstation: SESSION_HOST=paperspace scripts/session.sh status   (runs it over ssh in
# ~/$REMOTE_DIR, default g2/g2_packoff_cell).
# The image comes from scripts/pull_on_server.sh (needs REGISTRY/NAMESPACE) if g2cell/cell:dev is missing.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
NAME=${NAME:-g2cell}
IMAGE=${IMAGE:-g2cell/cell:dev}
SESSION=g2cell
DEFAULT_SCENE=/g2cell_ws/config/scenes/scene_g2cell.yaml
MOBILE_SCENE=/g2cell_ws/config/scenes/scene_g2cell_mobile.yaml

if [[ -n ${SESSION_HOST:-} && -z ${G2_SESSION_REMOTE:-} ]]; then
  REMOTE_DIR=${REMOTE_DIR:-g2/g2_packoff_cell}
  q=$(printf ' %q' "$@")
  env_fwd=""
  for v in PROFILE SCENE REGISTRY NAMESPACE REPO VARIANT; do
    [[ -n ${!v:-} ]] && env_fwd+=" $v=$(printf '%q' "${!v}")"
  done
  exec ssh -t "$SESSION_HOST" "cd $REMOTE_DIR && G2_SESSION_REMOTE=1$env_fwd scripts/session.sh$q"
fi

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }
in_ctr() { docker exec -u isaac-sim "$@"; }
running() { [[ -n $(docker ps -q --filter "name=^${NAME}$") ]]; }
stack_running() { running && in_ctr "$NAME" tmux has-session -t "$SESSION" 2>/dev/null; }

ensure_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 && return 0
  log "$IMAGE not here; pulling (scripts/pull_on_server.sh)"
  [[ -n ${NAMESPACE:-} ]] || die "no $IMAGE and NAMESPACE unset: export REGISTRY=ghcr.io NAMESPACE=<user>, or run scripts/pull_on_server.sh"
  "$HERE/pull_on_server.sh"
}

cmd_up() {
  ensure_image
  if running; then
    log "container $NAME already running; reusing it"
  else
    "$ROOT/docker/cell.sh" up
  fi
  # build only what is missing (a baked image seeded both already)
  in_ctr "$NAME" bash -lc '
    set -e
    if [ ! -f /workspace/devel/setup.bash ]; then
      echo "==> building genie_sim ROS workspace (geniesim ros build dev)"; cd /workspace && geniesim ros build dev
    fi
    if [ ! -f /g2cell_ws/install/setup.bash ]; then
      echo "==> building /g2cell_ws (colcon build)"; source /workspace/devel/setup.bash && cd /g2cell_ws && colcon build
    fi'
  log "up: $NAME ready (no sim). Shell: docker/cell.sh into"
}

start_stack() {  # <profile> <script> <scene>
  local profile=$1 script=$2 scene=$3; shift 3
  stack_running && die "a stack is already running (tmux session $SESSION); run: scripts/session.sh down"
  log "starting $script, PROFILE=$profile, scene $scene"
  in_ctr -e "PROFILE=$profile" "$@" "$NAME" bash -lc "bash /g2cell_ws/scripts/$script $(printf '%q' "$scene")"
  cat <<EOF
==> running. Attach:   docker exec -it -u isaac-sim $NAME tmux attach -t $SESSION
    Foxglove (laptop): ssh -N -L 8765:localhost:8765 paperspace   ->  ws://localhost:8765
    Stop:              scripts/session.sh down
EOF
}

cmd_collect() {
  local scene=${1:-${SCENE:-$DEFAULT_SCENE}}
  cmd_up
  start_stack "${PROFILE:-collect}" stage1_up.sh "$scene"
  echo "    teleop window: ros2 run g2cell_teleop keyboard_teleop --ros-args -p use_sim_time:=true"
}

cmd_showcase() {
  local mobile=0 scene=""
  while [[ $# -gt 0 ]]; do
    case $1 in --mobile) mobile=1; shift ;; *) scene=$1; shift ;; esac
  done
  cmd_up
  if [[ $mobile == 1 ]]; then
    start_stack "${PROFILE:-showcase}" mobile_up.sh "${scene:-${SCENE:-$MOBILE_SCENE}}" -e WITH_MOVEIT=1
  else
    start_stack "${PROFILE:-showcase}" stage1_up.sh "${scene:-${SCENE:-$DEFAULT_SCENE}}"
  fi
}

cmd_down() {
  if running; then
    if in_ctr "$NAME" tmux kill-session -t "$SESSION" 2>/dev/null; then log "stack stopped"; fi
    "$ROOT/docker/cell.sh" down >/dev/null && log "container $NAME removed"
  else
    log "container $NAME not running"
  fi
  if [[ ${1:-} == --poweroff ]]; then
    log "outputs stay on this disk; fetch them from the workstation with: scripts/sync_to_server.sh --pull-output"
    log "powering off in 10 s (Ctrl+C to cancel)"
    sleep 10
    sudo systemctl poweroff
  fi
}

cmd_status() {
  echo "machine:   up $(uptime -p | sed 's/^up //') since $(uptime -s)   (compute is billed while up)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed 's/^/gpu:       /'
  fi
  df -h --output=avail,size / | tail -1 | awk '{print "disk:      " $1 " free of " $2}'
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image:     $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c8-19))"
  else
    echo "image:     $IMAGE missing (scripts/pull_on_server.sh)"
  fi
  if running; then
    echo "container: $NAME running"
    if stack_running; then
      echo "stack:     tmux windows: $(in_ctr "$NAME" tmux list-windows -t "$SESSION" -F '#W' | paste -sd ' ')"
    else
      echo "stack:     none"
    fi
  else
    echo "container: not running"
  fi
  if [[ -x /usr/local/sbin/g2cell-idle-shutdown ]]; then
    echo "idle-off:  $(/usr/local/sbin/g2cell-idle-shutdown check 2>/dev/null | tail -1)"
  else
    echo "idle-off:  not installed (sudo scripts/idle_shutdown.sh install)"
  fi
}

case ${1:-} in
  up) shift; cmd_up "$@" ;;
  collect) shift; cmd_collect "$@" ;;
  showcase) shift; cmd_showcase "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) shift; cmd_status ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
