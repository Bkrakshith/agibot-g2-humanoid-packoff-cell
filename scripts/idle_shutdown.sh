#!/usr/bin/env bash
# Power the GPU server off after N idle minutes, so a forgotten machine stops billing.
#
# "Idle" means ALL of these hold at every check (every 5 min, from a systemd timer):
#   - no GPU above GPU_UTIL_THRESHOLD % utilisation (nvidia-smi)
#   - no established SSH connection (incl. tunnels such as ssh -N -L 8765:...; IGNORE_SSH=1 drops this test)
#   - no Genie Sim sim running in the g2cell container (ros2 launch ... app.launch.py / Isaac Sim kit)
#   - no docker pull/push/build/load/save running, and no inhibit file
# After IDLE_MINUTES (default 30) of continuous idleness: `systemctl poweroff`.
#
#   sudo scripts/idle_shutdown.sh install [--minutes 30] [--gpu-threshold 10] [--interval 5] [--ignore-ssh]
#   sudo scripts/idle_shutdown.sh uninstall       # remove timer, service, config
#   sudo scripts/idle_shutdown.sh disable|enable  # stop/start the timer, keep the files
#   sudo scripts/idle_shutdown.sh pause           # inhibit until the next reboot (long unattended job)
#   sudo scripts/idle_shutdown.sh resume
#   scripts/idle_shutdown.sh status               # timer state, config, current verdict
#   scripts/idle_shutdown.sh check [--act]        # one check; prints the verdict; --act may power off
#
# Files: /usr/local/sbin/g2cell-idle-shutdown (a copy of this script), /etc/default/g2cell-idle-shutdown
# (config), g2cell-idle-shutdown.{service,timer}, state in /run/g2cell-idle-shutdown (cleared at boot).
# Logs: journalctl -u g2cell-idle-shutdown
#
# On Paperspace, an OS poweroff leaves the machine "Off" in the console (compute billing stops; the
# disk is still billed). Verify this once in the console after the first idle shutdown.
set -euo pipefail

NAME=g2cell-idle-shutdown
BIN=/usr/local/sbin/$NAME
CONF=/etc/default/$NAME
UNIT_DIR=/etc/systemd/system
STATE=/run/$NAME
INHIBIT=$STATE/inhibit

IDLE_MINUTES=30
GPU_UTIL_THRESHOLD=10
INTERVAL_MINUTES=5
IGNORE_SSH=0
SSH_PORT=22
CONTAINER=g2cell
# shellcheck disable=SC1090
[[ -r $CONF ]] && . "$CONF"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_root() { [[ $EUID -eq 0 ]] || die "run with sudo"; }
usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

# Prints one reason per line for each activity found; prints nothing when idle.
activity() {
  [[ -e $INHIBIT ]] && echo "inhibited ($INHIBIT; 'resume' clears it)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local u max=0
    while read -r u; do
      u=${u//[!0-9]/}
      [[ -n $u && $u -gt $max ]] && max=$u
    done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    (( max > GPU_UTIL_THRESHOLD )) && echo "gpu ${max}% > ${GPU_UTIL_THRESHOLD}%"
  fi
  if [[ $IGNORE_SSH != 1 ]]; then
    local n
    n=$(ss -Htn state established "( sport = :$SSH_PORT )" 2>/dev/null | wc -l)
    (( n > 0 )) && echo "ssh connections: $n"
  fi
  if command -v docker >/dev/null 2>&1 \
     && [[ -n $(docker ps -q --filter "name=^${CONTAINER}$" 2>/dev/null) ]]; then
    # host-side process list of the container; no exec into it
    if docker top "$CONTAINER" -eo args 2>/dev/null | grep -qE 'app\.launch\.py|genie_sim_engine|isaacsim|kit/kit'; then
      echo "sim running in container $CONTAINER"
    fi
  fi
  if pgrep -f -- '(^|/)docker(-buildx)? (pull|push|build|buildx|load|save|image (pull|push|load|save))( |$)' >/dev/null 2>&1; then
    echo "docker transfer/build in progress"
  fi
  return 0
}

check() {
  local act=${1:-0} reasons now since idle_min
  mkdir -p "$STATE" 2>/dev/null || true
  reasons=$(activity)
  now=$(date +%s)
  if [[ -n $reasons ]]; then
    rm -f "$STATE/idle_since" 2>/dev/null || true
    echo "active: $(paste -sd ';' <<<"$reasons")"
    return 0
  fi
  if [[ -r $STATE/idle_since ]]; then since=$(cat "$STATE/idle_since"); else
    since=$now
    { echo "$now" > "$STATE/idle_since"; } 2>/dev/null || true
  fi
  idle_min=$(( (now - since) / 60 ))
  echo "idle for ${idle_min} min (power off at ${IDLE_MINUTES})"
  if (( idle_min >= IDLE_MINUTES )); then
    if [[ $act == 1 ]]; then
      logger -t "$NAME" "idle ${idle_min} min: powering off"
      wall "g2cell idle ${idle_min} min: powering off now (scripts/idle_shutdown.sh pause to prevent)" 2>/dev/null || true
      systemctl poweroff
    else
      echo "would power off now (run with --act, as the timer does)"
    fi
  fi
}

install_units() {
  need_root
  local minutes=$IDLE_MINUTES thr=$GPU_UTIL_THRESHOLD interval=$INTERVAL_MINUTES ign=$IGNORE_SSH
  while [[ $# -gt 0 ]]; do
    case $1 in
      --minutes) minutes=$2; shift 2 ;;
      --gpu-threshold) thr=$2; shift 2 ;;
      --interval) interval=$2; shift 2 ;;
      --ignore-ssh) ign=1; shift ;;
      *) die "unknown option: $1" ;;
    esac
  done
  [[ $minutes =~ ^[0-9]+$ && $thr =~ ^[0-9]+$ && $interval =~ ^[0-9]+$ ]] || die "numbers only"
  install -m 0755 "$(readlink -f "$0")" "$BIN"
  cat > "$CONF" <<EOF
# g2cell idle shutdown (scripts/idle_shutdown.sh). Edit, then nothing else is needed (read at every check).
IDLE_MINUTES=$minutes
GPU_UTIL_THRESHOLD=$thr
INTERVAL_MINUTES=$interval
IGNORE_SSH=$ign
SSH_PORT=$SSH_PORT
CONTAINER=$CONTAINER
EOF
  cat > "$UNIT_DIR/$NAME.service" <<EOF
[Unit]
Description=g2cell: power off after ${minutes} idle minutes (no GPU load, SSH or sim)
After=docker.service

[Service]
Type=oneshot
ExecStart=$BIN check --act
EOF
  cat > "$UNIT_DIR/$NAME.timer" <<EOF
[Unit]
Description=g2cell idle check every ${interval} min

[Timer]
OnBootSec=${interval}min
OnUnitActiveSec=${interval}min
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$NAME.timer"
  echo "installed: power off after $minutes idle min (GPU <= $thr %, ssh $([[ $ign == 1 ]] && echo ignored || echo counted)), checked every $interval min"
  echo "disable: sudo $0 disable   |   pause until reboot: sudo $0 pause   |   remove: sudo $0 uninstall"
}

uninstall_units() {
  need_root
  systemctl disable --now "$NAME.timer" 2>/dev/null || true
  rm -f "$UNIT_DIR/$NAME.timer" "$UNIT_DIR/$NAME.service" "$BIN" "$CONF"
  rm -rf "$STATE"
  systemctl daemon-reload
  echo "removed"
}

status() {
  echo "timer:   $(systemctl is-enabled "$NAME.timer" 2>/dev/null || echo not-installed) / $(systemctl is-active "$NAME.timer" 2>/dev/null || true)"
  systemctl list-timers "$NAME.timer" --no-pager 2>/dev/null | sed -n '2p' || true
  echo "config:  IDLE_MINUTES=$IDLE_MINUTES GPU_UTIL_THRESHOLD=$GPU_UTIL_THRESHOLD IGNORE_SSH=$IGNORE_SSH"
  echo "uptime:  $(uptime -p)  (since $(uptime -s))"
  printf 'verdict: '; check 0
}

case ${1:-} in
  install) shift; install_units "$@" ;;
  uninstall) uninstall_units ;;
  disable) need_root; systemctl disable --now "$NAME.timer"; echo "timer stopped (enable: sudo $0 enable)" ;;
  enable) need_root; systemctl enable --now "$NAME.timer" ;;
  pause) need_root; mkdir -p "$STATE"; touch "$INHIBIT"; echo "paused until reboot or '$0 resume'" ;;
  resume) need_root; rm -f "$INHIBIT"; echo "resumed" ;;
  status) status ;;
  check) shift; if [[ ${1:-} == --act ]]; then check 1; else check 0; fi ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
