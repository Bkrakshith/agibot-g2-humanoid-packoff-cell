#!/usr/bin/env bash
# First-time host setup for the G2 cell on a fresh GPU machine (Paperspace, Ubuntu 22.04/24.04,
# NVIDIA driver already installed, headless, user with sudo). Safe to rerun: every step checks
# before it changes anything.
#
#   1. Docker + nvidia-container-toolkit; registers the nvidia runtime (docker is restarted ONLY
#      when that runtime is missing, and refuses to if containers are running unless
#      FORCE_DOCKER_RESTART=1)
#   2. git-lfs
#   3. ../genie_sim at the pinned commit, LFS files pulled
#   4. a Python venv (.venv-host/) with huggingface_hub, then the asset subset -> ../geniesim_assets
#   5. disk-space warnings
#
# Env knobs:
#   GENIESIM_DIR   (default ../genie_sim)      ASSETS_DIR (default ../geniesim_assets)
#   GENIESIM_SHA   (default 6ca11c7...)        VENV_DIR   (default .venv-host)
#   SKIP_ASSETS=1  skip step 4                 FORCE_DOCKER_RESTART=1  see step 1
#   NO_GPU=1       skip the driver check (CPU-only builder, or before scripts/server_setup.sh installs it)
#   SKIP_DOCKER=1  skip step 1 (scripts/server_setup.sh does Docker, the daemon config and the runtime)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PARENT=$(cd "$ROOT/.." && pwd)
GENIESIM_URL=${GENIESIM_URL:-https://github.com/AgibotTech/genie_sim.git}
GENIESIM_SHA=${GENIESIM_SHA:-6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46}
GENIESIM_DIR=${GENIESIM_DIR:-$PARENT/genie_sim}
ASSETS_DIR=${ASSETS_DIR:-$PARENT/geniesim_assets}
VENV_DIR=${VENV_DIR:-$ROOT/.venv-host}
G2_MESHES=source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/g2/meshes
RECOMMENDED_GB=150

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  have sudo || die "need root or sudo"
  SUDO="sudo"
fi

APT_UPDATED=0
apt_install() {
  if [[ $APT_UPDATED -eq 0 ]]; then
    $SUDO apt-get update -y
    APT_UPDATED=1
  fi
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

# docker, with sudo if the current shell is not (yet) in the docker group
dk() {
  if docker info >/dev/null 2>&1; then docker "$@"; else $SUDO docker "$@"; fi
}

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# ---------------------------------------------------------------------------------------------
log "GPU driver"
if [[ ${NO_GPU:-0} == 1 ]]; then
  echo "skipped (NO_GPU=1)"
else
  have nvidia-smi || die "nvidia-smi not found: run scripts/server_setup.sh (installs driver 580), or NO_GPU=1"
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "driver $driver"
  [[ ${driver%%.*} -ge 580 ]] || warn "driver $driver < 580; Isaac Sim 5.1/6.0 may refuse to start"
fi

# ---------------------------------------------------------------------------------------------
log "base packages"
need=()
for p in ca-certificates curl gnupg git; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [[ ${#need[@]} -eq 0 ]]; then echo "ok"; else apt_install "${need[@]}"; fi

if [[ ${SKIP_DOCKER:-0} == 1 ]]; then
  log "Docker + nvidia runtime: skipped (SKIP_DOCKER=1)"
else
# ---------------------------------------------------------------------------------------------
log "Docker"
if have docker; then
  docker --version
else
  echo "installing Docker Engine from download.docker.com"
  # shellcheck disable=SC1091
  . /etc/os-release
  $SUDO install -m 0755 -d /etc/apt/keyrings
  $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  APT_UPDATED=0
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  $SUDO systemctl enable --now docker
fi
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
  $SUDO usermod -aG docker "$USER"
  warn "added $USER to the docker group; log out and back in (or 'newgrp docker') before running docker without sudo"
fi

# ---------------------------------------------------------------------------------------------
log "nvidia-container-toolkit"
if have nvidia-ctk; then
  nvidia-ctk --version | head -1
else
  echo "installing from nvidia.github.io/libnvidia-container"
  keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  [[ -f $keyring ]] || curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor -o "$keyring"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  APT_UPDATED=0
  apt_install nvidia-container-toolkit
fi

if dk info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  echo "docker already has the nvidia runtime; not touching the daemon"
else
  echo "registering the nvidia runtime with docker"
  $SUDO nvidia-ctk runtime configure --runtime=docker
  running=$(dk ps -q | wc -l)
  if [[ $running -gt 0 && ${FORCE_DOCKER_RESTART:-0} != 1 ]]; then
    die "$running container(s) running and docker needs a restart to load the nvidia runtime.
       Restarting kills them (and any docker build in progress). Stop them, or rerun with
       FORCE_DOCKER_RESTART=1."
  fi
  $SUDO systemctl restart docker
  dk info --format '{{json .Runtimes}}' | grep -q '"nvidia"' || die "nvidia runtime still missing after restart"
fi

fi

# ---------------------------------------------------------------------------------------------
log "git-lfs"
have git-lfs || apt_install git-lfs
git lfs install --skip-repo >/dev/null
git lfs version

# ---------------------------------------------------------------------------------------------
log "genie_sim @ ${GENIESIM_SHA:0:7} -> $GENIESIM_DIR"
if [[ ! -d $GENIESIM_DIR/.git ]]; then
  [[ -e $GENIESIM_DIR ]] && die "$GENIESIM_DIR exists but is not a git checkout; move it away"
  GIT_LFS_SKIP_SMUDGE=1 git clone "$GENIESIM_URL" "$GENIESIM_DIR"
fi
head=$(git -C "$GENIESIM_DIR" rev-parse HEAD)
if [[ $head != "$GENIESIM_SHA" ]]; then
  if [[ -n $(git -C "$GENIESIM_DIR" status --porcelain --untracked-files=no) ]]; then
    die "$GENIESIM_DIR is at $head with local changes; commit/stash them, then rerun"
  fi
  git -C "$GENIESIM_DIR" cat-file -e "$GENIESIM_SHA^{commit}" 2>/dev/null \
    || git -C "$GENIESIM_DIR" fetch origin
  GIT_LFS_SKIP_SMUDGE=1 git -C "$GENIESIM_DIR" -c advice.detachedHead=false checkout "$GENIESIM_SHA"
fi
git -C "$GENIESIM_DIR" lfs pull
if grep -lq "git-lfs.github.com/spec" "$GENIESIM_DIR/$G2_MESHES"/* 2>/dev/null; then
  die "G2 meshes are still LFS pointers after 'git lfs pull' in $GENIESIM_DIR"
fi
echo "LFS files: $(git -C "$GENIESIM_DIR" lfs ls-files | wc -l) (expected 330)"

# ---------------------------------------------------------------------------------------------
if [[ ${SKIP_ASSETS:-0} == 1 ]]; then
  log "assets: skipped (SKIP_ASSETS=1)"
else
  log "asset subset -> $ASSETS_DIR"
  py=""
  for c in python3.12 python3.11 python3.10 python3; do
    if have "$c" && "$c" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,12)))'; then
      py=$c; break
    fi
  done
  [[ -n $py ]] || die "need Python 3.10-3.12 (Ubuntu 22.04 ships 3.10, 24.04 ships 3.12)"
  if [[ ! -x $VENV_DIR/bin/python ]]; then
    "$py" -m venv "$VENV_DIR" 2>/dev/null || {
      apt_install "$(basename "$py")-venv" || apt_install python3-venv
      "$py" -m venv "$VENV_DIR"
    }
  fi
  "$VENV_DIR/bin/python" -m pip install -q --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -q 'huggingface_hub>=0.23'
  "$VENV_DIR/bin/python" "$HERE/fetch_assets.py" --dest "$ASSETS_DIR"
fi

# ---------------------------------------------------------------------------------------------
log "disk space"
docker_root=$(dk info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
for d in "$docker_root" /var/lib/containerd "$PARENT"; do
  [[ -d $d ]] || d=$(dirname "$d")
  gb=$(free_gb "$d" || true)
  [[ -n $gb ]] || continue
  echo "$d: ${gb} GB free"
  if [[ $gb -lt 80 ]]; then
    warn "$d has ${gb} GB free; docker/build_all.sh needs >= 80 GB (the containerd store and build cache roughly double image size)"
  elif [[ $gb -lt $RECOMMENDED_GB ]]; then
    warn "$d has ${gb} GB free; ${RECOMMENDED_GB} GB+ recommended"
  fi
done

log "done"
cat <<EOF
Next:
  docker run --rm --gpus all ubuntu nvidia-smi     # optional GPU-in-docker smoke test
  $ROOT/docker/build_all.sh                        # ~1 h; or docker load / pull, see docs/PAPERSPACE.md
  $ROOT/docker/cell.sh up
EOF
