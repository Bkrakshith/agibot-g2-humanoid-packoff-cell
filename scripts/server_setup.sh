#!/usr/bin/env bash
# One-shot, idempotent setup of a FRESH Ubuntu 22.04 Paperspace machine (no NVIDIA driver, no Docker).
# Run it on the server from the synced project; rerun it after the reboot it asks for.
#
#   scripts/server_setup.sh             # GPU server: grow disk, Docker, NVIDIA driver 580 (reboot), toolkit, self-test
#   scripts/server_setup.sh --builder   # CPU-only image builder: grow disk, Docker + buildx, git-lfs, genie_sim clone
#
# Steps (each checks before it changes anything):
#   1. grow the root filesystem into a resized disk (LVM: growpart, pvresize, lvextend, resize2fs)
#   2. base packages: git, git-lfs, rsync, tmux, growpart, ...
#   3. /etc/docker/daemon.json: classic overlay2 image store (containerd-snapshotter off: no 2x disk),
#      log rotation, parallel pulls. Written BEFORE Docker's first start.
#   4. Docker CE + buildx from download.docker.com; you join the docker group
#   5. GPU: nvidia-container-toolkit, nvidia runtime registered and made the default
#   6. GPU: persistent Kit/OV shader + cache dirs on the big disk (docker/cell.sh mounts them)
#   7. genie_sim at the pinned commit + LFS (scripts/bootstrap_host.sh, assets skipped: sync them)
#   8. GPU: NVIDIA driver nvidia-driver-580 (workstation: 580.178.04). After installing it the script
#      EXITS (code 3) and asks for a reboot; rerun it afterwards. REBOOT=1 reboots for you.
#   9. GPU: self-test: docker run --gpus all ... nvidia-smi
#
# Env knobs:
#   DRIVER_BRANCH=580          DRIVER_EXPECTED=580.178.04 (only warns on a mismatch)
#   DRIVER_PPA=1               use ppa:graphics-drivers if Ubuntu's archive has no nvidia-driver-580
#   ROOT_DISK=/dev/xvda ROOT_PART=3   override the auto-detected partition to grow
#   NO_GROW=1                  skip step 1         SKIP_GENIESIM=1   skip step 7
#   CACHE_ROOT=~/docker/isaac-sim     (same default as docker/cell.sh)
#   FORCE_DOCKER_RESTART=1     restart Docker for a config change even with containers running
#   REBOOT=1                   reboot automatically when the driver needs it
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
MODE=gpu
for a in "$@"; do
  case $a in
    --builder) MODE=builder ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

DRIVER_BRANCH=${DRIVER_BRANCH:-580}
DRIVER_EXPECTED=${DRIVER_EXPECTED:-580.178.04}
CACHE_ROOT=${CACHE_ROOT:-$HOME/docker/isaac-sim}
STATE_DIR=/var/lib/g2cell-setup
TARGET_USER=${SUDO_USER:-$USER}

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  have sudo || die "need root or sudo"
  SUDO="sudo"
fi
# shellcheck disable=SC1091
. /etc/os-release
[[ ${ID:-} == ubuntu ]] || die "written for Ubuntu (found ${ID:-unknown})"
[[ ${VERSION_ID:-} == 22.04 || ${VERSION_ID:-} == 24.04 ]] || warn "tested for Ubuntu 22.04; found ${VERSION_ID:-?}"
$SUDO mkdir -p "$STATE_DIR"

APT_UPDATED=0
apt_update() { $SUDO apt-get update -y; APT_UPDATED=1; }
apt_install() {
  [[ $APT_UPDATED -eq 1 ]] || apt_update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}
need_pkgs() {  # install only the missing ones
  local p miss=()
  for p in "$@"; do dpkg -s "$p" >/dev/null 2>&1 || miss+=("$p"); done
  [[ ${#miss[@]} -eq 0 ]] || apt_install "${miss[@]}"
}

# ---------------------------------------------------------------------------------------------
log "1. root filesystem"
need_pkgs cloud-guest-utils lvm2 gdisk
grow_root() {
  local src fstype lv="" vg="" pv disk part dev_bytes fs_bytes
  src=$(findmnt -no SOURCE /)
  fstype=$(findmnt -no FSTYPE /)
  if $SUDO lvs "$src" >/dev/null 2>&1; then
    lv=$src
    vg=$($SUDO lvs --noheadings -o vg_name "$lv" | tr -d ' ')
    pv=$($SUDO pvs --noheadings -o pv_name -S "vg_name=$vg" | tr -d ' ' | head -1)
  else
    pv=$src
  fi
  disk=${ROOT_DISK:-/dev/$(lsblk -no PKNAME "$pv" | head -1)}
  part=${ROOT_PART:-$(cat "/sys/class/block/$(basename "$(readlink -f "$pv")")/partition")}
  echo "root: $src ($fstype)${lv:+ on VG $vg, PV $pv}; partition $part of $disk"

  # growpart exits 0 when it can grow, 1 with NOCHANGE when the partition already fills the disk
  if $SUDO growpart --dry-run "$disk" "$part" >/dev/null 2>&1; then
    echo "growing partition $part of $disk"
    $SUDO growpart "$disk" "$part"
  else
    echo "partition already fills $disk"
  fi
  if [[ -n $lv ]]; then
    $SUDO pvresize "$pv" >/dev/null
    local free_ext
    free_ext=$($SUDO vgs --noheadings -o vg_free_count "$vg" | tr -d ' ')
    if [[ ${free_ext:-0} -gt 0 ]]; then
      echo "extending $lv by $free_ext free extents"
      $SUDO lvextend -l +100%FREE "$lv"
    else
      echo "no free space in VG $vg"
    fi
  fi
  dev_bytes=$($SUDO blockdev --getsize64 "$(readlink -f "$src")")
  fs_bytes=$(df -B1 --output=size / | tail -1 | tr -dc '0-9')
  if (( dev_bytes - fs_bytes > 2 * 1024 * 1024 * 1024 )); then
    echo "filesystem is smaller than its device; resizing"
    case $fstype in
      ext4|ext3|ext2) $SUDO resize2fs "$src" ;;
      xfs) $SUDO xfs_growfs / ;;
      *) warn "don't know how to grow $fstype; grow it by hand" ;;
    esac
  else
    echo "filesystem already fills its device"
  fi
  df -h / | tail -1
}
if [[ ${NO_GROW:-0} == 1 ]]; then echo "skipped (NO_GROW=1)"; else grow_root; fi

# ---------------------------------------------------------------------------------------------
log "2. base packages"
need_pkgs ca-certificates curl gnupg git git-lfs rsync tmux python3 jq htop
git lfs install --skip-repo >/dev/null
echo "ok"

# ---------------------------------------------------------------------------------------------
log "3. docker daemon config"
DOCKER_RESTART=0
desired=$(cat <<'JSON'
{
  "features": {"containerd-snapshotter": false},
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "3"},
  "max-concurrent-downloads": 6,
  "max-concurrent-uploads": 4
}
JSON
)
$SUDO mkdir -p /etc/docker
current='{}'
[[ -s /etc/docker/daemon.json ]] && current=$($SUDO cat /etc/docker/daemon.json)
# merge: our keys win, everything else (e.g. the nvidia runtime nvidia-ctk adds) is kept
merged=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); d=json.loads(sys.argv[2])
for k,v in d.items():
    c[k] = {**c.get(k,{}), **v} if isinstance(v,dict) and isinstance(c.get(k),dict) else v
print(json.dumps(c, indent=2, sort_keys=True))' "$current" "$desired")
if [[ $(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$current") != \
      $(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$merged") ]]; then
  if have docker && $SUDO docker image ls -q 2>/dev/null | grep -q . \
     && ! grep -q '"containerd-snapshotter": false' <<<"$current"; then
    warn "switching the image store hides images pulled so far (pull them again afterwards)"
  fi
  echo "$merged" | $SUDO tee /etc/docker/daemon.json >/dev/null
  echo "wrote /etc/docker/daemon.json"
  DOCKER_RESTART=1
else
  echo "daemon.json already up to date"
fi

# ---------------------------------------------------------------------------------------------
log "4. Docker CE"
if have docker && dpkg -s docker-ce >/dev/null 2>&1; then
  docker --version
else
  $SUDO install -m 0755 -d /etc/apt/keyrings
  [[ -f /etc/apt/keyrings/docker.asc ]] || \
    $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt_update
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  $SUDO systemctl enable --now docker
  DOCKER_RESTART=0   # it just started with the config above
fi
need_pkgs docker-buildx-plugin
if ! id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx docker; then
  $SUDO usermod -aG docker "$TARGET_USER"
  warn "added $TARGET_USER to the docker group: log out and in (or 'newgrp docker') to use docker without sudo"
fi

# ---------------------------------------------------------------------------------------------
if [[ $MODE == gpu ]]; then
  log "5. nvidia-container-toolkit"
  if ! have nvidia-ctk; then
    keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    [[ -f $keyring ]] || curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | $SUDO gpg --dearmor -o "$keyring"
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
      | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    apt_update
    apt_install nvidia-container-toolkit
  fi
  nvidia-ctk --version | head -1
  if $SUDO grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json 2>/dev/null \
     && $SUDO grep -q '"nvidia"' /etc/docker/daemon.json; then
    echo "nvidia runtime registered and default"
  else
    $SUDO nvidia-ctk runtime configure --runtime=docker --set-as-default
    DOCKER_RESTART=1
  fi
fi

if [[ $DOCKER_RESTART == 1 ]]; then
  running=$($SUDO docker ps -q | wc -l)
  if [[ $running -gt 0 && ${FORCE_DOCKER_RESTART:-0} != 1 ]]; then
    die "docker needs a restart for the new config but $running container(s) are running.
       Stop them, or rerun with FORCE_DOCKER_RESTART=1 (kills them and any build/pull)."
  fi
  echo "restarting docker"
  $SUDO systemctl restart docker
fi
storage=$($SUDO docker info --format '{{.Driver}} {{json .DriverStatus}}')
if grep -q 'io.containerd.snapshotter' <<<"$storage"; then
  die "docker still uses the containerd image store ($storage); check /etc/docker/daemon.json"
fi
echo "docker storage: ${storage%% *}, root $($SUDO docker info --format '{{.DockerRootDir}}')"

# ---------------------------------------------------------------------------------------------
if [[ $MODE == gpu ]]; then
  log "6. persistent caches -> $CACHE_ROOT"
  # Same layout as docker/cell.sh. The Kit shader cache (cache/main/ov, computecache) survives
  # container restarts and machine stop/start, so only the first launch compiles shaders.
  mkdir -p "$CACHE_ROOT"/cache/main/{ov,warp,kit,pip,numba} "$CACHE_ROOT"/cache/computecache \
           "$CACHE_ROOT"/config "$CACHE_ROOT"/data/{documents,Kit} "$CACHE_ROOT"/logs "$CACHE_ROOT"/pkg
  df -h "$CACHE_ROOT" | tail -1
fi

# ---------------------------------------------------------------------------------------------
log "7. genie_sim checkout"
if [[ ${SKIP_GENIESIM:-0} == 1 ]]; then
  echo "skipped (SKIP_GENIESIM=1)"
else
  # bootstrap_host.sh only clones/checks out/LFS-pulls here: Docker, the runtime and the driver
  # are this script's job. Assets come from the workstation (scripts/sync_to_server.sh);
  # SKIP_ASSETS=0 downloads the HuggingFace subset instead.
  NO_GPU=1 SKIP_DOCKER=1 SKIP_ASSETS=${SKIP_ASSETS:-1} "$HERE/bootstrap_host.sh"
fi

# ---------------------------------------------------------------------------------------------
if [[ $MODE == gpu ]]; then
  log "8. NVIDIA driver ($DRIVER_BRANCH)"
  boot_epoch=$(date -d "$(uptime -s)" +%s)
  marker=$STATE_DIR/driver-installed-at
  if nvidia-smi >/dev/null 2>&1; then
    ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    [[ $ver == "$DRIVER_EXPECTED" ]] || warn "driver $ver != workstation $DRIVER_EXPECTED (fine if ${ver%%.*} == $DRIVER_BRANCH)"
    [[ ${ver%%.*} -ge 580 ]] || die "driver $ver < 580: Isaac Sim 5.1/6.0 need >= 580"
  else
    if dpkg -s "nvidia-driver-$DRIVER_BRANCH" >/dev/null 2>&1 && [[ -f $marker ]] \
       && (( $(cat "$marker") > boot_epoch )); then
      echo "driver installed but not loaded yet: reboot needed"
    elif dpkg -s "nvidia-driver-$DRIVER_BRANCH" >/dev/null 2>&1; then
      die "nvidia-driver-$DRIVER_BRANCH is installed but nvidia-smi fails. If it was installed outside this
       script, reboot first. Otherwise check: dmesg | grep -i -e nvidia -e nouveau;  dkms status;
       mokutil --sb-state (Secure Boot blocks unsigned DKMS modules)"
    else
      need_pkgs ubuntu-drivers-common "linux-headers-$(uname -r)"
      if [[ ${DRIVER_PPA:-0} == 1 ]]; then
        need_pkgs software-properties-common
        $SUDO add-apt-repository -y ppa:graphics-drivers/ppa
        apt_update
      fi
      # ubuntu-drivers picks the prebuilt signed kernel modules for this kernel when they exist
      # (linux-modules-nvidia-580-*), else DKMS. Full driver, not -headless/-server: Isaac Sim's RTX
      # renderer needs the Vulkan/GL user-space libraries.
      if ! $SUDO ubuntu-drivers install "nvidia:$DRIVER_BRANCH"; then
        apt_install "nvidia-driver-$DRIVER_BRANCH" || die "nvidia-driver-$DRIVER_BRANCH not installable here; rerun with DRIVER_PPA=1"
      fi
      date +%s | $SUDO tee "$marker" >/dev/null
      echo "installed: $(dpkg-query -W -f='${Package} ${Version}\n' "nvidia-driver-$DRIVER_BRANCH")"
    fi
    echo
    echo "REBOOT REQUIRED to load the NVIDIA driver. Then rerun:  $0 $*"
    if [[ ${REBOOT:-0} == 1 ]]; then $SUDO systemctl reboot; fi
    echo "  sudo reboot      # Paperspace: the machine comes back in ~1-2 min"
    exit 3
  fi

  # -------------------------------------------------------------------------------------------
  log "9. self-test: GPU inside a container"
  if $SUDO docker run --rm --gpus all ubuntu:22.04 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
    echo "GPU visible in containers"
  else
    die "docker run --gpus all failed; check 'nvidia-ctk runtime configure' and 'systemctl status docker'"
  fi
  $SUDO docker image rm ubuntu:22.04 >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------------------------
log "done ($MODE)"
df -h / | tail -1
if [[ $MODE == gpu ]]; then
  cat <<EOF
Next (see docs/OPTIMISATION.md "Build & deploy"):
  docker login ghcr.io                             # read-only token
  REGISTRY=ghcr.io NAMESPACE=<user> $ROOT/scripts/pull_on_server.sh
  sudo $ROOT/scripts/idle_shutdown.sh install      # power off after 30 idle minutes
  $ROOT/scripts/session.sh collect                 # or: up | showcase | status | down
EOF
else
  cat <<EOF
Next on this builder:
  docker login ghcr.io                             # token with write:packages
  NAMESPACE=<user> $ROOT/scripts/build_and_push.sh
EOF
fi
