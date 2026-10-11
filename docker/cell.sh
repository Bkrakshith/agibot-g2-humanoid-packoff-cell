#!/usr/bin/env bash
# Run the G2 cell container headless. Same script on this workstation and on Paperspace.
#
#   cell.sh up      start the container (detached), then seed prebuilt workspaces (baked image)
#   cell.sh into    shell inside it (as the isaac-sim user)
#   cell.sh down    stop and remove it
#   cell.sh seed    copy the baked image's prebuilt ROS workspaces into the bind mounts (up does this)
#
# Baked images (Dockerfile.cell --target baked, scripts/pull_on_server.sh) carry both ROS workspaces
# prebuilt under /opt/g2cell/prebuilt. `seed` copies them to /workspace/devel{,_build} and
# /g2cell_ws/install only where those do not exist yet, and only if they were built from the same
# genie_sim commit / ros_ws/src content (STAMP); otherwise it says what to build. SEED_PREBUILT=0 skips.
#
# Container settings mirror genie_sim/docker/start.sh --headless, plus our ROS workspace.
#   /workspace        genie_sim checkout (GENIESIM_DIR)
#   /geniesim_assets  asset subset (ASSETS_DIR)
#   /g2cell_ws        this project's ros_ws
# Caches under CACHE_ROOT (keep it on persistent disk; measured in docs/OPTIMISATION.md "Sim runtime"):
#   cache/main/nvidia   driver Vulkan/GL shader cache (~/.cache/nvidia): OVRTX first frame 167 s cold -> ~20 s
#   kitcache            Kit portable cache (isaacsim/kit/cache: DerivedDataCache, shader caches, ~0.4-0.9 GB):
#                       Isaac Sim/Kit startup 76 s cold -> 5 s; it lives in the container layer otherwise, so
#                       every `cell.sh up` (docker rm + run) paid the 76 s again
#   optix               OptiX denoiser cache (/var/tmp/OptixCache_isaac-sim)
#   cache/computecache  CUDA JIT cache
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
GENIESIM_DIR=${GENIESIM_DIR:-$(cd "$ROOT/../genie_sim" && pwd)}
ASSETS_DIR=${ASSETS_DIR:-$(cd "$ROOT/../geniesim_assets" && pwd)}
IMAGE=${IMAGE:-g2cell/cell:dev}
NAME=${NAME:-g2cell}
CACHE_ROOT=${CACHE_ROOT:-$HOME/docker/isaac-sim}

up() {
  [[ -f "$ASSETS_DIR/pyproject.toml" ]] || { echo "no geniesim_assets at $ASSETS_DIR" >&2; exit 1; }
  docker info 2>/dev/null | grep -q nvidia || docker info 2>/dev/null | grep -qi "cdi" \
    || echo "warning: docker shows no nvidia runtime; --gpus will fail" >&2
  mkdir -p "$CACHE_ROOT"/cache/main/{ov,warp} "$CACHE_ROOT"/cache/computecache "$CACHE_ROOT"/config \
           "$CACHE_ROOT"/kitcache "$CACHE_ROOT"/optix \
           "$CACHE_ROOT"/data/{documents,Kit} "$CACHE_ROOT"/logs "$CACHE_ROOT"/pkg "$ROOT/output"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -itd --name "$NAME" --init --gpus all --network=host --privileged \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e ROS_DISTRO=jazzy -e GENIESIM_PY_CMD=python3 -e GENIESIM_BREAK_SYSTEM_PKGS=1 \
    -e __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1 \
    -v "$CACHE_ROOT/cache/main:/home/isaac-sim/.cache:rw" \
    -v "$CACHE_ROOT/cache/computecache:/home/isaac-sim/.nv/ComputeCache:rw" \
    -v "$CACHE_ROOT/logs:/home/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "$CACHE_ROOT/config:/home/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "$CACHE_ROOT/data:/home/isaac-sim/.local/share/ov/data:rw" \
    -v "$CACHE_ROOT/pkg:/home/isaac-sim/.local/share/ov/pkg:rw" \
    -v "$CACHE_ROOT/kitcache:/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache:rw" \
    -v "$CACHE_ROOT/optix:/var/tmp/OptixCache_isaac-sim:rw" \
    -v "$GENIESIM_DIR:/workspace:rw" \
    -v "$ASSETS_DIR:/geniesim_assets:rw" \
    -v "$GENIESIM_DIR/docker/entrypoint.sh:/usr/local/bin/geniesim-entrypoint:ro" \
    -v "$ROOT/ros_ws:/g2cell_ws:rw" \
    -v "$ROOT/output:/g2cell_output:rw" \
    -w /workspace \
    "$IMAGE" tail -f /dev/null >/dev/null
  echo "waiting for entrypoint ..."
  for _ in $(seq 1 120); do
    docker logs "$NAME" 2>&1 | grep -q GENIESIM_READY && { echo "ready: $NAME"; return 0; }
    sleep 2
  done
  echo "entrypoint did not report GENIESIM_READY; see: docker logs $NAME" >&2
  exit 1
}

seed() {
  [[ ${SEED_PREBUILT:-1} == 1 ]] || { echo "seed: skipped (SEED_PREBUILT=0)"; return 0; }
  local stamp want_sha want_ws
  if ! stamp=$(docker exec "$NAME" cat /opt/g2cell/prebuilt/STAMP 2>/dev/null); then
    echo "seed: image has no prebuilt workspaces (not a baked image); build them in the container"
    return 0
  fi
  # shellcheck source-path=SCRIPTDIR source=../scripts/lib/g2_images.sh
  export G2_ROOT=$ROOT; source "$ROOT/scripts/lib/g2_images.sh"
  want_sha=$(git -C "$GENIESIM_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
  want_ws=$(g2_ws_hash)
  if grep -qx "GENIESIM_SHA=$want_sha" <<<"$stamp"; then
    docker exec -u isaac-sim "$NAME" bash -c '
      if [ -f /workspace/devel/setup.bash ]; then echo "seed: /workspace/devel exists; kept"
      else cp -a /opt/g2cell/prebuilt/workspace/devel /opt/g2cell/prebuilt/workspace/devel_build /workspace/ \
           && echo "seed: genie_sim workspace -> /workspace/devel"; fi'
  else
    echo "seed: prebuilt genie_sim is for another commit ($stamp); run 'geniesim ros build dev' in /workspace"
  fi
  if grep -qx "G2CELL_WS_HASH=$want_ws" <<<"$stamp"; then
    docker exec -u isaac-sim "$NAME" bash -c '
      if [ -f /g2cell_ws/install/setup.bash ]; then echo "seed: /g2cell_ws/install exists; kept"
      else cp -a /opt/g2cell/prebuilt/g2cell_ws/install /g2cell_ws/ && echo "seed: g2cell workspace -> /g2cell_ws/install"; fi'
  else
    echo "seed: ros_ws/src changed since the image was baked; run 'colcon build' in /g2cell_ws (~2 min)"
  fi
}

into() { docker exec -it -u isaac-sim -w /workspace "$NAME" bash -l; }
down() { docker rm -f "$NAME"; }

case "${1:-}" in
  up) up; seed ;;
  seed) seed ;;
  into) into ;;
  down) down ;;
  *) echo "usage: $0 {up|into|down|seed}" >&2; exit 2 ;;
esac
