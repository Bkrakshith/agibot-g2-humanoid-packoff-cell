#!/usr/bin/env bash
# Build both images, then free the build cache.
#   1. base: g2cell/geniesim:3.2.0-isaac6.0.1.0  (docker/build_geniesim.sh; skipped if it exists)
#   2. cell: g2cell/cell:dev                      (docker/Dockerfile.cell)
#   3. docker builder prune -af                   (frees ~50 GB with the containerd image store)
#
#   MIN_FREE_GB=80   abort below this much free space where docker stores data
#   REBUILD_BASE=1   rebuild the base even if the tag exists
#   NO_PRUNE=1       keep the build cache
#   BAKED=1          build Dockerfile.cell's "baked" target (ROS workspaces prebuilt, seeded by cell.sh up)
#
# This is the FALLBACK (build where you run, e.g. on the GPU box, ~1.5 h of billed GPU time).
# Preferred: scripts/build_and_push.sh on a CPU machine + scripts/pull_on_server.sh on the GPU box
# (docs/OPTIMISATION.md "Build & deploy").
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAACSIM6_VERSION=${ISAACSIM6_VERSION:-6.0.1.0}
BASE_IMAGE=${BASE_IMAGE:-g2cell/geniesim:3.2.0-isaac${ISAACSIM6_VERSION}}
CELL_IMAGE=${CELL_IMAGE:-g2cell/cell:dev}
MIN_FREE_GB=${MIN_FREE_GB:-80}

docker info >/dev/null 2>&1 || { echo "cannot talk to docker (daemon down, or not in docker group?)" >&2; exit 1; }
docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' \
  || echo "warning: docker has no nvidia runtime; images build but cell.sh up will fail (run scripts/bootstrap_host.sh)" >&2

docker_root=$(docker info --format '{{.DockerRootDir}}')
for d in "$docker_root" /var/lib/containerd; do
  [[ -d $d ]] || continue
  gb=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
  echo "$d: ${gb} GB free"
  if [[ $gb -lt $MIN_FREE_GB ]]; then
    echo "need >= ${MIN_FREE_GB} GB free on $d (Isaac Sim base ~23 GB, final images much more; the build" >&2
    echo "cache roughly doubles that). Free space or set MIN_FREE_GB to override." >&2
    exit 1
  fi
done

if [[ ${REBUILD_BASE:-0} != 1 ]] && docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "base $BASE_IMAGE exists; skipping (REBUILD_BASE=1 to rebuild)"
else
  IMAGE=$BASE_IMAGE ISAACSIM6_VERSION=$ISAACSIM6_VERSION "$HERE/build_geniesim.sh"
fi

cell_args=(--target cell)
if [[ ${BAKED:-0} == 1 ]]; then
  ROOT=$(cd "$HERE/.." && pwd)
  GENIESIM_DIR=${GENIESIM_DIR:-$(cd "$ROOT/../genie_sim" && pwd)}
  # shellcheck source-path=SCRIPTDIR source=../scripts/lib/g2_images.sh
  export G2_ROOT=$ROOT; source "$ROOT/scripts/lib/g2_images.sh"
  cell_args=(--target baked
    --build-arg GENIESIM_SHA="$(git -C "$GENIESIM_DIR" rev-parse HEAD)"
    --build-arg G2CELL_WS_HASH="$(g2_ws_hash)"
    --build-context geniesim_ros="$GENIESIM_DIR/source/geniesim_ros/src/ros_ws"
    --build-context g2cell_src="$ROOT/ros_ws/src")
fi
DOCKER_BUILDKIT=1 docker build \
  -f "$HERE/Dockerfile.cell" \
  "${cell_args[@]}" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$CELL_IMAGE" \
  "$HERE"
echo "built $CELL_IMAGE"

if [[ ${NO_PRUNE:-0} != 1 ]]; then
  docker builder prune -af
fi
docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep '^g2cell/' || true
df -h "$docker_root"
