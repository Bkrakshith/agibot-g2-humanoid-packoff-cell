#!/usr/bin/env bash
# On the GPU server: pull the prebuilt cell image from the private registry and tag it with the
# local name docker/cell.sh runs (g2cell/cell:dev). Replaces building on billed GPU time.
#
#   REGISTRY=ghcr.io NAMESPACE=<user> scripts/pull_on_server.sh               # baked (default)
#   scripts/pull_on_server.sh --namespace <user> --variant cell               # no prebuilt ROS workspaces
#   scripts/pull_on_server.sh --namespace <user> --tag cell-latest            # explicit tag
#   scripts/pull_on_server.sh --namespace <user> --with-base                  # also the base (to rebuild cell here)
#
# Options (or env): --registry [REGISTRY], --namespace [NAMESPACE], --repo [REPO],
#   --variant baked|cell [VARIANT], --tag T (skip the hash lookup), --with-base, --keep-old.
#
# Which tag: the content hash of docker/Dockerfile.* (+ ros_ws/src for baked), computed from the
# files here exactly as scripts/build_and_push.sh did on the builder. If the exact baked tag is
# missing (ros_ws/src changed since the push), it falls back to baked-<cell hash>: the genie_sim
# workspace is still prebuilt and cell.sh skips the stale g2cell workspace (colcon build it, ~2 min).
#
# Log in once first, with a READ-ONLY token where possible (stored by docker, not by this script):
#   echo "$GHCR_READ_PAT" | docker login ghcr.io -u <github-user> --password-stdin    # read:packages
#   docker login -u <hub-user>                                                         # Docker Hub
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
G2_ROOT=$ROOT
# shellcheck source-path=SCRIPTDIR source=lib/g2_images.sh
source "$HERE/lib/g2_images.sh"

VARIANT=${VARIANT:-baked}
TAG=""
WITH_BASE=0
KEEP_OLD=0
MIN_FREE_GB=${MIN_FREE_GB:-70}

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry) REGISTRY=$2; shift 2 ;;
    --namespace) NAMESPACE=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --variant) VARIANT=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --with-base) WITH_BASE=1; shift ;;
    --keep-old) KEEP_OLD=1; shift ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
REF=$(g2_repo) || exit 2

docker info >/dev/null 2>&1 || die "cannot talk to docker (run scripts/server_setup.sh; 'newgrp docker' after it)"
if docker info --format '{{json .DriverStatus}}' | grep -q 'io.containerd.snapshotter'; then
  echo "warning: docker uses the containerd image store: the pull needs ~2x the disk." >&2
  echo "         scripts/server_setup.sh sets features.containerd-snapshotter=false (do it before pulling)." >&2
fi
docker_root=$(docker info --format '{{.DockerRootDir}}')
free=$(df -BG --output=avail "$docker_root" | tail -1 | tr -dc '0-9')
echo "free on $docker_root: ${free} GB"
[[ $free -ge $MIN_FREE_GB ]] || die "need >= ${MIN_FREE_GB} GB free (the image unpacks to ~56 GB); MIN_FREE_GB overrides"

candidates=()
if [[ -n $TAG ]]; then
  candidates=("$TAG")
else
  case $VARIANT in
    baked) candidates=("$(g2_baked_tag)" "$(g2_baked_cell_tag)") ;;
    cell)  candidates=("$(g2_cell_tag)") ;;
    *) die "--variant must be baked or cell" ;;
  esac
fi

pulled=""
t0=$SECONDS
for t in "${candidates[@]}"; do
  echo "==> docker pull $REF:$t"
  if docker pull "$REF:$t"; then pulled=$t; break; fi
  echo "    not available" >&2
done
if [[ -z $pulled ]]; then
  die "none of [${candidates[*]}] could be pulled from $REF.
       Not logged in ('docker login $REGISTRY'), or the images for these Dockerfiles were never pushed:
       run scripts/build_and_push.sh on the builder, or pass --tag cell-latest / baked-latest to take the newest anyway."
fi
[[ $pulled == "${candidates[0]}" ]] || echo "note: using $pulled (fallback); colcon build /g2cell_ws in the container before use" >&2
docker tag "$REF:$pulled" "$LOCAL_CELL_IMAGE"
echo "tagged $REF:$pulled -> $LOCAL_CELL_IMAGE ($(( (SECONDS - t0) / 60 )) min)"

if [[ $WITH_BASE == 1 ]]; then
  docker pull "$REF:$(g2_base_tag)"
  docker tag "$REF:$(g2_base_tag)" "$LOCAL_BASE_IMAGE"
  echo "tagged base -> $LOCAL_BASE_IMAGE"
fi

if [[ $KEEP_OLD != 1 ]]; then
  # drop older tags of this repo; layers shared with the new image stay
  keep="$REF:$pulled"
  [[ $WITH_BASE == 1 ]] && keep="$keep $REF:$(g2_base_tag)"
  docker image ls "$REF" --format '{{.Repository}}:{{.Tag}}' | while read -r img; do
    [[ " $keep " == *" $img "* ]] || docker image rm "$img" >/dev/null 2>&1 || true
  done
  docker image prune -f >/dev/null
fi
docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -E "^(g2cell/|$REF)" || true
df -h "$docker_root" | tail -1
echo "next: docker/cell.sh up    (or scripts/session.sh up)"
