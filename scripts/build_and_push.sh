#!/usr/bin/env bash
# Build the G2 cell images on a cheap CPU-only machine and push them to a PRIVATE registry, so the
# billed GPU server only pulls (scripts/pull_on_server.sh). No GPU is needed to build.
#
#   scripts/build_and_push.sh --namespace <ghcr-user-or-org>                 # base + cell + baked -> ghcr.io
#   scripts/build_and_push.sh --registry docker.io --namespace <hub-user>    # Docker Hub private repo
#   scripts/build_and_push.sh --namespace me --targets cell,baked            # base already pushed
#   scripts/build_and_push.sh --namespace me --print-tags                    # show the tags, build nothing
#
# Options (or the env var in brackets):
#   --registry R      ghcr.io (default) or docker.io, or any registry          [REGISTRY]
#   --namespace N     GHCR user/org or Docker Hub user (required)              [NAMESPACE]
#   --repo NAME       repository name (default g2cell); one repo, tag prefixes [REPO]
#   --targets LIST    comma list of base,cell,baked (default all three)        [TARGETS]
#   --compression C   zstd (default; faster to unpack on the GPU box) or gzip  [COMPRESSION]
#   --force           rebuild and push even if the content-hash tag exists     [FORCE=1]
#   --print-tags      print the image references and exit
#
# Before the first run, log in ONCE on this machine (credentials stay in ~/.docker/config.json or
# your credential helper; this script never sees or stores them):
#   GHCR:       create a classic PAT with write:packages (+ read:packages), then
#               echo "$GHCR_PAT" | docker login ghcr.io -u <github-user> --password-stdin
#   Docker Hub: create an access token (Read & Write), make the repo <user>/g2cell PRIVATE, then
#               docker login -u <hub-user>
# The images contain the NVIDIA Isaac Sim base: keep the package/repo PRIVATE (licence forbids
# public redistribution). New GHCR packages are private by default; check under
# github.com/<user>?tab=packages after the first push.
#
# Requirements: Docker with buildx, ~150 GB free disk, genie_sim at the pinned commit with LFS files
# (scripts/server_setup.sh --builder sets all of this up). Tags are content hashes, see
# scripts/lib/g2_images.sh. The build runs in a docker-container buildx builder and pushes straight
# from its cache: nothing is loaded into the local image store, so the disk holds one copy, not two.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
G2_ROOT=$ROOT
# shellcheck source-path=SCRIPTDIR source=lib/g2_images.sh
source "$HERE/lib/g2_images.sh"

GENIESIM_DIR=${GENIESIM_DIR:-$(cd "$ROOT/.." && pwd)/genie_sim}
TARGETS=${TARGETS:-base,cell,baked}
COMPRESSION=${COMPRESSION:-zstd}
BUILDER=${BUILDER:-g2cell-builder}
FORCE=${FORCE:-0}
PRINT_ONLY=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry) REGISTRY=$2; shift 2 ;;
    --namespace) NAMESPACE=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --targets) TARGETS=$2; shift 2 ;;
    --compression) COMPRESSION=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --print-tags) PRINT_ONLY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n==> %s\n' "$*"; }
want() { [[ ,$TARGETS, == *,$1,* ]]; }

REF=$(g2_repo) || exit 2
BASE_TAG=$(g2_base_tag)
CELL_TAG=$(g2_cell_tag)
BAKED_TAG=$(g2_baked_tag)
BAKED_CELL_TAG=$(g2_baked_cell_tag)

if [[ $PRINT_ONLY == 1 ]]; then
  printf 'base   %s:%s\ncell   %s:%s\nbaked  %s:%s  (also :%s)\n' \
    "$REF" "$BASE_TAG" "$REF" "$CELL_TAG" "$REF" "$BAKED_TAG" "$BAKED_CELL_TAG"
  exit 0
fi

# zstd unpacks several times faster than gzip on the GPU box (Docker >= 23 pulls it). The base
# also recompresses NVIDIA's gzip layers (force-compression); cell/baked reuse the base's
# registry layers as they are and only compress their own.
case $COMPRESSION in
  zstd) OUT_COMP="compression=zstd,compression-level=3,oci-mediatypes=true" ;;
  gzip) OUT_COMP="compression=gzip,oci-mediatypes=false" ;;
  *) die "--compression must be zstd or gzip" ;;
esac
FORCE_COMP=""
[[ $COMPRESSION == zstd ]] && FORCE_COMP=",force-compression=true"

# ---------------------------------------------------------------------------------------------
log "preflight"
docker info >/dev/null 2>&1 || die "cannot talk to docker (daemon down, or not in the docker group?)"
docker buildx version >/dev/null 2>&1 || die "docker buildx missing (apt install docker-buildx-plugin)"
if ! grep -q "\"${REGISTRY}" "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null \
   && ! grep -q '"credsStore"' "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null; then
  echo "warning: no saved login for $REGISTRY found; run 'docker login $REGISTRY' first (see --help)" >&2
fi

if want base || want baked; then
  [[ -d $GENIESIM_DIR/.git ]] || die "no genie_sim checkout at $GENIESIM_DIR (scripts/bootstrap_host.sh clones it)"
  head=$(git -C "$GENIESIM_DIR" rev-parse HEAD)
  [[ $head == "$GENIESIM_SHA" ]] || die "genie_sim is at $head, expected $GENIESIM_SHA"
  if grep -lq "git-lfs.github.com/spec" \
      "$GENIESIM_DIR"/source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/g2/meshes/* 2>/dev/null; then
    die "genie_sim LFS files are still pointers: run 'git lfs pull' in $GENIESIM_DIR"
  fi
fi

docker_root=$(docker info --format '{{.DockerRootDir}}')
free=$(df -BG --output=avail "$docker_root" | tail -1 | tr -dc '0-9')
echo "free on $docker_root: ${free} GB"
[[ $free -ge 120 ]] || echo "warning: < 120 GB free; a full base build needs ~100 GB (layers + pip cache + export)" >&2

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "creating buildx builder $BUILDER (docker-container driver)"
  docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
fi
docker buildx inspect --bootstrap "$BUILDER" >/dev/null

exists() { docker buildx imagetools inspect "$1" >/dev/null 2>&1; }

# build <kind> <output extras> <primary tag> <extra tags...> -- <buildx args...>
build() {
  local kind=$1 extra=$2 tag=$3; shift 3
  local tags=(-t "$REF:$tag")
  while [[ $1 != -- ]]; do tags+=(-t "$REF:$1"); shift; done
  shift
  if [[ $FORCE != 1 ]] && exists "$REF:$tag"; then
    echo "$kind: $REF:$tag already in the registry; skipping (--force to rebuild)"
    return 0
  fi
  log "$kind -> $REF:$tag"
  local t0=$SECONDS
  docker buildx build --builder "$BUILDER" --provenance=false --sbom=false \
    "${tags[@]}" \
    --output "type=image,push=true,${OUT_COMP}${extra}" \
    "$@"
  echo "$kind pushed in $(( (SECONDS - t0) / 60 )) min"
}

# ---------------------------------------------------------------------------------------------
if want base; then
  build base "$FORCE_COMP" "$BASE_TAG" geniesim-latest -- \
    -f "$ROOT/docker/Dockerfile.geniesim.split" \
    --build-arg ISAACSIM6_VERSION="$ISAACSIM6_VERSION" \
    "$GENIESIM_DIR"
fi

if want cell || want baked; then
  exists "$REF:$BASE_TAG" || die "base $REF:$BASE_TAG is not in the registry; add 'base' to --targets"
fi

if want cell; then
  build cell "" "$CELL_TAG" cell-latest -- \
    -f "$ROOT/docker/Dockerfile.cell" --target cell \
    --build-arg BASE_IMAGE="$REF:$BASE_TAG" \
    "$ROOT/docker"
fi

if want baked; then
  build baked "" "$BAKED_TAG" "$BAKED_CELL_TAG" baked-latest -- \
    -f "$ROOT/docker/Dockerfile.cell" --target baked \
    --build-arg BASE_IMAGE="$REF:$BASE_TAG" \
    --build-arg GENIESIM_SHA="$GENIESIM_SHA" \
    --build-arg G2CELL_WS_HASH="$(g2_ws_hash)" \
    --build-context geniesim_ros="$GENIESIM_DIR/source/geniesim_ros/src/ros_ws" \
    --build-context g2cell_src="$ROOT/ros_ws/src" \
    "$ROOT/docker"
fi

log "done"
printf 'base   %s:%s\ncell   %s:%s\nbaked  %s:%s\n' "$REF" "$BASE_TAG" "$REF" "$CELL_TAG" "$REF" "$BAKED_TAG"
cat <<EOF
On the GPU server (same project files, synced with scripts/sync_to_server.sh):
  docker login $REGISTRY     # read-only token is enough there
  REGISTRY=$REGISTRY NAMESPACE=$NAMESPACE scripts/pull_on_server.sh
Free this machine's disk when done:  docker buildx prune --builder $BUILDER -af
EOF
