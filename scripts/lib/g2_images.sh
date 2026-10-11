# shellcheck shell=bash
# Image names and content-hash tags shared by scripts/build_and_push.sh, scripts/pull_on_server.sh,
# scripts/session.sh and docker/cell.sh. Source it; it defines functions and defaults only.
#
# Tags are computed from file contents, not git, so the workstation, the CPU builder and the GPU
# server agree as long as they hold the same files (sync_to_server.sh copies them):
#   base   geniesim-<GENIESIM_VERSION>-isaac<ISAACSIM6_VERSION>-<base hash>
#          base hash = genie_sim commit + Isaac Sim 6 version + docker/Dockerfile.geniesim
#   cell   cell-<cell hash>                cell hash = base hash + docker/Dockerfile.cell
#   baked  baked-<cell hash>-<ws hash>     ws hash   = every file under ros_ws/src (no caches)
#          baked-<cell hash>               moving tag: newest baked build for that cell hash
# Each kind also gets a moving "<kind>-latest" tag.
#
# Registry layout: ONE private repository, kinds told apart by tag prefix, so a Docker Hub free
# account (one private repo) works too:  ${REGISTRY}/${NAMESPACE}/${REPO}:<tag>

G2_ROOT=${G2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GENIESIM_SHA=${GENIESIM_SHA:-6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46}
GENIESIM_VERSION=${GENIESIM_VERSION:-3.2.0}
ISAACSIM6_VERSION=${ISAACSIM6_VERSION:-6.0.1.0}
REGISTRY=${REGISTRY:-ghcr.io}
NAMESPACE=${NAMESPACE:-}
REPO=${REPO:-g2cell}

# Local names the run scripts expect (docker/cell.sh uses LOCAL_CELL_IMAGE).
LOCAL_BASE_IMAGE=${LOCAL_BASE_IMAGE:-g2cell/geniesim:${GENIESIM_VERSION}-isaac${ISAACSIM6_VERSION}}
LOCAL_CELL_IMAGE=${LOCAL_CELL_IMAGE:-g2cell/cell:dev}

g2_sha12() { sha256sum | cut -c1-12; }

g2_base_hash() {
  { printf 'genie_sim=%s\nisaacsim6=%s\n' "$GENIESIM_SHA" "$ISAACSIM6_VERSION"
    cat "$G2_ROOT/docker/Dockerfile.geniesim"; } | g2_sha12
}

g2_cell_hash() {
  { g2_base_hash; cat "$G2_ROOT/docker/Dockerfile.cell"; } | g2_sha12
}

# Content hash of ros_ws/src: relative path + sha256 of every file, sorted, caches excluded.
g2_ws_hash() {
  ( cd "$G2_ROOT/ros_ws/src" \
    && find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info' \) -prune -o \
              -type f ! -name '*.pyc' -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum ) | g2_sha12
}

g2_repo() {
  if [[ -z $NAMESPACE ]]; then
    echo "NAMESPACE is not set (your GHCR user/org or Docker Hub user); pass --namespace or export NAMESPACE" >&2
    return 1
  fi
  # GHCR and Docker Hub need lower-case repository paths
  printf '%s/%s/%s\n' "$REGISTRY" "${NAMESPACE,,}" "${REPO,,}"
}

g2_base_tag()  { printf 'geniesim-%s-isaac%s-%s\n' "$GENIESIM_VERSION" "$ISAACSIM6_VERSION" "$(g2_base_hash)"; }
g2_cell_tag()  { printf 'cell-%s\n' "$(g2_cell_hash)"; }
g2_baked_tag() { printf 'baked-%s-%s\n' "$(g2_cell_hash)" "$(g2_ws_hash)"; }
g2_baked_cell_tag() { printf 'baked-%s\n' "$(g2_cell_hash)"; }
