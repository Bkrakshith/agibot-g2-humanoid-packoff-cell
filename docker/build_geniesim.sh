#!/usr/bin/env bash
# Build the Genie Sim base image (Isaac Sim 5.1 + pinned Isaac Sim 6.0.x + ROS 2 Jazzy).
# Same command on this workstation and on the Paperspace server.
#
#   GENIESIM_DIR   genie_sim checkout (default: ../../genie_sim), with LFS files pulled
#   GENIESIM_SHA   commit it must be at (checked, not changed)
#   IMAGE          tag to build
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENIESIM_DIR=${GENIESIM_DIR:-$(cd "$HERE/../../genie_sim" && pwd)}
GENIESIM_SHA=${GENIESIM_SHA:-6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46}
ISAACSIM6_VERSION=${ISAACSIM6_VERSION:-6.0.1.0}
IMAGE=${IMAGE:-g2cell/geniesim:3.2.0-isaac${ISAACSIM6_VERSION}}

actual=$(git -C "$GENIESIM_DIR" rev-parse HEAD)
if [[ "$actual" != "$GENIESIM_SHA" ]]; then
  echo "genie_sim is at $actual, expected $GENIESIM_SHA" >&2
  exit 1
fi
if grep -q "git-lfs.github.com/spec" \
    "$GENIESIM_DIR"/source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/g2/meshes/* 2>/dev/null; then
  echo "G2 meshes are still LFS pointers: run 'git lfs pull' in $GENIESIM_DIR" >&2
  exit 1
fi

DOCKER_BUILDKIT=1 docker build \
  -f "$HERE/Dockerfile.geniesim" \
  --build-arg ISAACSIM6_VERSION="$ISAACSIM6_VERSION" \
  -t "$IMAGE" \
  "$GENIESIM_DIR"
echo "built $IMAGE"
