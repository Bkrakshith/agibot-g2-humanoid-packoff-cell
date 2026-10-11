#!/usr/bin/env bash
# Stop a replay started by replay_rviz.sh (launch, replay_node, both robot_state_publishers, RViz).
pkill -INT -f '[r]eplay\.launch\.py' || true
sleep 2
pkill -f '[t]ools/viz/replay_node\.py' || true
pkill -f '[t]ools/viz/replay\.rviz' || true
pkill -f '[_]_ns:=/(recorded|policy) ' || true
pgrep -af '[r]eplay_node\.py --ros-args|[r]viz2 -d .*replay\.rviz|[r]obot_state_publisher .*__ns:=/(recorded|policy)' || echo "replay stopped"
