#!/usr/bin/env bash
# Start the MOBILE G2 scene (SLAM/Nav use case) inside the g2cell container, tmux session "g2cell":
#   sim (lidar launcher) -> relay (head cam + lidar body) -> freecam -> foxglove -> servo (chassis)
# Run inside the container:  bash /g2cell_ws/scripts/mobile_up.sh [scene.yaml]
#   WITH_MOVEIT=1 also starts move_group (IK/FK in base_link) for arm work on the mobile base
#   (Layout v2 drive-in + packing demo); default 0 = the SLAM stack as before.
#   PROFILE=collect|teleop|showcase: render cost profile (see stage1_up.sh). With a PROFILE the lidars are
#   rendered only for the SLAM stack (LIDAR=1 by default without WITH_MOVEIT=1); the packing stack
#   (WITH_MOVEIT=1, closed loop on ground-truth /odom) runs without them (LIDAR=0). Override with LIDAR=0|1.
#   Without a PROFILE the lidars are always on (as before).
#   OBJ_RELAY=1|0 (default 1) is passed through to stage1_up.sh (object pose relay to the renderer).
# Then: ros2 launch g2cell_slam mapping.launch.py   (see ros_ws/src/g2cell_slam/README.md)
set -euo pipefail
SCENE=${1:-/g2cell_ws/config/scenes/scene_g2cell_mobile.yaml}
SRC="source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash"
# 1) lidar prims into the staged robot USD (seeded from the cell scene's converted robot on first use)
python3 /g2cell_ws/scripts/inject_lidars.py --scene "$SCENE" --seed-from scene_g2cell
# 2) the usual stack, with the lidar-enabled launcher, the lidar body relay, no MoveIt, and
#    fake_slam:=false (otherwise the engine publishes an identity map->odom that fights slam_toolbox/AMCL)
LIDAR=${LIDAR:-$([ "${WITH_MOVEIT:-0}" = 1 ] && echo 0 || echo 1)}
LIDAR=$LIDAR LAUNCHER=/g2cell_ws/config/launchers/launcher_ovrtx_isaac_physx_lidar.yaml \
RELAY_ARGS="--lidar-body base_link" NO_MOVEIT=$([ "${WITH_MOVEIT:-0}" = 1 ] && echo 0 || echo 1) SIM_ARGS="fake_slam:=false" \
  bash /g2cell_ws/scripts/stage1_up.sh "$SCENE"
# 3) chassis servo: /cmd_twist (geometry_msgs/Twist, base frame) -> /cmd_4ws (steer pos + drive vel).
#    GeneralServo (plain 4WS inverse kinematics) instead of the default OptimalServo (MPC): measured, the MPC
#    turns a spin request issued right after driving into a 0.6 m arc with ~36 deg of yaw in 3 s, and
#    stalls (7 deg in 3 s) in other cases; GeneralServo spins 75 deg in 3 s with <= 3 cm drift.
tmux new-window -t g2cell -n servo \
  "$SRC && ros2 run genie_sim_controllers genie_sim_chassis_servo_node --ros-args -r __node:=servo_node \
     --params-file \$(ros2 pkg prefix genie_sim_controllers)/share/genie_sim_controllers/config/genie_g2.yaml \
     -p use_sim_time:=true -p default_servo:=GeneralServo 2>&1 | tee /g2cell_output/servo.log"
echo "mobile scene up${PROFILE:+ (PROFILE=$PROFILE, lidar $LIDAR)} (windows: sim ${PROFILE:+render }relay [freecam] foxglove teleop servo)"
