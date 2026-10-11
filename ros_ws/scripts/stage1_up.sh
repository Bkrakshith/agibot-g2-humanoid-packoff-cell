#!/usr/bin/env bash
# Start the Stage 1 stack inside the g2cell container, one tmux window per process:
#   sim (headless G2 scene) -> head-camera TF relay -> MoveIt IK (move_group only) -> Foxglove
# Then attach with `tmux attach -t g2cell` and run the teleop in a new window:
#   ros2 run g2cell_teleop keyboard_teleop --ros-args -p use_sim_time:=true
# Run inside the container:  bash /g2cell_ws/scripts/stage1_up.sh [scene.yaml]
#   PROFILE=collect|teleop|showcase picks the render cost profile (docs/OPTIMISATION.md "Sim runtime"):
#     collect   head RGB+D 640x400 at 30 Hz, no FreeCam   (teleop data collection, policy eval)
#     teleop    head RGB+D 640x400 at 15 Hz, no FreeCam   (interactive driving, lowest GPU load)
#     showcase  head + FreeCam 1280x720, 30 Hz target, upstream render mode (videos)
#   OBJ_RELAY=1|0 (default 1): stream bag/carton/tote poses from physics to the renderer (see below).
#   GRASP_ASSIST=1|0 (default 1, needs OBJ_RELAY=1): sticky grasp for the rigid bags (a closing gripper over a bag
#     carries it until the gripper opens; g2cell_bringup/object_relay.py GraspAssist).
#   unset = the upstream render path, unchanged. Per-run overrides: RENDER_MODE (raster|rtpt|pathtrace|offline),
#   RENDER_FPS, FREECAM (on|off), LIDAR (0|1; set by mobile_up.sh). With a PROFILE the render node runs in its own
#   tmux window "render" (scripts/render_profile.py) and logs to render.log.
# Logs: /g2cell_output/{sim,render,relay,moveit,foxglove}.log
set -euo pipefail

SCENE=${1:-/g2cell_ws/config/scenes/scene_pnp_g2_op_cam.yaml}
# optional overrides (used by mobile_up.sh): launcher YAML, extra relay args, skip MoveIt
LAUNCHER=${LAUNCHER:-launcher_ovrtx_isaac_physx}
RELAY_ARGS=${RELAY_ARGS:-}
NO_MOVEIT=${NO_MOVEIT:-0}
SIM_ARGS=${SIM_ARGS:-}   # extra app.launch.py args, e.g. fake_slam:=false
PROFILE=${PROFILE:-}
LIDAR=${LIDAR:-0}
# SIM_CPUS: pin the sim + render processes to these CPUs (taskset), e.g. 0-15. Default with a PROFILE: "auto" =
# on a hybrid CPU (P/E cores: per-CPU max clocks differ by > 15 %) pin to the fast cores, otherwise no pinning.
# Why: the physics loop has ~0.5 ms slack per 10 ms tick; ticks scheduled on an i9-14900K E-core overran
# ("loop fell behind"). "none" disables pinning.
SIM_CPUS=${SIM_CPUS:-$([ -n "$PROFILE" ] && echo auto || echo none)}
if [ "$SIM_CPUS" = auto ]; then
  SIM_CPUS=$(for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
      c=${f#/sys/devices/system/cpu/cpu}; echo "${c%%/*} $(cat "$f")"; done 2>/dev/null |
    awk '{c[NR]=$1; f[NR]=$2; if ($2>mx) mx=$2; if (mn==""||$2<mn) mn=$2}
         END {if (NR==0 || mn >= 0.85*mx) exit; s=""; for (i=1;i<=NR;i++) if (f[i] > (mn+mx)/2) s=s (s==""?"":",") c[i]; print s}')
  [ -z "$SIM_CPUS" ] && SIM_CPUS=none
fi
PIN=""
[ "$SIM_CPUS" != none ] && PIN="taskset -c $SIM_CPUS" && echo "pinning sim + render to CPUs $SIM_CPUS (SIM_CPUS=none to disable)"
# cost-only env for every profile: freeze the engine's (unused, headless) Kit viewport, and never evict the
# driver shader cache (~/.cache/nvidia, host-mounted by docker/cell.sh)
SIM_ENV="ISAACSIM_VIEWPORT_UPDATES=0 __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1"
RENDER_LOG=sim
if [ -n "$PROFILE" ]; then
  case "$PROFILE" in collect|teleop|showcase) ;; *) echo "PROFILE must be collect|teleop|showcase" >&2; exit 2;; esac
  # upstream launcher minus its render node (render_profile.py starts it with our settings)
  LAUNCHER=/g2cell_ws/config/launchers/launcher_ovrtx_isaac_physx_norender.yaml
  RENDER_LOG=render
  FREECAM=${FREECAM:-$([ "$PROFILE" = showcase ] && echo on || echo off)}
  RP_ARGS="--profile $PROFILE --lidar $LIDAR --freecam $FREECAM"
  [ -n "${RENDER_MODE:-}" ] && RP_ARGS="$RP_ARGS --mode $RENDER_MODE"
  [ -n "${RENDER_FPS:-}" ] && RP_ARGS="$RP_ARGS --fps $RENDER_FPS"
else
  SIM_ENV=""
  FREECAM=on
fi
SESSION=g2cell
SRC="source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash"
OUT=/g2cell_output

# OBJ_RELAY=1 (default): run the engine as g2cell_bringup/g2cell_engine_isaacsim.py = upstream engine + a
# post-step hook that streams the poses of the scene's dynamic rigid bodies (bags, cartons, totes) to /tf_render,
# so they render where PhysX puts them (upstream relays robot links only). OBJ_RELAY=0 = upstream engine.
# Rate/keyframe: G2CELL_OBJ_RELAY_HZ (30), G2CELL_OBJ_RELAY_KEYFRAME_S (2); see g2cell_bringup/object_relay.py.
OBJ_RELAY=${OBJ_RELAY:-1}
GRASP_ASSIST=${GRASP_ASSIST:-1}
if [ "$OBJ_RELAY" = 1 ]; then
  case "$LAUNCHER" in
    */*) LAUNCHER_SRC=$LAUNCHER ;;
    *) LAUNCHER_SRC=/workspace/devel/share/genie_sim_bringup/config/${LAUNCHER%.yaml}.yaml ;;
  esac
  LAUNCHER=/tmp/g2cell_launcher_objrelay.yaml
  bash -c "$SRC && python3 -m g2cell_bringup.object_relay make-launcher $LAUNCHER_SRC $LAUNCHER" >/dev/null
  echo "object pose relay on (engine g2cell_engine_isaacsim.py, launcher from $LAUNCHER_SRC); OBJ_RELAY=0 to disable"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; stop it first: tmux kill-session -t $SESSION" >&2
  exit 1
fi

# truncate old logs first: otherwise the readiness grep below can match the previous run's
# "Render Stats" before tee recreates sim.log
for f in sim render relay moveit foxglove freecam; do : > "$OUT/$f.log"; done
date +%s.%N > "$OUT/perf_t0"   # launch time; tools/sim_perf.py reports time-to-first-frame from it

tmux new-session -d -s "$SESSION" -n sim \
  "$SRC && G2CELL_GRASP_ASSIST=$GRASP_ASSIST $SIM_ENV $PIN ros2 launch genie_sim_bringup app.launch.py scene:=$SCENE \
     launcher_config:=$LAUNCHER headless:=true $SIM_ARGS 2>&1 | tee $OUT/sim.log"
if [ -n "$PROFILE" ]; then
  # waits for this launch's assemble_scene output, then execs render_ovrtx with the profile
  tmux new-window -t "$SESSION" -n render \
    "$SRC && $SIM_ENV $PIN python3 /g2cell_ws/scripts/render_profile.py --scene $SCENE $RP_ARGS 2>&1 | tee $OUT/render.log"
fi

# A cold shader cache on a new GPU/driver can take >6 min before the first frame (391 s measured on the
# Paperspace A6000, 8 vCPU); warm starts take ~15-20 s. READY_TIMEOUT_S overrides the wait.
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}
echo "waiting up to ${READY_TIMEOUT_S} s for the sim to render (first launch compiles shaders) ..."
for _ in $(seq 1 $((READY_TIMEOUT_S / 5))); do
  grep -q "Render Stats" "$OUT/$RENDER_LOG.log" 2>/dev/null && break
  if grep -qE "Traceback|process has died" "$OUT/sim.log" "$OUT/render.log" 2>/dev/null; then
    echo "sim failed; see $OUT/sim.log and $OUT/render.log" >&2
    exit 1
  fi
  sleep 5
done
grep -q "Render Stats" "$OUT/$RENDER_LOG.log" || { echo "sim not rendering after ${READY_TIMEOUT_S} s; see $OUT/$RENDER_LOG.log" >&2; exit 1; }

# The render node never matches the engine's /tf_render frame names, so without this relay
# the head camera stays at the world origin (see the comment in the scene YAML).
tmux new-window -t "$SESSION" -n relay \
  "$SRC && python3 /g2cell_ws/scripts/head_cam_tf_relay.py --scene $SCENE $RELAY_ARGS 2>&1 | tee $OUT/relay.log"
# Genie Sim's Isaac PhysX path ignores the YAML viewer_camera; aim the FreeCam ourselves (1 Hz latch).
# Not needed when the profile does not render the FreeCam.
if grep -q "^viewer_camera:" "$SCENE" && [ "$FREECAM" = on ]; then
  tmux new-window -t "$SESSION" -n freecam \
    "$SRC && python3 /g2cell_ws/scripts/free_cam_pose.py --scene $SCENE 2>&1 | tee $OUT/freecam.log"
fi
if [ "$NO_MOVEIT" != 1 ]; then
tmux new-window -t "$SESSION" -n moveit \
  "$SRC && ros2 launch genie_sim_moveit wbc.launch.py arm:=crs gripper:=omnipicker \
     use_ros2_control:=false use_sim_time:=true 2>&1 | tee $OUT/moveit.log"
fi
tmux new-window -t "$SESSION" -n foxglove \
  "$SRC && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 2>&1 | tee $OUT/foxglove.log"
tmux new-window -t "$SESSION" -n teleop "$SRC && exec bash"

echo "up${PROFILE:+ (PROFILE=$PROFILE)}: tmux attach -t $SESSION   (windows: sim ${PROFILE:+render }relay [freecam] moveit foxglove teleop)"
