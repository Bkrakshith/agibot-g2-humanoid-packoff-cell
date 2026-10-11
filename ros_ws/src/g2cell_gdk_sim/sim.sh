# Source this file INSIDE the g2cell container to run GDK Python code against Genie Sim.
# Afterwards `import agibot_gdk` resolves to this package's sim-backed agibot_gdk 3.3.8 API
# (the analogue of agibot-gdk-training's fake.sh). On the real robot source the vendor GDK
# environment instead (agibot-gdk-training's real.sh / robot.sh), never this file.
#
#   source /g2cell_ws/src/g2cell_gdk_sim/sim.sh [TRAINING_DIR]
#
# TRAINING_DIR (default /g2cell_ws/third_party/agibot-gdk-training) is put on PYTHONPATH as well,
# so the training exercises can import gdk_training. Its fake/ folder is NOT added.
# Build first: cd /g2cell_ws && colcon build --packages-select g2cell_gdk_sim

_g2ws="${G2CELL_WS:-/g2cell_ws}"
[ -f /workspace/devel/setup.bash ] && source /workspace/devel/setup.bash
if [ -f "$_g2ws/install/setup.bash" ]; then
    source "$_g2ws/install/setup.bash"
else
    echo "sim.sh: $_g2ws/install/setup.bash not found; build the workspace first" >&2
fi
_gdk_site="$(ls -d "$_g2ws"/install/g2cell_gdk_sim/lib/python3*/site-packages 2>/dev/null | head -n 1)"
if [ -z "$_gdk_site" ] || [ ! -d "$_gdk_site/agibot_gdk" ]; then
    echo "sim.sh: agibot_gdk is not installed; run: cd $_g2ws && colcon build --packages-select g2cell_gdk_sim" >&2
    unset _g2ws _gdk_site
    return 1
fi
export GDK_TRAINING_DIR="${1:-${GDK_TRAINING_DIR:-$_g2ws/third_party/agibot-gdk-training}}"
# agibot_gdk first, so no other agibot_gdk (vendor wheel or the training fake) can shadow it
export PYTHONPATH="$_gdk_site:$GDK_TRAINING_DIR${PYTHONPATH:+:$PYTHONPATH}"
export GDK_SIM_SCENE="${GDK_SIM_SCENE:-$_g2ws/config/scenes/scene_g2cell.yaml}"
# Genie Sim stamps everything with wall time; never run GDK code with sim time.
unset ROS_USE_SIM_TIME
echo "agibot_gdk: $(python3 -c 'import agibot_gdk; print(agibot_gdk.__version__, agibot_gdk.__file__)' 2>&1)"
echo "training:   $GDK_TRAINING_DIR $( [ -d "$GDK_TRAINING_DIR/gdk_training" ] || echo '(missing: copy the repo there)')"
echo "scene:      $GDK_SIM_SCENE   ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
unset _g2ws _gdk_site
