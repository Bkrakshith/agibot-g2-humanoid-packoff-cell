"""Teleop + recording session: ros2 launch g2cell_bringup teleop_record.launch.py [start_moveit:=false]

Starts MoveIt move_group (IK only, use_ros2_control:=false) unless stage1_up.sh already runs it,
and prints how to start the keyboard teleop. The teleop is not started here by default: ros2
launch does not hand a node the terminal's stdin, so the keys only work from `ros2 run` (or with
start_teleop:=true terminal_prefix:='xterm -e' on a machine with a display).
Do not run this together with cell.launch.py's motion_server: both would publish /joint_command.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

HELP = """
==== g2cell teleop + recording ====
1. Check: `ros2 topic info /joint_command` shows 0 publishers (no motion_server / demo running).
2. In a terminal (tmux window 'teleop'):
     ros2 run g2cell_teleop keyboard_teleop --ros-args \\
       --params-file $(ros2 pkg prefix g2cell_teleop)/share/g2cell_teleop/config/teleop_g2.yaml \\
       -p use_sim_time:=true -p scene:=scene_g2cell -p seed:=0
   Wait for the key help (~1 min after the sim is up).
3. Keys: 1/2 arm, w s a d q e move, i k j l u o rotate, g/h close/open gripper,
   r start/stop recording (MCAP -> HDF5 in /g2cell_output/episodes), n/f mark success/failure,
   0 home, p pause, x quit.
4. Health: ros2 run g2cell_bringup check_stack   (/g2cell/* entries are expected to be missing here)
"""


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument("start_moveit", default_value="true",
                              description="false when stage1_up.sh already runs move_group"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("start_teleop", default_value="false",
                              description="include g2cell_teleop teleop.launch.py (needs terminal_prefix)"),
        DeclareLaunchArgument("terminal_prefix", default_value="xterm -e"),
        DeclareLaunchArgument("scene", default_value="scene_g2cell"),
        DeclareLaunchArgument("seed", default_value="0"),
    ]
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("genie_sim_moveit"),
                                                   "launch", "wbc.launch.py")),
        launch_arguments={"arm": "crs", "gripper": "omnipicker", "use_ros2_control": "false",
                          "use_sim_time": lc("use_sim_time")}.items(),
        condition=IfCondition(lc("start_moveit")))
    teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("g2cell_teleop"),
                                                   "launch", "teleop.launch.py")),
        launch_arguments={"terminal_prefix": lc("terminal_prefix"), "scene": lc("scene"),
                          "seed": lc("seed")}.items(),
        condition=IfCondition(lc("start_teleop")))
    return LaunchDescription(args + [LogInfo(msg=HELP), moveit, teleop])
