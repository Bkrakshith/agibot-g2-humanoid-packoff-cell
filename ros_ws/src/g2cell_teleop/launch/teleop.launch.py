"""G2 teleop: ros2 launch g2cell_teleop teleop.launch.py device:=keyboard|spacemouse scene:=S seed:=N

ros2 launch does not hand the node your terminal's stdin, so the keyboard (and the r/n/f/0/p
episode keys) only works through terminal_prefix (e.g. terminal_prefix:='xterm -e') or with
`ros2 run g2cell_teleop keyboard_teleop --ros-args --params-file ...` (see the README).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("g2cell_teleop")
    args = [
        DeclareLaunchArgument("device", default_value="keyboard", choices=["keyboard", "spacemouse"]),
        DeclareLaunchArgument("spacemouse_source", default_value="topic", choices=["topic", "pyspacemouse"]),
        DeclareLaunchArgument("scene", default_value="scene_pnp_g2_op", description="stored in each episode"),
        DeclareLaunchArgument("seed", default_value="-1", description="stored in each episode"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "teleop_g2.yaml")),
        DeclareLaunchArgument("launch_spacenav", default_value="false",
                              description="also start spacenav_node here (needs spacenavd + the device on this machine)"),
        DeclareLaunchArgument("terminal_prefix", default_value="",
                              description="e.g. 'xterm -e' to give the keyboard its own terminal"),
    ]
    lc = LaunchConfiguration
    teleop = Node(
        package="g2cell_teleop",
        executable="keyboard_teleop",
        name="g2cell_keyboard_teleop",
        output="screen",
        emulate_tty=True,
        prefix=lc("terminal_prefix"),
        parameters=[
            lc("params_file"),
            {
                "device": lc("device"),
                "spacemouse_source": lc("spacemouse_source"),
                "scene": ParameterValue(lc("scene"), value_type=str),
                "seed": ParameterValue(lc("seed"), value_type=int),
            },
        ],
    )
    spacenav = Node(package="spacenav", executable="spacenav_node", name="spacenav", output="screen",
                    condition=IfCondition(lc("launch_spacenav")))
    return LaunchDescription(args + [spacenav, teleop])
