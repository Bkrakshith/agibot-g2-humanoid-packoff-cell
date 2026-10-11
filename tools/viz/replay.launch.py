"""Launch: replay_node + two robot_state_publishers (recorded/, policy/) + RViz.

    ros2 launch tools/viz/replay.launch.py npz:=/path/replay.npz [speed:=1.0] [loop:=true]
        [rviz:=true] [urdf:=...]
Use replay_rviz.sh, which exports the .npz and sets up the clean ROS env (domain 7).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_URDF = os.path.realpath(os.path.join(HERE, "../../../g2_viz_ws/g2_crs_omnipicker.urdf"))


def _setup(context):
    urdf = LaunchConfiguration("urdf").perform(context)
    with open(urdf) as f:
        desc = f.read()
    acts = []
    for pfx in ("recorded", "policy"):
        acts.append(Node(package="robot_state_publisher", executable="robot_state_publisher",
                         namespace=pfx, name="robot_state_publisher", output="log",
                         parameters=[{"robot_description": desc, "frame_prefix": f"{pfx}/",
                                      "publish_frequency": 60.0}]))
    acts.append(ExecuteProcess(
        cmd=["/usr/bin/python3", os.path.join(HERE, "replay_node.py"), "--ros-args",
             "-p", "npz:=" + LaunchConfiguration("npz").perform(context),
             "-p", "urdf:=" + urdf,
             "-p", "speed:=" + str(float(LaunchConfiguration("speed").perform(context))),
             "-p", "loop:=" + LaunchConfiguration("loop").perform(context),
             "-p", "ghost_offset_y:=" + str(float(LaunchConfiguration("ghost_offset_y").perform(context)))],
        output="screen", name="g2_replay"))
    acts.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="log",
                     arguments=["-d", os.path.join(HERE, "replay.rviz")],
                     condition=IfCondition(LaunchConfiguration("rviz"))))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("npz"),
        DeclareLaunchArgument("urdf", default_value=DEFAULT_URDF),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("ghost_offset_y", default_value="0.0"),
        OpaqueFunction(function=_setup),
    ])
