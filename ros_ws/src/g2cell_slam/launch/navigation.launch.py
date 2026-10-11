"""Localization (map_server + AMCL) + Nav2 on a saved map, for the mobile G2.

  ros2 launch g2cell_slam navigation.launch.py [map:=<yaml>] [rviz:=false]
  ros2 run g2cell_slam nav_goal --ros-args -p x:=1.6 -p y:=0.0 -p yaw_deg:=180.0

AMCL needs an initial pose: nav_goal publishes /initialpose from the ground-truth /odom
(set_initial_pose:=true), or use RViz "2D Pose Estimate". use_sim_time false: see mapping.launch.py.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("g2cell_slam")
    nav2 = os.path.join(get_package_share_directory("nav2_bringup"), "launch")
    cfg = lambda f: os.path.join(share, "config", f)  # noqa: E731
    common = {"use_sim_time": "false", "autostart": "true",
              "params_file": LaunchConfiguration("params_file")}
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=os.path.join(share, "maps", "factory_room.yaml")),
        DeclareLaunchArgument("params_file", default_value=cfg("nav2_params.yaml")),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("cloud", default_value="/lidar/merged_full"),
        # full-circle, motion-compensated union of /lidar/livox_front + /lidar/livox_back (see cloud_merger.py)
        Node(package="g2cell_slam", executable="cloud_merger", name="g2cell_cloud_merger", output="screen"),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
             name="pointcloud_to_laserscan", parameters=[cfg("pointcloud_to_laserscan.yaml")],
             remappings=[("cloud_in", LaunchConfiguration("cloud")), ("scan", "/scan")], output="screen"),
        Node(package="g2cell_slam", executable="cmd_vel_bridge", name="g2cell_cmd_vel_bridge", output="screen"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav2, "localization_launch.py")),
                                 launch_arguments={**common, "map": LaunchConfiguration("map")}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav2, "navigation_launch.py")),
                                 launch_arguments=common.items()),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
             arguments=["-d", os.path.join(share, "rviz", "g2cell_slam.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
