"""Online SLAM for the mobile G2 (scene_g2cell_mobile.yaml, started with scripts/mobile_up.sh).

/lidar/livox_{front,back} -> cloud_merger -> /lidar/merged_full (base_link) -> pointcloud_to_laserscan -> /scan
-> slam_toolbox (online async) -> /map + map->odom.  /cmd_vel -> cmd_vel_bridge -> /cmd_twist.

  ros2 launch g2cell_slam mapping.launch.py [drive:=true] [rviz:=false]
  ros2 run nav2_map_server map_saver_cli -f <dir>/factory_room --ros-args -p map_subscribe_transient_local:=true

use_sim_time is false on purpose: the Genie Sim engine and render node stamp /tf, /odom and the
lidar clouds with wall-clock time (genie_sim_engine realtime_ros_node.cpp), while /clock is sim time.
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
    cfg = lambda f: os.path.join(share, "config", f)  # noqa: E731
    drive = LaunchConfiguration("drive")
    rviz = LaunchConfiguration("rviz")
    return LaunchDescription([
        DeclareLaunchArgument("drive", default_value="false", description="run the scripted mapping loop"),
        DeclareLaunchArgument("rviz", default_value="false", description="start RViz (needs a display)"),
        DeclareLaunchArgument("cloud", default_value="/lidar/merged_full"),
        # full-circle, motion-compensated union of /lidar/livox_front + /lidar/livox_back (see cloud_merger.py)
        Node(package="g2cell_slam", executable="cloud_merger", name="g2cell_cloud_merger", output="screen"),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
             name="pointcloud_to_laserscan", parameters=[cfg("pointcloud_to_laserscan.yaml")],
             remappings=[("cloud_in", LaunchConfiguration("cloud")), ("scan", "/scan")], output="screen"),
        # slam_toolbox 2.8 (Jazzy) is a lifecycle node: upstream's launch configures + activates it
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")),
            launch_arguments={"slam_params_file": cfg("slam_toolbox_mapping.yaml"),
                              "use_sim_time": "false", "autostart": "true"}.items()),
        Node(package="g2cell_slam", executable="cmd_vel_bridge", name="g2cell_cmd_vel_bridge", output="screen"),
        Node(package="g2cell_slam", executable="drive_path", name="g2cell_drive_path", output="screen",
             condition=IfCondition(drive)),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
             arguments=["-d", os.path.join(share, "rviz", "g2cell_slam.rviz")], condition=IfCondition(rviz)),
    ])
