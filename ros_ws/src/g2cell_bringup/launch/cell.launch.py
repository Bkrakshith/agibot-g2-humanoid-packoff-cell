"""Cell stack (everything except the sim): ros2 launch g2cell_bringup cell.launch.py [infra:=true]

Starts motion_server (MoveArm, the single /joint_command publisher), pick_bag_server (PickBag)
and cell_manager (CellStatus + services). With infra:=true it also starts MoveIt move_group
(IK/FK only, use_ros2_control:=false), the head-camera TF relay, the FreeCam aimer and Foxglove.

The sim itself is NOT started here: it stays with scripts/stage1_up.sh, because (a) only one
operator may run or restart it, (b) it takes minutes to compile shaders and must be
rendering before the relay/MoveIt are useful (stage1_up.sh waits for "Render Stats"), and
(c) it should outlive restarts of the use-case nodes. stage1_up.sh already starts the relay,
FreeCam, MoveIt and Foxglove, so after stage1_up.sh use the default infra:=false.
Quit the keyboard teleop / pick_place_demo first: motion_server refuses to start while another
node publishes /joint_command.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lc = LaunchConfiguration
    share = get_package_share_directory("g2cell_bringup")
    args = [
        DeclareLaunchArgument("infra", default_value="false",
                              description="also start move_group, head-cam relay, FreeCam aimer, Foxglove "
                                          "(false when stage1_up.sh already runs them)"),
        DeclareLaunchArgument("scene", default_value="/g2cell_ws/config/scenes/scene_g2cell.yaml"),
        DeclareLaunchArgument("scripts_dir", default_value="/g2cell_ws/scripts"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "cell_params.yaml")),
        DeclareLaunchArgument("start_tasks", default_value="true",
                              description="pick_bag_server + cell_manager (false = motion_server only)"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
    ]
    sim_time = {"use_sim_time": PythonExpression(["'", lc("use_sim_time"), "' == 'true'"])}

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("genie_sim_moveit"),
                                                   "launch", "wbc.launch.py")),
        launch_arguments={"arm": "crs", "gripper": "omnipicker", "use_ros2_control": "false",
                          "use_sim_time": lc("use_sim_time")}.items())
    relay = ExecuteProcess(cmd=["python3", [lc("scripts_dir"), "/head_cam_tf_relay.py"], "--scene", lc("scene")],
                           name="head_cam_tf_relay", output="screen")
    freecam = ExecuteProcess(cmd=["python3", [lc("scripts_dir"), "/free_cam_pose.py"], "--scene", lc("scene")],
                             name="free_cam_pose", output="screen")
    foxglove = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(get_package_share_directory("foxglove_bridge"),
                                                "launch", "foxglove_bridge_launch.xml")),
        launch_arguments={"port": lc("foxglove_port")}.items())
    infra = GroupAction([moveit, relay, freecam, foxglove], condition=IfCondition(lc("infra")))

    motion = Node(package="g2cell_motion", executable="motion_server", name="g2cell_motion_server",
                  output="screen", parameters=[lc("params_file"), sim_time])
    tasks = GroupAction([
        Node(package="g2cell_tasks", executable="pick_bag_server", name="g2cell_pick_bag_server",
             output="screen", parameters=[lc("params_file"), sim_time]),
        Node(package="g2cell_tasks", executable="cell_manager", name="g2cell_cell_manager",
             output="screen", parameters=[lc("params_file"), sim_time]),
    ], condition=IfCondition(lc("start_tasks")))
    return LaunchDescription(args + [infra, motion, tasks])
