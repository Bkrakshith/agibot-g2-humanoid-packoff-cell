cmake_minimum_required(VERSION 3.8)
project(g2cell_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BagInfo.msg"
  "msg/CellStatus.msg"
  "msg/PickPlaceResult.msg"
  "srv/SetLineState.srv"
  "srv/SpawnBags.srv"
  "srv/EpisodeControl.srv"
  "action/PickBag.action"
  "action/MoveArm.action"
  DEPENDENCIES builtin_interfaces geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
