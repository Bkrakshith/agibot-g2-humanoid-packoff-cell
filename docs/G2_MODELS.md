# AgiBot G2 model files (what exists, where it is)

AgiBot publishes no standalone G2 SDK: the G2 GDK installs only from the robot. The robot **models** are public,
through Genie Sim. All paths are relative to `~/ntu_dev/Agibot_dev/`.

| Format | What | Local path | Source |
|---|---|---|---|
| URDF/xacro (ROS package) | G2 body, chassis, head; arms `crs` / `crsB` (7-DoF); grippers omnipicker / swiftpicker; meshes (Git LFS) | `genie_sim/source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/g2/` | GitHub AgibotTech/genie_sim (also AgibotTech/genie_sim_robot_model) |
| URDF (flat) | G2 omnipicker, fixed base, dual arm; joint index map | `genie_sim/source/data_collection/config/robot_cfg/G2/` | genie_sim |
| USD (Isaac Sim) | **G2_omnipicker**: official USD with physics/sensor payloads and all cameras (head front/left/right, 3 fisheyes, wrist `Left_Camera`/`Right_Camera`). Variants `robot_fix.usda` (fixed base) and `robot_scoop.usda` | `geniesim_assets/robot/G2_omnipicker/` | HF agibot-world/GenieSimAssets |
| USD | **G2_90d**: 90° gripper variant (`robot_gipper_90.usda`, `robot_fixed.usda`, `robot_scoop.usda`) | `geniesim_assets/robot/G2_90d/` | HF |
| USD | **G2_place_workpiece**: task-specific G2 | `geniesim_assets/robot/G2_place_workpiece/` | HF |
| USD + URDF (cuRobo) | G2 config for cuRobo motion planning: `config.yaml`, `G2_omnipicker_fixed_dual.urdf`, USD | `geniesim_assets/robot/curobo_robot/assets/robot/G2/` | HF |
| MJCF (MuJoCo) | `G2_place_workpiece.xml` + parts (arm_crs_pw, body_t2_crs_pw, chassis, hand_pw) and meshes | `geniesim_assets/mujoco/robot/` | HF |
| Real-robot SDK manual | Genie 02 GDK v2.1.0 (ROS 2 Humble, `genie_msgs`, `agibot_gdk`) | `g2_sdk/genie02_gdk_manual_v2.1.0/` | RobotShop CDN |

Which one the cell uses: the Genie Sim RT Engine builds the robot from the **URDF/xacro**
(`g2_crs_omnipicker`) at launch. The official **USD** is the reference for camera mounts; the
head camera pose in our scene YAML is copied from it. Switching the engine to load the official
USD directly (with its built-in cameras, including the wrist cameras) is a candidate improvement,
needed because the spec asks for left/right wrist RGB.
