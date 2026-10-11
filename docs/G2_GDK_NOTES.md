# G2 real-robot interface (GDK v2.6.3): notes for sim-to-real

Source: the community repo https://github.com/TYTianYuan/agibot-g2-dev (commit `796ae9d`,
2026-09-28; no licence; cloned read-only to `../g2_sdk/community/agibot-g2-dev`). Its README
transcribes AgiBot's GDK v2.6.3 docs (official portal: https://support.agibot.com/?gdk_version=2.6.3).
These notes are a summary in our own words, kept for reference. Copy no code from that repo; it has
no licence. The GDK itself (`genie_msgs`, `libgdk_adapter.so`, `agibot_gdk`) still comes only from
the robot: `curl -sSL http://10.42.1.101:8849/install.sh | bash`, with the dev PC on 10.42.1.102.

## Why it matters for this project

The real G2 uses the **same joint names as our sim** (`idx01_body_joint1` … `idx05`, head
`idx11/12/13`, arms `idx21–27` left and `idx61–67` right, gripper leaders). Code written against
these names can move to the robot with only a topic/message translation, with no renaming.

## Interface map: sim (Genie Sim, our packages) → real G2 (GDK)

| Function | Sim today | Real G2 (GDK) |
|---|---|---|
| Joint state | `/joint_states` (sensor_msgs/JointState, best-effort, ~100 Hz) | `/hal/joint_state` (`genie_msgs/JointState`: name, mode, position, velocity, effort, **motor_position**, motor_velocity, motor_current, error_code). The docs say to read `motor_position`; `position` is reserved. |
| Joint position command | `/joint_command` (sensor_msgs/JointState, best-effort, 50 Hz stream) | `/MotionControlService/JointPosition/request` (`genie_msgs/JointPositionRequst` [sic]: header, lifetime, joint_names, joint_positions, joint_velocities, uuid, details), response on `/MotionControlService/JointPosition/response` (data, uuid, detail). Every command needs a fresh uuid. |
| Arm/waist/head servo (C++) | — | `MoveArmJointServo`, `MoveWaistJointServo`, `MoveHeadJointServo`, `JointServoControl` |
| End-effector pose | MoveIt `/compute_ik` + our `g2cell_motion` | `EndEffectorPoseControl` (Cartesian), `MoveEEPos` (gripper, omnipicker) |
| Base motion | `/cmd_twist` → servo_node → `/cmd_4ws` (`g2cell_slam`) | PNC: `MoveChassis`, navigation tasks with `GetTaskState` / `CancelTask` |
| Head RGB / depth | `/genie_sim/head_front_camera_{rgb,depth}` | `/camera/head_color`, `/camera/head_depth`, plus `head_stereo_{left,right}` and 3 head fisheyes |
| **Wrist cameras** | not published yet | `/camera/hand_{left,right}_{color,depth}` (RGB-D on each hand). The spec's wrist views are needed in the sim. |
| LiDAR / IMU | `livox_front` / `livox_back` (our estimated mounts) | Livox front/back: `/imu/livox_front`, `/imu/livox_back`, `/imu/chassis`, plus the lidar topics named in the GDK docs |
| TF | sim TF tree | `GetTfFromBaseLink`, `GetTfFromSensor`, `GetAllFrameNames` (C++) |
| Time | engine stamps wall time | PTP time sync between robot and dev PC (scripts in the GDK) |

ROS 2 topics are **off by default** on the robot: you start GDK forwarding nodes (e.g.
`source ~/.cache/agibot/app/gdk/scripts/ros_env.sh gdk_controller && ros2 launch gdk_controller controller.launch.py`).
The GDK docs target **ROS 2 Humble / Ubuntu 22.04 x86_64**, while our sim container is **Jazzy**,
so the bridge has to cross distros (DDS between the two works for simple types; `genie_msgs` must be built in both).

## Gotchas reported in the repo (unverified by us)

- In one GDK session, sending `EndEffectorPoseControl` (Cartesian) then `MoveEEPos` returns success
  but the omnipicker does not move. The author grasps with the planner only, with no Cartesian move
  before closing the gripper.
- The docs show a `uuid` member on the C++ `JointControlReq`, but the installed header doesn't have it.
  Trust the installed headers over the docs.
- Read joint feedback from `motor_position`, not `position`.

## Suggested next step: `g2cell_gdk_bridge`

A small package that makes the sim speak the real robot's ROS 2 interface:
- a local `genie_msgs` rebuilt from the documented fields, to be swapped for the robot's own
  package when available;
- a sim-side node publishing `/hal/joint_state` and accepting `/MotionControlService/JointPosition/request`
  (forwarded to `/joint_command`, answered on `.../response`), and republishing camera topics
  under the real names.

Use-case code that targets the GDK names then runs unchanged in the sim and on the robot.

## Second source: GDK v2.3.4 community API reference (MIT)

https://github.com/SailnMobula/agibot-gdk-skil, `skills/genie-gdk/SKILL.md` (commit `7ec86ca`,
MIT licence), cloned to `../g2_sdk/community/agibot-gdk-skil`. It is a compact API reference for
Python (`agibot_gdk.Robot/Camera/Imu/Lidar/Slam/Pnc/Map/TF/...`), C++ and ROS 2 forwarding, with
enums, structs, motion-planning requests, return codes and DDS topic names. It covers
v2.3.4; the other repo covers v2.6.3. Check against the GDK version installed on the robot.

What it adds:
- **LiDAR topics** `/lidar/livox_front`, `/lidar/livox_back` (+ IMUs `/imu/livox_front|back`,
  `/imu/chassis`). These match the names our SLAM scene already uses for its estimated lidars.
- **Hand cameras**: in v2.3.4, each hand has upper and lower colour + depth cameras
  (`/camera/hand_{left,right}_{upper,lower}_{color,depth}`). In v2.6.3 the docs show `hand_{left,right}_{color,depth}`.
- **SLAM/PNC on the robot**: `Slam.start_mapping/stop_mapping/get_curr_pose`,
  `Pnc.normal_navi/high_precision_navi/relative_move/move_chassis`, with task states
  0 idle … 8 fail, 9 success. Our `g2cell_slam` (slam_toolbox + Nav2) is the sim stand-in for these.
- **Real soft joint limits** (`gdk/config/t2_soft_limit.json`) compared with our sim URDF (2026-10-11):
  all 14 **arm joints are identical**; waist `body_joint3` and all 3 head joints are **narrower in
  the sim** (e.g. head yaw ±0.78 vs ±1.57 rad). The sim is never wider than the real limits, so
  joint targets that are valid in the sim are within the real robot's soft limits.

## Third source: GDK 3.3.8 training (Apache-2.0)

https://github.com/SailnMobula/agibot-gdk-training (commit `29e4dc3`, 2026-10-07, Apache-2.0), cloned
to `../g2_sdk/community/agibot-gdk-training`. It has 10 exercises plus reference solutions for the
**Python `agibot_gdk` 3.3.8** API, tested by the author on one G2 with an Ubuntu 24.04 laptop on
2026-10-07: joints, status, frames/forces, joint moves, gripper, servo, EE pose, TF, camera,
camera+TF. This is the newest GDK version we've seen (vs 2.6.3 and 2.3.4 above).

Facts it adds (author's observations, one unit):
- **Python 3.12** on the robot and in the laptop venv. The robot's laptop package ships a CPython
  3.10 binding only, so SETUP.md explains building a 3.12 wheel. Our sim container is also Python 3.12.
- Network: AORTA discovery (etcd) on robot port **2379**, data over TCP/UDP, install/docs HTTP on
  **8849** (`http://<ROBOT_IP>:8849` serves the GDK reference). Env: `AORTA_DISCOVERY_URI`,
  `LOCATOR_IP`, `APP_CONF_PATH`. Commands need a firewall rule (see its SETUP.md). Wi-Fi works as well as the Debug cable.
- API calls used: `Robot.get_joint_states / get_whole_body_status / get_motion_control_status /
  get_end_state / set_control_mode / joint_control_request / move_{head,waist,arm}_joint /
  joint_servo_control / move_ee_pos / end_effector_pose_control`, `TF.get_all_frame_names /
  lookup_transform_latest / get_tf_from_base_link / get_tf_from_sensor`, `Camera.get_latest_image /
  get_camera_intrinsic`.
- `fake/agibot_gdk/__init__.py` is a stand-in with the real names and return shapes but toy
  behaviour (linear joint ramps, simplified kinematics, synthetic camera image).

## Recommended: a sim-backed `agibot_gdk`

Replace their toy stand-in with one backed by **Genie Sim**:
- joint states from `/joint_states`;
- joint/servo/arm moves streamed on `/joint_command`;
- `move_ee_pos` driving the omnipicker leader joints;
- `end_effector_pose_control` through MoveIt `/compute_ik`;
- TF from tf2;
- cameras from the sim image topics.

The training's reference solutions then become an acceptance test, and the same GDK Python code
runs on the sim and on the G2. This supersedes the topic-level `g2cell_gdk_bridge` idea for Python
users; keep the topic bridge only if C++/ROS 2 GDK code needs it too. Apache-2.0 allows adapting
their stand-in's structure with attribution.
