# g2cell_gdk_sim: `agibot_gdk` 3.3.8 backed by Genie Sim

GDK Python code written for the real AgiBot G2 (`import agibot_gdk`) runs unchanged against our
Genie Sim cell. This package installs a Python module named `agibot_gdk` (`__version__ =
"3.3.8-sim"`). It has the same public names, signatures, enums and return shapes as the 3.3.8 calls
used by [agibot-gdk-training](https://github.com/SailnMobula/agibot-gdk-training) (Apache-2.0,
commit `29e4dc3`). Every call is served by a ROS 2 backend talking to the sim. See [NOTICE](NOTICE)
for attribution.

## Use

Inside the `g2cell` container, with the sim and `move_group` running (`stage1_up.sh` with
`scene_g2cell.yaml`) and **no other `/joint_command` publisher** (quit `keyboard_teleop` and
`motion_server` first):

```bash
cd /g2cell_ws && colcon build --packages-select g2cell_gdk_sim
source /g2cell_ws/src/g2cell_gdk_sim/sim.sh          # agibot_gdk first on PYTHONPATH (like fake.sh)
python3 my_gdk_script.py
```

To run the same script on the real robot, source the vendor GDK environment instead of `sim.sh`.
That is agibot-gdk-training's `real.sh <ROBOT_IP>` on a laptop, or `robot.sh`
(`/home/agi/app/env.sh`) on the robot. Do not source both in one shell. `python3 -c "import
agibot_gdk; print(agibot_gdk.__version__)"` prints `3.3.8-sim` for the sim. The real binding
prints its own version.

### Training solutions as the acceptance test

```bash
# the training repo (commit 29e4dc3) is copied to ros_ws/third_party/agibot-gdk-training
# (third_party has a COLCON_IGNORE). To refresh it, on the host:
#   cp -r ../g2_sdk/community/agibot-gdk-training ros_ws/third_party/
# in the container, sim + move_group up, teleop quit:
bash /g2cell_ws/src/g2cell_gdk_sim/run_training_solutions.sh            # all exercises
bash /g2cell_ws/src/g2cell_gdk_sim/run_training_solutions.sh -- 04 07   # a subset
```

Each solution runs with `SHOW_ONCE=1` and a timeout. The script checks the expected output (the
same strings the training repo's tests check) and prints a summary table. Logs go to
`/g2cell_output/gdk_acceptance/<time>/`. Exercises 04 (head), 06 (left wrist sine) and 07 (left
end effector +5 cm, then -5 cm) move the robot. 04 returns to its start pose.

| Exercise | Sim status |
|---|---|
| 01 joints, 02 status, 03 frames/forces, 05 gripper, 08 TF, 10 camera+TF | run |
| 04 joint moves | runs a scripted prompt in a temp copy (poses.json stays out of the repo) |
| 06 servo, 07 EE pose (up, then down) | run |
| 09 camera | **expected skip**: it opens both wrist cameras, which the sim does not publish. `09_camera_head_only` runs the solution's `info()`/`grab()` on head colour and depth instead (`scripts/head_camera_check.py`) |
| `gdk_training.doctor` | **expected skip**: it checks the robot network and a compiled vendor binding |

`RUN_SKIPPED=1` runs the skipped ones anyway.

### Offline check (no sim)

```bash
ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST python3 scripts/offline_smoke.py
```

This starts a fake sim on an isolated domain and checks the whole ROS path: `/joint_states`,
`/joint_command` streaming, the guard, IK, FK, TF and the camera topics. It refuses to run on
domain 0. With `--serve` it only runs the fake sim, so `run_training_solutions.sh` can be run
against it on the same domain.

Host unit tests (no ROS): `../.venv-tools/bin/python -m pytest -q test`.

## How it works

`gdk_init()` creates one rclpy node (`agibot_gdk_sim_<pid>`, `use_sim_time` false) spun by a
`MultiThreadedExecutor` in a daemon thread. Importing `agibot_gdk` does not import rclpy.
`gdk_release()` destroys the node.

| GDK call | Sim implementation |
|---|---|
| `get_joint_states` | `/joint_states` (best effort). The 22 body joints in GDK order. `motor_position`/`motor_velocity` from the sim, and `position`/`velocity` mirror them |
| `joint_control_request`, `move_head/waist/arm_joint` | Synchronised quintic interpolation, streamed at 50 Hz on `/joint_command`. Duration = max over joints of 1.875·Δq/v, so no joint exceeds its requested speed. v defaults to 0.3 rad/s and is capped at the URDF velocity limit. Returns 0 once every joint is within 0.02 rad, or `kRuntimeError` (3) after a 2 s settle timeout or when a newer command preempts its joints |
| `joint_servo_control` | Sets the targets directly; the 50 Hz stream carries them. `control_period` is ignored |
| `end_effector_pose_control` | MoveIt `/compute_ik` (`simple_arm_l`/`simple_arm_r`, `ik_link_name` `arm_l_end_link`/`arm_r_end_link`, pose in `base_link`, seeded with the current command). The solution becomes the arm targets |
| `move_ee_pos` | OmniPicker leader joints `idx31_gripper_l_inner_joint1` and `idx71_gripper_r_inner_joint1` |
| `get_end_state` | The leader joints from `/joint_states` |
| `TF.*` | tf2 Buffer and Listener on `/tf` and `/tf_static`, plus two virtual frames |
| `Camera.*` | `/genie_sim/head_front_camera_rgb/image_raw` (rgba8) and `_depth/image_raw` (32FC1), subscribed when a `Camera` is created |

### Frame and gripper mappings

- **End-effector pose.** Training exercise 7 says the GDK pose is the pose of `arm_l_end_link` /
  `arm_r_end_link` in `base_link`. The gripper centre is a separate frame further along the tool
  axis. Our MoveIt IK link is the same `arm_*_end_link`, in the same `base_link`. So the mapping is
  the identity, with **no** tool offset. `get_motion_control_status().frame_poses` reports those
  frames from TF, which closes the loop that exercise 7 measures.
- **Gripper centre.** The sim's TF tree has no `gripper_*_center_link`. The package adds
  `gripper_l_center_link` / `gripper_r_center_link` as virtual frames at `TOOL_OFFSET` = 0.125 m
  along the end link's +Z. That is `g2cell_motion`'s fingertip offset, used by exercise 8's tool
  lookup. The real robot's offset may differ.
- **`lookup_transform_latest(target, source)`** returns the pose of `target` in `source`: the
  transform that maps `target` points into `source`. That is the direction exercises 8 and 10
  rely on (tf2's `lookup_transform(source, target)`). `get_tf_from_base_link(f)` is the pose of `f`
  in `base_link`.
- **Head camera.** `get_tf_from_sensor(kHeadRGBDToHeadLink3)` is the official mount from the scene
  YAML, converted from the USD camera convention to the optical frame (z forward, x right,
  y down). `kHeadDepthToHeadColor` is the identity, because one camera prim renders both.
  `GDK_SIM_SCENE` selects the YAML (default `/g2cell_ws/config/scenes/scene_g2cell.yaml`).
- **Gripper values.** Training exercise 5 states that the GDK OmniPicker value is in rad, with
  -0.785 open and 0 closed. The sim's leader joint uses the same values, so the mapping is the
  identity, clamped to [-0.785, 0]. `get_end_state` reports the measured leader joint.

### Joint limits

A target outside the **real** soft limits (`t2_soft_limit.json`, see `_limits.py`) raises
`RuntimeError("kInvalidInput: <joint> <value> outside <lo> .. <hi>")`, as the robot rejects it. A
target inside the real limits but outside the narrower **sim URDF** limits is clamped into the
intersection, and the clamp is logged. The sim is narrower for head yaw (±0.781 vs ±1.571), head
pitch (±0.151 vs ±0.349), head roll (-0.018 … 0.523 vs ±0.535) and waist joint 3 (upper 1.039 vs
1.571). The arms are identical. The sim's physics lets a joint rest a few mrad past its limit
(the waist sits about 5 mrad above 0.0), so a saved pose can lie slightly outside the real
limit. Targets up to 0.01 rad outside, or as far outside as the joint already measures, are
clamped rather than rejected.

## Sim vs real: every difference

| Topic | Real G2 (3.3.8) | This package |
|---|---|---|
| Transport | AORTA/DDS to the robot, vendor binding | ROS 2 topics and services of Genie Sim; `gdk_init` only creates a local node |
| `get_app_version()` | GDK version | `3.3.8-sim (Genie Sim backend, ROS_DOMAIN_ID=…)` |
| Joint `position`/`velocity` | reserved, no usable data | mirror `motor_position`/`motor_velocity` |
| `motor_current` | measured | NaN (not simulated) |
| `mode` of a joint | drive mode | 0 |
| Joint effort | measured | the sim's `/joint_states` effort (NaN if empty) |
| Whole-body status | measured errors, e-stops | **synthetic**: all errors 0, e-stops false, arm control true, end models `omnipicker` |
| Motion-control status | controller state | `mode` 1, the persisted `control_mode`, `error_code` 0. Frame poses from TF; **twists, wrenches and `arm_disturbance_force` are NaN** (no force sensing), with the same list lengths |
| `get_chassis_power_state` | battery data | **synthetic**: two batteries at 100 % SOC; voltage, current and temperature NaN |
| `GDK_SIM_STRICT=1` | – | the synthetic status calls raise `RuntimeError("kRuntimeError …")` instead |
| `set_control_mode` | switches the controller | stored (persisted in `~/.cache/agibot_gdk_sim/control_mode`, as the robot keeps it across processes). Mode 2 blocks the joint calls, as on the robot. Impedance behaviour is not simulated: every mode is position control |
| `clear_motion_control_error` | clears errors | no-op, returns 0 |
| Blocking moves | robot planner, returns on arrival | quintic interpolation (see above). Joints start and arrive together. `life_time` ignored. No balance/centre-of-gravity check for the waist |
| Settle failure | – | returns 3 (`kRuntimeError`) after 2 s |
| Servo | collision protection unless `enable_low_latency` | head-clearance check unless `enable_low_latency`. Setpoints more than 0.5 rad from the current command are rejected (`kInvalidInput`) |
| Collision protection | robot self-collision model | the sim G2 has no arm colliders. Arm motions are FK-checked (`/compute_fk`) to keep arm and gripper links 0.20 m from `head_link3` and the head camera. A violating request raises `kRuntimeError`, unless it moves away from an already-close pose. `GDK_SIM_HEAD_CLEARANCE=0` disables the check |
| EE pose stream | robot IK, waist groups supported | MoveIt IK per call (about 5–20 ms). Arm groups only. The waist groups raise `kInvalidInput`. A pose more than 5 cm from the last one (or from TF) raises `kInvalidInput`. An IK failure, or a branch flip of more than 0.5 rad, returns 3 |
| `move_ee_pos` after an EE pose stream | reported to be dropped for a few seconds (gotcha in G2_GDK_NOTES) | executed immediately (the quirk is not emulated) |
| Gripper | force-limited OmniPicker; current, temperature and voltage measured | sim leader-joint position drive; current, temperature and voltage NaN |
| Cameras | head colour (JPEG per the training fake), head depth Z16, stereo, fisheye, wrist RGB-D | head colour as **UNCOMPRESSED BGR** 640x400 (from rgba8), head depth as UNCOMPRESSED `RS2_FORMAT_Z16` uint16 mm (0 = invalid). **No wrist, stereo or fisheye cameras**: `get_latest_image` returns None after `timeout_ms`; shape, intrinsic and fps raise `kRuntimeError`. Add topics with `GDK_SIM_CAMERA_TOPICS="kHandLeftColor=/topic;…"` once the sim publishes them |
| Camera fps | camera rate | measured arrival rate, else the scene's nominal 10 Hz |
| Sensor extrinsics | calibrated | head RGB-D from the scene YAML; depth→colour identity; all others raise `kRuntimeError` |
| TF frames | robot tree | sim tree (`base_link`, `head_link3`, `arm_*_end_link`, `gripper_*_base_link`, …) plus the virtual `gripper_*_center_link` |
| Time | PTP-synced robot clock | wall time (the engine stamps wall time; `use_sim_time` false) |
| Other GDK classes | Imu, Lidar, Slam, Pnc, Map, Interaction, … | not provided (not used by the training) |
| `/joint_command` ownership | – | single-publisher guard: the first motion command raises `kRuntimeError` if another node publishes `/joint_command`. If one appears later, streaming stops and further commands raise |

Enum values: names and the fake's values are kept. `EndEffectorControlGroup` kLeftArm/kRightArm/
kBothArms = 1/2/3 as in the 3.3.8 training fake. The v2.3.4 reference lists 4/8/12, so read
members by name.

## Configuration (environment)

`GDK_SIM_SCENE`, `GDK_SIM_RATE_HZ` (50), `GDK_SIM_SETTLE_TOL` (0.02 rad), `GDK_SIM_SETTLE_TIMEOUT`
(2 s), `GDK_SIM_DEFAULT_VEL` (0.3 rad/s), `GDK_SIM_HEAD_CLEARANCE` (0.20 m, 0 = off),
`GDK_SIM_MAX_SERVO_STEP` (0.5 rad), `GDK_SIM_MAX_EE_STEP` (0.05 m), `GDK_SIM_IK_TIMEOUT` (0.05 s),
`GDK_SIM_GUARD_WAIT` (1.5 s discovery before the guard check), `GDK_SIM_JOINT_WAIT` (2 s),
`GDK_SIM_STATE_DIR`, `GDK_SIM_STRICT`, `GDK_SIM_CAMERA_TOPICS`.

## Layout

| Path | Content |
|---|---|
| `agibot_gdk/__init__.py` | public API: `gdk_init`, `Robot`, `TF`, `Camera`, structs, enums |
| `agibot_gdk/_types.py` | enums and structs |
| `agibot_gdk/_controller.py` | all behaviour, ROS-free, driven through a port |
| `agibot_gdk/_sim_backend.py` | the ROS 2 port (node, executor thread, guard, IK/FK, tf2, images) |
| `agibot_gdk/_limits.py`, `_motion.py`, `_convert.py`, `_geometry.py` | limits, quintic, conversions, poses |
| `test/` | host pytest with a fake port; `fake_surface_3.3.8.json` is the API snapshot |
| `sim.sh`, `run_training_solutions.sh`, `scripts/` | environment, live acceptance, offline smoke, head-camera check |
