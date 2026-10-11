# g2cell_slam: SLAM and Nav2 for the mobile G2

> **Stale map (2026-10-11):** `maps/factory_room.*` was built on the Stage 2 layout. Layout v2 (U cell, end-of-line
> machines, start pose (-2, 0) in `scene_g2cell_mobile.yaml`) changed the room; remap (scripted loop, new waypoints
> clear of the sealer/palletiser at X 0.95..4.75, Y -4.05..-0.85) before using navigation.

This package maps the pack-off room with slam_toolbox and then drives the G2 to goals with Nav2 (AMCL on
the saved map). It runs against the mobile scene `config/scenes/scene_g2cell_mobile.yaml`, which uses the
same `cell.usda` as Stage 2. The differences are that the G2 base is free (`pin_base_to_world: false`),
it starts on open floor at (3.4, 0) facing the conveyor, and it has two chassis lidars.

Evidence is in `output/slam/` and in `output/renders/g2_slam_nav.{mp4,png}`.

## Run

All commands run inside the `g2cell` container. Only one stack can run at a time, so stop the cell
scene first with `tmux kill-session -t g2cell`.

```bash
bash /g2cell_ws/scripts/mobile_up.sh          # sim + relay + freecam + foxglove + chassis servo (tmux g2cell)
cd /g2cell_ws && colcon build --packages-select g2cell_slam && source install/setup.bash

# (a) mapping: lidar merge -> /scan -> slam_toolbox; drive:=true runs the scripted loop (98 s, 22.8 m)
ros2 launch g2cell_slam mapping.launch.py drive:=true
ros2 run nav2_map_server map_saver_cli -f src/g2cell_slam/maps/factory_room \
    --ros-args -p map_subscribe_transient_local:=true
python3 /g2cell_ws/scripts/map_preview.py src/g2cell_slam/maps/factory_room.yaml /g2cell_output/slam/factory_room_preview.png

# (b) localization + Nav2 on the saved map (stop the mapping launch first, then rebuild so the map is installed)
ros2 launch g2cell_slam navigation.launch.py
ros2 run g2cell_slam nav_goal --ros-args -p x:=1.65 -p y:=-0.8 -p yaw_deg:=180.0 -p set_initial_pose:=true
```

`nav_goal` takes the AMCL initial pose from ground-truth `/odom`, sends `NavigateToPose`, and prints a
JSON result: success, time, final ground-truth pose, goal error and localisation error. Add `rviz:=true`
to either launch file if you have a display; the config is `rviz/g2cell_slam.rviz`.

To record video, start these before the drive or goal (both start and stop on `/g2cell_demo/state`):

- `ros2 run g2cell_demo record_cams` on `/genie_sim/free_camera_rgb/image_raw/compressed` with
  `wait_for_start:=true`
- `python3 /g2cell_ws/scripts/map_frames.py --out <dir> --wait`

Then run `scripts/make_slam_video.sh`.

## Interfaces (measured)

| What | Topic / frame | Notes |
|---|---|---|
| Base command | `/cmd_twist` (`geometry_msgs/Twist`) | `servo_node` (genie_sim_controllers, 4WS) turns it into `/cmd_4ws` (`JointState`: steer positions + drive velocities) for the engine. There is no `/cmd_vel` in Genie Sim; `cmd_vel_bridge` maps `/cmd_vel` to `/cmd_twist` (clamps to 0.4 m/s and 0.8 rad/s, stops on a 0.5 s timeout). |
| Servo strategy | `default_servo:=GeneralServo` | See "Known limits". The base is used as non-holonomic: `linear.y` did nothing under the default servo, and the bridge drops it. |
| Odometry | `/odom` (~99 Hz), TF `odom -> base_footprint -> base_link` | Published by the engine from the PhysX base pose, so it is **ground truth**: `odom` == world, with no drift. `fake_slam:=false` turns off the engine's identity `map -> odom`, so slam_toolbox or AMCL own that edge. |
| Lidar | `/lidar/livox_front`, `/lidar/livox_back` (~4 Hz each, sensor frames `livox_front` / `livox_back`) | OVRTX OmniLidar render products, about 14k points per scan each (see below). |
| Merged cloud | `/lidar/merged_full` (~8 Hz, `base_link`) | Built by `cloud_merger`, which also publishes the `base_link -> livox_*` static TF. |
| Scan | `/scan` (~8 Hz) | From `pointcloud_to_laserscan`: slice z 0.05..0.60 m in `base_link`, 0.5 deg bins, range 0.55..15 m. |
| Map | `/map`, `maps/factory_room.{pgm,yaml}` | 0.05 m cells, 197 x 198, origin (-4.427, -4.942). |

**Time base: `use_sim_time` is false on purpose.** The engine stamps `/tf`, `/odom` and `/tf_render`
with wall time (`node_->now()` in `realtime_ros_node.cpp`, where upstream notes "use_sim_time=false
project-wide"). The lidar clouds carry the `/tf_render` stamp. `/clock` is sim time (about 140 s after
start), so sim-time nodes would compare wall-time stamps against a 140 s clock.

## How the lidars work

Genie Sim's lidar path (`assemble_scene.py`, `render_node.cpp`, render param `lidar: true`) only references
OmniLidar prims that already exist in the robot USD. Neither the URDF-built G2 nor the official
`G2_omnipicker/robot.usda` has lidar prims; the official USD has cameras only. So the setup adds them:

1. `scripts/inject_lidars.py` writes `/robot/base_link/livox_{front,back}` OmniLidar prims into the staged
   `geniesim_assets/scenes/scene_g2cell_mobile/robot.usda`.
   - That file is cache-gated by `urdf.sha256`, so the edit survives relaunches.
   - On first use, the converted robot is seeded from `scenes/scene_g2cell`.
   - The profile is a generic 16-channel rotary lidar: elevation -7..52 deg (similar to a Livox Mid-360 field
     of view), 0.25 deg azimuth step, 0.1-40 m.
2. `config/launchers/launcher_ovrtx_isaac_physx_lidar.yaml` is upstream's launcher plus `lidar: true`.
3. `scripts/head_cam_tf_relay.py --lidar-body base_link` publishes base_link's world pose for
   `/RenderOVRTX/Lidars/base_link`. Without it the render node never moves the lidar copy, for the same
   frame-name mismatch as the head camera.
4. `cloud_merger` combines the two clouds. The render node's own `/lidar/merged` only contains the lidar(s)
   that finished a scan in that render frame, so it alternates between front-only and back-only clouds.
   Each half-blind scan then had about 180 deg of `inf` beams, and slam_toolbox ray-traced those as free
   space, which erased walls. `cloud_merger` keeps the latest cloud of each lidar, moves each one into
   `odom` with the TF at its own stamp, and publishes the union in `base_link`.

**The mount poses are estimates.** The real G2 publishes `/lidar/livox_front` and `/lidar/livox_back`
(GDK manual 5.1.4), but no mount pose exists in the local assets. The values used are:
- front: (0.40, 0, 0.18), yaw 0
- back: (-0.40, 0, 0.18), yaw 180 deg

Both sit in `base_link`, just outside the chassis hull (X -0.375..0.382). They are set in
`scene_g2cell_mobile.yaml` (`lidars[].mount`) and in `cloud_merger`'s `mounts` parameter; keep the two the same.

Range check, robot at (3.4, 0) facing -X: rear wall 2.0 m (5.4 - 3.4), side walls 4.9 m, front wall
7.8 m, ceiling 3.97 m. All match the room.

## Results (2026-10-11, RTX 3070)

- **Map** (`maps/factory_room.*`, `output/slam/factory_room_preview.png` with the ground-truth layout
  overlaid):
  - Wall coverage within +-0.1 m of the true wall line: W 99 %, N 99 %, S 100 %, E 64 %. The east wall
    falls on the map's last column.
  - Conveyor legs and cross members, pick-table legs, rack legs, pallet, crate stack, waste bin,
    handwash and light-curtain post all fall inside their true outlines.
  - Fixtures show up as legs because the scan slice is a 0.09-0.64 m band above the floor.
- **Nav2 goal 1**, open floor (3.40, -0.07) to in front of the conveyor (1.65, -0.80, 180 deg):
  - Succeeded in 11.9 s.
  - Final ground truth (1.735, -0.748, -179.1 deg): error 0.10 m, 0.9 deg.
  - AMCL localisation error 0.022 m (`output/slam/nav_result_conveyor.json`).
- **Nav2 goal 2**, around the conveyor's south end to the rack side (-2.0, 0.3, 0 deg):
  - Succeeded in 37.7 s.
  - Error 0.084 m, 4.0 deg.
  - Localisation error 0.025 m (`nav_result_rack.json`).
- Goal tolerance is 0.10 m / 0.10 rad.

## Known limits

- **Odometry is ground truth.** SLAM and AMCL never see odometry drift. Add noise to `/odom` (or use
  wheel odometry) before drawing conclusions about real-robot SLAM.
- **Mount poses and the lidar profile are estimates** (see above). The lidar is idealised RTX: no
  motion distortion (`instantLidar`) and no noise.
- **Specular stainless kick plates** (0.10-0.30 m on the walls) return few points at low elevation. The
  4-7 deg channels above them carry the walls.
- **Rendering cost:** with the two lidars on, the render node drops to about 11.6 fps (86 ms per frame) on
  the 8 GB RTX 3070, so each lidar runs at about 4 Hz. Speeds are kept at or below 0.35 m/s.
- **Chassis servo:** the default `OptimalServo` (MPC) is unreliable for spin-in-place.
  - A spin issued right after driving became a 0.6 m arc with only 36 deg of yaw.
  - Another spin stalled at 7 deg in 3 s.
  - `mobile_up.sh` therefore runs `GeneralServo` (4WS inverse kinematics), measured at 75 deg in 3 s with
    3 cm or less of drift.
  - Lateral `vy` did not move the base under `OptimalServo`. `GeneralServo`'s inverse kinematics handles
    `vy`, but that was not tested, and `cmd_vel_bridge` drops `vy`.
- **The stack is mobile or cell, not both:** the mobile scene runs without MoveIt; the cell scene keeps
  the pinned base.
- **Nav2 docking and route servers** start with their default (TurtleBot) parameters. They are unused.
