# g2cell_motion

Reusable G2 arm motion for the pack-off cell, and the `motion_server` node: the **only** `/joint_command`
publisher while use-case code runs. Factored out of `g2cell_demo` (which now imports it).

## What it does

- **Waypoint planning** (`planner.plan_chain`): MoveIt `/compute_ik` for a chain of fingertip points with one
  shared gripper-down orientation, each waypoint seeded with the previous solution. Orientation search order:
  `prefer`, then 8 yaws x tilts 0/20/40 deg (the Stage 2 demo order), or one fixed quaternion.
- **Head clearance** (`clearance.path_clearance`): the G2's arm colliders are disabled in Genie Sim, so every
  arm/gripper link must stay 0.20 m from `head_link3` and the head camera along the sampled min-jerk joint
  path (MoveIt `/compute_fk`). Paths below that are rejected.
- **Quintic joint interpolation** (`geometry.min_jerk`, `streamer.run_segment`) streamed at 50 Hz, best effort,
  on `/joint_command`. Segment time keeps the peak joint speed under `max_joint_vel`; a goal `duration`
  only stretches it.
- **Gripper**: omnipicker leader joints, 0 = closed, -0.785 = open (`robot.gripper_value`).
- **Named poses** from `config/named_poses.yaml` (`home`, `above_crate`, `above_table` (left only),
  `pre_crate_left`): joints in degrees, or a fingertip point plus optional yaw/tilt.
- **Waist-first turning** (`waist_policy`, default ON, parameter `waist_mode`, goal field `waist_mode`): to face a
  target the torso turns at the waist (`idx05_body_joint5`) so the target's bearing from the waist axis (pivot
  (0.258, 0) in base_link) is within +-30 deg of the torso's forward direction, with the smallest turn, clamped to
  +-100 deg (joint limit +-174.5). What the waist cannot cover is reported as `base_turn` / `base_turn_needed`
  in the result (the base is never driven). IK (7-DoF, MoveIt) is solved with the waist set to the turned yaw,
  seeded from the previous solution and from the forward home posture; solutions with arm joint1 > 90 deg from
  home or a joint 3/5/7 jump > 1.5 rad from the seed ("backward twist") are dropped, the rest ranked by distance
  to home. Fallback waist yaws: 30 deg steps back to the current yaw (reach sweep: a smaller turn often reaches
  further; steps of `waist_fallback_step_deg`, 15, each trying `waist_fallback_orients` orientations, 0 = all), one
  step past, no turn. `absolute` with via points turns the via points evenly on the way to `waist_yaw`.
  Goal `lookahead` points (the next moves, e.g. grasp/lift under an approach) are planned with the same orientation
  at the final waist yaw and head-checked but not executed (`lookahead_ik_failed`). Waist and arm move on ONE quintic segment (synchronised). Head clearance is
  re-checked with the head re-evaluated per sample (it turns with the waist). Modes: `waist_first`, `fixed`,
  `absolute` (`waist_yaw`). Named pose `home` has `waist_deg: 0`. A goal that turns the waist is refused (BUSY)
  while the other arm is moving. Reach: `output/reach/g2_reach_envelope_waist.png`.
- **Single-publisher guard**: refuses to start (exit 2) when another node publishes `/joint_command`, and stops
  publishing and aborts goals if one appears later. Nothing is published until the first goal.

Pure modules (no ROS, unit-tested): `geometry robot named_poses clearance planner guard streamer waist_policy`.
ROS modules: `ik_client` (IK/FK mixin), `motion_server`, `client` (blocking `MotionClient`, cancel-aware).

## Use

```bash
# move_group (IK only) must run (stage1_up.sh starts it); quit keyboard_teleop / pick_place_demo first
ros2 run g2cell_motion motion_server --ros-args -p use_sim_time:=true
ros2 action send_goal /g2cell/move_arm g2cell_interfaces/action/MoveArm "{arm: left, named_pose: home}"
ros2 action send_goal --feedback /g2cell/move_arm g2cell_interfaces/action/MoveArm \
  "{arm: left, pose: {position: {x: 0.10, y: 0.56, z: 1.07}}, auto_orientation: true}"
ros2 action send_goal /g2cell/move_arm g2cell_interfaces/action/MoveArm "{arm: left, gripper_only: true, gripper: 1}"
```

From Python (node spun by a `MultiThreadedExecutor` in another thread):

```python
from g2cell_motion.client import MotionClient, ok, quat_of
mc = MotionClient(node)
r = mc.move_tool("left", (0.10, 0.56, 1.07))                 # orientation searched
mc.move_tool("left", (0.10, 0.56, 0.93), quat=quat_of(r))   # same wrist orientation
mc.gripper("left", close=True)
mc.move_named("left", "home")
```

Parameters: `g2cell_bringup/config/cell_params.yaml` (`rate_hz`, `max_joint_vel`, `min_segment_time`,
`gripper_time`, `head_clearance`, `ik_attempts`, `named_poses_file`, `action_name`, `waist_mode`, `waist_cone_deg`,
`waist_max_turn_deg`, `waist_pivot`, `twist_j1_max_deg`, `twist_jump_max`, `waist_fallback_step_deg`,
`waist_fallback_orients`, `local_ik`, `ik_confirm`, `ik_fallback_attempts` (default 0: deterministic), `plan_cache`).

## Tests

`../.venv-tools/bin/python -m pytest -q test` (from this folder): 108 tests.
`ros_ws/scripts/offline_smoke.py` runs motion_server + tasks against a fake robot/move_group on an isolated
ROS domain (see the top-level README). **Not yet run against the sim.**
