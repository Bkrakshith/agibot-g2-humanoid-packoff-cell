# g2cell_interfaces

Messages, services and actions for the pack-off cell use cases (ament_cmake + rosidl; build in the container).

| Interface | Name on the graph | Served / published by |
|---|---|---|
| `msg/CellStatus` | `/g2cell/cell_status` (transient local, 2 Hz) | `g2cell_tasks cell_manager` |
| `msg/BagInfo` | in `SpawnBags` responses | Stage 5 spawner (not yet) |
| `msg/PickPlaceResult` | `/g2cell/pick_results`, `PickBag` result | `g2cell_tasks pick_bag_server` |
| `srv/SetLineState` | `/g2cell/set_line_state` | `cell_manager` (answers `STATUS_NOT_IMPLEMENTED` until Stage 3) |
| `srv/SpawnBags` | `/g2cell/spawn_bags` | `cell_manager` (answers `STATUS_NOT_IMPLEMENTED` until Stage 5) |
| `srv/EpisodeControl` | `/g2cell/episode_control` | `cell_manager` |
| `action/PickBag` | `/g2cell/pick_bag` | `pick_bag_server` |
| `action/MoveArm` | `/g2cell/move_arm` | `g2cell_motion motion_server` |

Outcome codes 0-4 in `PickPlaceResult` are the spec's per-pick outcomes (success, wrong_target, drop, regrasp,
overflow). Codes 10-15 mean the pick never reached a scored placement: `unverified` (motion done, nobody
observed where the bag ended up; the Stage 5 scorer is missing), `plan_failed`, `grasp_failed`, `unreachable`,
`aborted`, `invalid`. `g2cell_tasks/outcome.py` and test `test_outcome_codes_match_msg_constants` keep the
Python side in sync.

`MoveArm` notes: `geometry_msgs/Pose` defaults to w = 1, so set `auto_orientation: true` to let the server search
gripper-down orientations. `frame` is `cell_tool` (fingertip point in the cell frame, default) or `base_link`
(end link). `orientation_used` is all zeros after a joint-space (named) move.
Waist (fields appended 2026-10-11, old goals unchanged): `waist_mode` `""` (server parameter, default
`waist_first`), `waist_first` (turn idx05_body_joint5 so the target is within the torso cone, see
g2cell_motion `waist_policy`), `fixed` (keep the waist), `absolute` (turn to `waist_yaw`, rad). The result reports
the final `waist_yaw` and, when the waist could not cover the bearing, `base_turn_needed` / `base_turn` (rad; the
base is never driven).

```bash
colcon build --packages-select g2cell_interfaces && source install/setup.bash
ros2 interface show g2cell_interfaces/action/PickBag
```
