# g2cell_tasks

Task layer of the pack-off cell: the `pick_bag_server` (PickBag action) and the `cell_manager` skeleton. All
motion goes through `g2cell_motion`'s MoveArm action; this package never publishes `/joint_command`.

## PickBag (`/g2cell/pick_bag`)

```bash
ros2 run g2cell_tasks pick_bag_server --ros-args -p use_sim_time:=true
ros2 action send_goal --feedback /g2cell/pick_bag g2cell_interfaces/action/PickBag "{bag_id: any}"
ros2 action send_goal /g2cell/pick_bag g2cell_interfaces/action/PickBag "{bag_id: Bag_02, target: rack_lane, lane: L1_1}"
```

- **Task loop** (`state_machine.PickPlaceMachine`, pure): approach -> grasp -> lift -> transfer -> place ->
  retract. A missed grasp or a slip during lift re-approaches (up to `max_grasp_attempts`); losing the bag in
  transfer is a `drop`; a motion failure is `plan_failed` and runs `recover` (open, home); cancel is `aborted`
  with no further motion.
- **Hooks**: `grasp_detected`, `holding` (gripper-joint heuristic in `grasp.py`, off by default:
  `grasp_check:=gripper` once validated), `observe_placement` (**TODO Stage 5**: needs a scene-body pose
  relay, since `/tf_render` only carries robot links), `classify` (`outcome.classify`).
- **Outcomes** (`outcome.py`): drop > wrong_target > overflow > regrasp > success. With no placement
  observation the result is `unverified`, never `success`.
- **Arm selection** (`reachability.py`, MoveIt table + Layout v2 offline waist sweep): table picks only with the
  left arm and only for Y <= 0.85 (Bag_03 at Y 1.05 is `unreachable`; the waist does not extend table reach);
  crate: either arm (the one on that side first); rack lanes per `rack_lane_arms` (L1_1 right/left, L2_1 right
  only, L1_2 / L2_2 none), flagged unverified (offline kinematics only).
- **Waist** (`reachability.waist_for`): the approach and the transfer send `waist_mode: waist_first` toward the
  table / rack, `absolute` 0 toward the carton (motion_server turns the transfer's via point part way). Grasp,
  lift, place, retract and back-off hold the waist (`fixed`); `home` turns it back to 0.
  `reachability.waist_first: false` sends `fixed` everywhere.
- **Stages** (`runner.MotionRunner`, pure, `test_runner.py`): the approach plans grasp and lift as MoveArm
  `lookahead`; the transfer plans release and retract as lookahead and falls back to a searched orientation, then
  to `waist_first`; **recover** opens the gripper, backs 8 cm straight up and goes home (arm-first home if refused,
  BUSY retried) and reports failure only if the arm is not home. Root cause and before/after:
  docs/OPTIMISATION.md "PickBag transfer fix".
- **Target selection** (`sorting.py`): label `good` -> pack crate; `reject` -> L1_1, `sample` -> L1_2, any other SKU ->
  L2_2 (`config/tasks.yaml`). Goal `target`/`lane` override the rule.
- **Waypoints** (`waypoints.py`): Stage 2 demo heights (approach +0.13 over the bag top, grasp -0.01, lift +0.17,
  midway via point no lower than the lift, release 0.03 over the rim, retract 0.20).
- Bags: the three Stage 2 placeholders from `config/tasks.yaml` (**TODO Stage 5**: BagInfo from the spawner).

Every result is also published on `/g2cell/pick_results`.

## cell_manager

```bash
ros2 run g2cell_tasks cell_manager --ros-args -p use_sim_time:=true
ros2 service call /g2cell/episode_control g2cell_interfaces/srv/EpisodeControl "{command: 0, episode_id: ep001}"
ros2 topic echo /g2cell/cell_status
```

Publishes `CellStatus` (2 Hz, transient local). Real: episode start/stop/mark (with `record_episodes:=true` it
drives `g2cell_recorder` through g2cell_teleop's `RecorderController`), pick counts from `/g2cell/pick_results`
(only verified placements count into the crate). Not implemented, answering `STATUS_NOT_IMPLEMENTED`:
`/g2cell/set_line_state` (**TODO Stage 3**: conveyor, kinematic stop gate, photo-eye, indexing) and
`/g2cell/spawn_bags` (**TODO Stage 5**). The status's `not_implemented` field says which numbers are assumed.
The TODOs are in `cell_state.py`.

## Tests

`/home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest -q test`: 77 tests (state machine, stage runner,
outcome, reachability, sorting, bags, waypoints, cell state, grasp heuristic, config, and sync with the interface
constants). Offline replay of the whole PickBag against a standalone move_group: `tools/perf/pickbag_replay.py`.
