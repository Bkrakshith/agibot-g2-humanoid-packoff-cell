# g2cell_bringup

Launch files, parameters and a health check for the use-case stack.

| File | What |
|---|---|
| `launch/cell.launch.py` | `motion_server`, `pick_bag_server`, `cell_manager`; with `infra:=true` also MoveIt move_group (IK/FK only, `use_ros2_control:=false`), head-cam TF relay, FreeCam aimer, Foxglove |
| `launch/teleop_record.launch.py` | MoveIt (`start_moveit:=false` if stage1_up.sh runs it) + printed keyboard-teleop/recording instructions; `start_teleop:=true terminal_prefix:='xterm -e'` to include the teleop launch |
| `config/cell_params.yaml` | parameters of the three nodes |
| `check_stack` | lists required topics/services/actions and what is missing (exit 1 if anything is) |
| `g2cell_engine_isaacsim.py` + `object_relay.py` | the Genie Sim Isaac engine with the object pose relay hook (below); `stage1_up.sh` uses it when `OBJ_RELAY=1` |

**The sim is not started by these launch files.** It stays with `scripts/stage1_up.sh`: only one operator
may run or restart it, it needs minutes of shader compilation and must be rendering before the relay and MoveIt
are useful (stage1_up.sh waits for "Render Stats"), and it should outlive restarts of the use-case nodes.
stage1_up.sh already starts the relay, FreeCam, MoveIt and Foxglove, hence `infra:=false` by default.

```bash
bash /g2cell_ws/scripts/stage1_up.sh /g2cell_ws/config/scenes/scene_g2cell.yaml   # sim + infra
ros2 launch g2cell_bringup cell.launch.py                                           # use-case nodes
ros2 run g2cell_bringup check_stack
```

`motion_server` refuses to start while the keyboard teleop or `pick_place_demo` publishes `/joint_command`;
teleop sessions and the cell stack are mutually exclusive.

Tests: `../.venv-tools/bin/python -m pytest -q test` (34: object relay logic 15, grasp assist 12
incl. the USD adapters on an in-memory stage, check_stack logic, launch files compile). The launch files were only parsed (`ros2 launch ... --show-args`), not run against the sim.

## Object pose relay (`object_relay.py`, `engine_objrelay.py`)

Genie Sim's engine publishes `/tf_render` only for the robot's rigid bodies (`kit/stage.py` `_collect_body_paths`
walks `/<robot_prefix>`), so the OVRTX render node never moved bags, cartons or totes. The render stage opens the same
`cell.usda` as the physics stage, and the render node applies a `child_frame_id` that starts with `/` as that prim's
local xform (`render_node.cpp` `on_tf_render`). Object poses only exist inside the engine process, and no upstream
option exports them, so the relay runs inside the engine:

- `g2cell_engine_isaacsim.py` loads the installed upstream `genie_sim_engine_isaacsim.py` as a module (same
  SimulationApp bootstrap). It wraps `EngineSession.startup` to append one `post_step_hook`, the engine's own
  extension point, and then calls upstream `main()`. Upstream files are untouched.
- The hook selects the top-level, non-kinematic, enabled rigid bodies outside `/<robot_prefix>` (14 in the cell
  scene). It reads their USD local xform, which PhysX writes back after `fetch_results`, once every
  `ceil(physics_hz / G2CELL_OBJ_RELAY_HZ)` ticks (default 30 Hz, so every 4th tick, 25 Hz). It publishes only the
  bodies that moved more than 0.5 mm or 0.2° since their last published pose, and all of them every
  `G2CELL_OBJ_RELAY_KEYFRAME_S` (2 s). The keyframe repairs dropped best-effort messages and a late render start.
- Publishing happens on a separate rclpy context and thread (latest wins), so the physics tick never waits on
  DDS. Messages are stamped with wall time and frame `map`, like the engine's own `/tf_render`.
- `/g2cell/object_poke` (std_msgs/String JSON `{"path", "xyz"|"dxyz", "vel"}`, relayed bodies only) is applied on
  the physics thread between steps: a teleport (translate op) and/or a linear velocity. CLI:
  `scripts/object_poke.py`.
- Every 10 s the engine log shows `[obj_relay] 10 s: samples=… msgs=… tfs=… cost/sample …`. Its share of the
  per-tick hook time also appears in the engine's `Extras` stats line.
- Switches: `OBJ_RELAY=0` runs the upstream engine. `G2CELL_OBJ_RELAY_KINEMATIC=1` also relays kinematic bodies.
  `G2CELL_OBJ_RELAY_PUBLISH=0` keeps pokes but sends nothing (A/B checks only).
- Limits: poses carry no scale, so a relayed body must have unit scale (true for all cell objects). Nested
  rigid bodies are moved through their top-level body.

Measured on the workstation, pinned cell scene, `PROFILE=collect`, 30 s windows
(`output/readiness/relay/perf_*.json`):

| | relay off | relay on (idle) |
|---|---|---|
| engine CPU | 331 % | 331 % |
| render fps / frame time | 30.0 / 22.7 ms | 30.0 / 22.8 ms |
| physics step mean / p99, overruns | 4.45 / 6.0 ms, 0 | 4.45 / 5.8 ms, 0 |
| `Extras` per tick (hook) | 0.00 ms | 0.05 ms (max 0.23) |
| relay sample | – | 0.17 ms per sample, 25 Hz, about 0.4 % of one core |

While objects were being pushed continuously, the relay sent 9 messages per second carrying 15 poses per second.
The sample cost stayed at 0.18 ms. Physics slowed to 96 Hz with 5 overruns per 500 steps. The contacts caused this
(solver 4.8 ms vs 4.45 ms), together with the load from about 60 `object_poke.py` processes; the relay did not.

## Sticky grasp (`GraspAssist`, 2026-10-11)

The omnipicker cannot physically hold the 0.21 × 0.30 × 0.07 m rigid placeholder bag (PickBag grips 1 cm below its
top, and the fingers resting on the box cannot even close until the lift starts), so a grasp never lifted the bag.
The relay hook therefore also runs `GraspAssistHook` (`GRASP_ASSIST=1` on stage1_up.sh, default; `G2CELL_GRASP_ASSIST`
inside the engine):

- **attach** when a gripper's COMMANDED leader joint (`/joint_command`, any publisher; measured value as fallback)
  goes from open to closing (> −0.30 rad) while its fingertip (gripper_*_base_link + 0.12 m along +Z) lies in a
  `/World/Bags/*` footprint (+4 cm) between the bag bottom and top + 4 cm. The bag becomes kinematic with its
  colliders off and follows the gripper link every physics tick with the grasp-time offset;
- **detach** when the command opens past −0.55 rad: dynamic again, colliders on, zero velocity, it drops;
- a close with no bag in reach logs `[grasp_assist] ... no bag in reach: fingertip ..., nearest ...`.
- `/g2cell/object_poke` also takes `yaw_deg` and `stop` (zero linear + angular velocity): g2cell_datagen's
  `teleport` reset uses it.

Live: 20/20 pilot episodes attached on the grasp and released over the carton (`output/readiness/pilot/`); physics
99.7 Hz, 0.44 overruns/500 during datagen (`perf_during_datagen.json`).
