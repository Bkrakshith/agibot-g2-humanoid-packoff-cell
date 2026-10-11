# G2 salad pack-off cell: build plan

Spec: "Isaac Sim Build Spec – G2 Salad Pack-Off Cell" (separate document, not in this
repository). This file records how
the build departs from the spec and the state of each stage. The spec stays the source of truth
for dimensions, physics values and acceptance checks.

## Locked decisions (2026-10-10)

1. **Runs in Docker, headless, on a Paperspace GPU server.** This workstation (RTX 3070, 8 GB)
   is for building and smoke tests only. Nothing may depend on a local X display.
2. **Base = Genie Sim RT Engine (`geniesim_ros`) at commit `6ca11c7`.** Its "5.1" image runs the
   RT Engine on the system-python Isaac Sim **6.0**, which upstream leaves unpinned (it would
   resolve to 6.1.0 today). We pin **6.0.1.0** in `docker/Dockerfile.geniesim`.
   Spec said "Isaac Sim 5.1"; 5.1 only drives Genie Sim's legacy benchmark app.
3. **Teleop = keyboard first, SpaceMouse later. No Pico VR.** Genie Sim's teleop and HDF5
   recorder are Pico-only, so we add our own:
   - `g2cell_teleop`: a terminal keyboard node (works over SSH) that moves a Cartesian EE
     target per arm, calls MoveIt `/compute_ik`, and publishes **`/joint_command`**.
     MoveIt runs with `use_ros2_control:=false` so ours is the only publisher there
     (Genie Sim's `teleop-bridge` pattern).
   - Grippers go through `/joint_command` as well.
4. **Viewing = Foxglove** (`foxglove_bridge` in the image, opened through an SSH tunnel):
   head camera, scene camera, joint states, TF.
5. **Recording = `ros2 bag record` (MCAP) → our HDF5 converter.** There is no upstream converter
   for RT Engine bags.
6. **Assets = the subset we need.** `geniesim_assets/` holds the package files plus
   `scenes/GDK` and `robot/G2_omnipicker` (165 MB), not the full 29 GB. The G2 itself comes
   from the URDF/xacro in `genie_sim_robot_model` (LFS meshes pulled), converted at launch.

## Layout

| path | what |
|---|---|
| `docker/Dockerfile.geniesim` | upstream `Dockerfile.5.1` + Isaac Sim 6.0 pin + the two big pip installs split into smaller layers (same files; registry layer limits, see docs/OPTIMISATION.md "Build & deploy") |
| `docker/build_geniesim.sh` | builds the base image; checks the genie_sim commit and LFS meshes |
| `docker/` (Stage 1, next) | cell image `FROM` the base: foxglove_bridge + our ROS packages; run script, Paperspace notes |
| `ros_ws/src/g2cell_teleop` (Stage 1) | keyboard EE teleop → `/compute_ik` → `/joint_command` |

## Stages

| # | Stage | State |
|---|---|---|
| 1 | Platform: base image, stock `scene_pnp_g2_op` headless, keyboard teleop, Foxglove view | **passed locally**; Paperspace run and Foxglove app check open |
| 2 | Static cell: room, conveyor, table, rack, crates as USD assets | **passed locally** (2026-10-11): `cell.usda` composed and running in Genie Sim with the G2, physics settle/drop tests pass, scripted demo + MP4 renders in `output/renders/`. Open: far half of the pick-table spawn zone is out of arm reach (see checks) |
| 3 | Moving line: surface-velocity conveyor, stop gate, photo-eye, indexing | not started |
| 4 | Bags: three volume-deformable variants, bench-tuned | not started |
| 5 | Task loop: scene controller, sort rules, scoring, recorder | not started |
| 6 | Photorealism + domain randomisation | not started |

No stage starts until the previous one passes its checks.

### Stage 1 checks

- [x] Base image builds from `docker/build_geniesim.sh` with no manual steps (2026-10-11; a docker restart mid-build kills it)
- [x] `docker info` lists the `nvidia` runtime
- [x] `scene_pnp_g2_op` launches headless; G2 joints on `/joint_states` (~100 Hz, best-effort QoS)
- [x] Head camera renders the scene from the official mount (needs `head_cam_tf_relay.py`)
- [~] Foxglove: bridge serves 45 topics incl. the head camera to a `foxglove.sdk.v1` client; not yet
      opened in the Foxglove app through an SSH tunnel
- [x] Keyboard moves each arm's EE (joints track commands within 0.0012 rad) and opens/closes each
      gripper (0 rad = closed, -0.785 = open, confirmed on camera); one `/joint_command` publisher
- [x] A teleop episode records (MCAP) and converts to HDF5 with actions matching measured joints
- [x] Cold start: fresh container + `stage1_up.sh` brings the whole stack up
- [ ] Same image + commands run on the Paperspace server

### Stage 2 checks

Scene: `assets/cell/build.py` -> `geniesim_assets/scenes/g2cell/cell.usda`, YAML
`ros_ws/config/scenes/scene_g2cell.yaml`. Details and deviations: [assets/cell/README.md](assets/cell/README.md).
Evidence in `output/stage2/` and `output/renders/`.

- [x] Assets compose into one scene (`/World` + `/PhysicsScene` like GDK/demo.usda); 123 asset + cell tests pass
- [x] Placement vs spec within 20 mm: conveyor (0.600, 0.000), rack (-0.780, 0.000), pack crate (0.600, 0.000, z 0.820),
      upstream crates Y 0.68/1.32, bags on the table top z 0.870 (USD world bboxes, `test_cell.py`). Robot base:
      engine TF head_link3 = MoveIt FK + init z 0.04 (< 1 mm). Pick table deliberately at (-0.05, **0.85**), not 0.95
- [x] Crates rest on the conveyor, no fall-through: 60 s PhysX settle drift 0.0 mm; drop test from +50 mm lands at
      z 0.820 (pack/lower) and 1.044 (stacked) (`stage2/physics_check.json`, `physics_drop.json`)
- [x] Rigid bags rest on the table: 0.0 mm drift in 60 s, drop test lands at 0.870
- [x] Totes rest in the sort-rack slots: 0.3 mm drift, 20 deg tilt kept
- [x] Physics stable for 60 s: standalone PhysX 60 s run above; Genie Sim engine ran > 25 min at 95-96 Hz,
      rtf 1.00, no overruns in the last stats windows, while the demo was run ~10 times
- [x] Head camera sees the table and the conveyor/pack crate (head yaw 40 deg, pitch 28 deg via `init_joint_pos`;
      `stage2/head_cam_table_and_crate.png`, `renders/g2cell_head_*.png`)
- [x] GPU memory fits: 4.2 GB / 8 GB with sim + render (head 640x400 + FreeCam 1280x720) + MoveIt running
- [~] Reachability (MoveIt `/compute_ik`, `stage2/reach_check.txt`): pack crate reachable by both arms; left arm reaches
      the spawn zone's inner corners and centre; outer corners (Y 1.30) unreachable from a stationary base. Table moved
      0.10 m (max allowed by the chassis); further fixes proposed in assets/cell/README.md
- [x] Scripted demo (`g2cell_demo pick_place_demo`): left arm reaches a bag, grips, lifts, swings over the pack crate,
      releases, returns; right arm re-arranges in the crate. Commands smooth (max 0.062 rad per 36 ms;
      `stage2/joints.csv`), joints track within 0.001 rad; planner keeps arm links >= 0.20 m from the head camera
- [x] Renders: `output/renders/g2cell_overview.mp4` (1280x720), `g2cell_head.mp4` (640x400),
      `g2cell_demo.mp4` (side by side, 1824x540); H.264 yuv420p, 12 fps, 43.8 s; 4 stills per camera; no black frames
- [ ] Known limits: scene rigid bodies are not synced to the render stage (bags never move on camera); the conveyor
      stop-gate joint is inactive in Genie Sim (classifier rejects non-robot joints); occasional ~1 s render pose
      stall seen in 1 of ~6 takes on this 8 GB box

### Layout v2 (2026-10-11): U cell, drive-in, cartons, end of line

Details and every number: [assets/cell/README.md](assets/cell/README.md) "Layout v2". Evidence: `output/v2_work/`
(take7 = the rendered run), `output/reach/layout_v2_final/`, `output/renders/g2cell_packing*`.

- [x] U layout in `assets/cell/build.py` (single generator): table left (centre (-0.215, 0.73), turned -90°), conveyor
      front, sort rack right ((0.0, -0.82), turned +90°, open front facing the robot), U open toward -X. 199 asset tests pass
      (`assets/tests`, incl. the new Layout v2 checks in `test_cell.py`)
- [x] Drive-in clearance: chassis half-width 0.318 m (robot.urdf base_link.dae), U inner edges ±0.43 m → 0.112 m per side;
      min clearance measured along the driven path 0.112 m (chassis) and ≥ 0.10 m for the tucked arms (3D, `env_check`)
- [x] Rack clear of the conveyor (X ≤ 0.275 vs frame 0.305); lips 0.75 / 1.30 m, tilt 20°
- [~] Rack reach: right arm reaches **1 of 2** totes (L1_1, waist -95°); L2_1 (opening Z ≥ 1.59) has no IK at any waist.
      Asked: 2 of 4. The 2-lane rack's second column was behind the robot (0 % reach), so the rack is now 1 column × 2 levels
- [~] Table reach (offline, waist swept): spawn zone 35 % → 41 % (target 80 % not reachable with this table; Bag_01/02
      reachable by MoveIt, Bag_03 not)
- [x] Cartons replace crates (open cartons on the belt incl. the pack position, sealed in the sealer / on the pallet);
      case erector feeding the +Y end; drop gap 0.30 m + floor bin + right-angle transfer (placeholder) + case sealer +
      cobot palletiser along +X (the straight line does not fit the room)
- [x] Scenes: `scene_g2cell.yaml` (pinned) and `scene_g2cell_mobile.yaml` (free base, start (-2.0, 0) facing +X, drive tuck,
      lidars) both on the new `cell.usda`; `mobile_up.sh` gained `WITH_MOVEIT=1` (move_group in the mobile scene;
      default unchanged); pinned scene relaunched and rendering
- [x] Drive-in (`g2cell_demo drive_in`, closed loop on /odom, /cmd_twist): 2.0 m in 10.3 s, v ≤ 0.25 m/s accel-limited,
      no in-place spin; final error **0.57 cm / 0.00°** (5 runs: 0.5–0.6 cm)
- [x] Arm work on the mobile base with MoveIt IK in base_link, waist-first (project rule): table pick waist **+55°**, carton
      place waist 0°, rack drop waist **-95°**; every path checked against the head (≥ 0.20 m) and the fixtures (≥ 0.015 m)
- [x] Base drift during arm work **0.43 cm / 0.07°** (active station-keeping on /odom; without it a reach kicked the free
      base up to 23 cm / 40°, and closing the gripper on the rigid bag pushed it 3.2 cm)
- [x] Video (one continuous take): drive in → left pick (Bag_01) → place over the pack carton, gripper open → right reach
      to tote L1_1, gripper open → retract. `output/renders/g2cell_packing.mp4` (1280x720), `g2cell_packing_head.mp4`
      (640x400), `g2cell_packing_side_by_side.mp4` (1824x540); 67.8 s, H.264, planning time cut out; stills
      `g2cell_packing*_t*.png`, sheets `g2cell_packing_sheet.png`, `g2cell_packing_head_sheet.png`
- [x] Room brighter: light scale 2.2, fill 40 nits (≈1470 lux at the robot)
- [ ] SLAM map `ros_ws/src/g2cell_slam/maps/factory_room.*` is **STALE** (pre-v2 layout); the old Nav2 goals and the
      SLAM start (3.4, 0) now sit in/near the sealer/palletiser. Remap before using Nav2
- [ ] Known limits: bags/cartons still render at their spawn pose (grasp is symbolic: gripper closes 20 mm above the
      rigid bag); the rack drop clearance to the upper stop bar is 2.6 cm; IK planning takes 2–5 min per run (randomised,
      retried); `g2cell_tasks` sort rules still name lanes L1_2/L2_2 that no longer exist (tasks.yaml positions updated)

## RESUME HERE (paused 2026-10-11 evening)

**State:** everything is stopped. The workstation container `g2cell` is stopped (`docker start g2cell` resumes it). The Paperspace server is powered off.

**Done today:** the object pose relay (bags and cartons render where physics puts them); the PickBag transfer fix; the scripted episode generator `g2cell_datagen`; a **pilot dataset of 20 episodes, all successful** (`output/episodes/pilot_v1/`, 9.6 GB); a CPU training dry run (val loss 0.45 → 0.075); kinematic replay tools (`tools/viz/`: RViz live replay + MP4 clips). Audit: `output/readiness/AUDIT.md`. Readiness and server commands: `output/readiness/READINESS.md`.

**Open decision:** for the next step toward real training, pick one of:
1. run 200 scripted episodes on the workstation (about 2.4 h unattended, about 19 GB raw), then push to the A6000, convert and train;
2. a pilot training on the A6000 with the 20 episodes first, to validate CUDA env, speed and checkpoints;
3. fix the caveats first.

**Caveats:**
- The grasp is a scripted attach/release (the rigid placeholder bag can't be held physically).
- Reject→tote episodes are excluded (the left arm can't reach the tote with a bag); this needs a base turn or a right-arm handoff.
- In 3 of 20 episodes the grasp is outside the head camera's view.
- In 12 of 20 the retract can't be planned and recovery goes home instead.

**Housekeeping before collecting more:**
- Workstation disk is ~30 GB free. `output/readiness/pilot/aside/` (3.2 GB of probe runs) can be deleted.
- h5py in the container is broken until the cell image is rebuilt (on the server it's fixed in Dockerfile.cell).

**SSH warning:** the last `ssh paperspace` (after the server was powered off) printed a host-key warning block ("@@@…"). The IP may now belong to a different machine, or the server was reprovisioned. Do NOT bypass it. When the server is back, confirm its IP and host key in the Paperspace console/web terminal (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`; the last known fingerprint was SHA256:1rgGt68GXwA4o8ogFx8qzWHpD7auz1BUzyH8fIHaKis), then update `~/.ssh/known_hosts`.

## Paperspace A6000 (2026-10-11): deployed, audited

Audit: `output/a6000/AUDIT.md` (every claim re-derived from raw logs). Server: RTX A6000 48 GB, Xeon Gold 5315Y 3.2 GHz, 8 vCPU.

| Item | Result |
|---|---|
| Builds on the GPU box | base image 14.5 min, baked cell image 2 min (estimate was 1.5 h) |
| Startup | cold shader compile 391 s (once; cache persisted), warm full stack 46 s |
| Render (`PROFILE=collect`) | 30 fps at target, GPU 40 %, 3.6 GB VRAM, head-cam readback 9–13 ms |
| GDK acceptance | 11/11 pass, 2 expected skips |
| Use-case stack | check_stack 12/12; live PickBag **failed at transfer (ik_failed)**. Fixed offline (waist sequencing, pick-column lookahead, safe recovery): Bag_01 + Bag_02 motion-complete 5/5 in the replay, not yet re-run in the sim (docs/OPTIMISATION.md "PickBag transfer fix") |
| Packing demo + video | end to end; drive-in 0.6 cm; base drift 0.14 cm; planning 17 s cold, 0.3 s cached; `output/a6000/a6000_packing.mp4` |
| **Physics** | **49 Hz instead of 100 Hz (43 Hz in the demo)**: the CPU is ~2–2.4× slower per physics step than the workstation i9-14900K |

**Decision (2026-10-11): split the work.** Real-time sim, teleop and data collection run on the
workstation (i9, 100 Hz physics). The A6000 runs training (`g2cell_learning`, `configs/server_a6000.yaml`)
and showcase renders (`PROFILE=showcase`), which are not real-time-critical. Idle auto-shutdown
is installed on the server (30 min; `sudo scripts/idle_shutdown.sh status|disable|uninstall`).
Fixes from deployment: `sync_to_server.sh` now copies genie_sim's generated URDFs (MoveIt needs them);
`stage1_up.sh` has `READY_TIMEOUT_S` (default 900 s) for cold shader compiles.

## Latest (2026-10-11, afternoon)

| Item | Result | Evidence |
|---|---|---|
| Layout v2 (U-cell, cartons, erector → conveyor → drop gap + floor bin → transfer → sealer → palletiser; brighter room ~1470 lux) | done | `assets/cell/README.md` "Layout v2"; 199 asset tests |
| Drive-in to the U (closed loop on /odom, no in-place spin) | 0.5–0.6 cm / 0.00° over 5 runs, 0.112 m clearance per side | `output/v2_work/` |
| Waist-first arm work (waist +55° table, 0° carton, −95° rack) | base drift 0.43 cm / 0.07° | motion 80 / tasks 62 / teleop 84 tests |
| Packing video (67.75 s, H.264) | verified | `output/renders/g2cell_packing{,_head,_side_by_side}.mp4` |
| Sim-backed `agibot_gdk` 3.3.8, live acceptance | **11/11 training solutions pass** against Genie Sim (2 expected skips: wrist cameras, real-network doctor); EE pose miss 0.65 mm | `output/gdk_acceptance/20261011_040820/` |
| Object pose relay: bags/cartons/totes render where PhysX puts them (`OBJ_RELAY=1`, default; engine = upstream + one post-step hook in `g2cell_bringup`) | verified live in the pinned cell: Bag_01 teleported into the pack carton, Bag_03 dropped to the floor, Bag_02 and Carton_Up_01 pushed; head + FreeCam frames match the physics poses. Cost: 0.17 ms per 25 Hz sample, +0.05 ms per physics tick, ~0.4 % of one core; render 30.0 fps / 22.8 ms and physics step 4.45 ms are the same with and without the hook | `output/readiness/relay/` (PNGs, `perf_relay_{on,off}_idle.json`, `sim_*.log`); 22 bringup tests (15 new) |
| Pilot dataset `pilot_v1` (datagen, 20 seeds, teleport reset, sticky grasp, fixture-aware planner) | 20/20 `success` (bag in carton, scored from relay poses), 42.4 s/episode wall, physics 99.7 Hz; CPU train 300 steps, val split by episode, L1 0.45 → 0.075 | `output/readiness/READINESS.md` |

Still open: rack-tote drops (reject label) fail planning for the left arm at all 35 sweep points; 3/20 pilot grasps happen with the waist ~0° (out of the head camera's view); table spawn zone 41% reachable (target 80%); rack tote L2_1 (upper) is not reachable by the planner, only L1_1; SLAM map stale for Layout v2; arm–fixture contact can push the free base (mitigated with fixture collision checks + base hold).

## Morning summary (2026-10-11)

All planned items are built and checked. "Verified" means the evidence was re-run or inspected
directly, not only reported.

| Item | Result | Evidence |
|---|---|---|
| Sim environment: G2 in the pack-off cell | verified | `output/renders/g2cell_overview_t08.0s.png`; Stage 2 checks below |
| Videos | verified: 3 cell videos of 43.8 s, 1 SLAM/Nav video of 37.7 s, all H.264 | `output/renders/*.mp4` |
| Teleop (keyboard + SpaceMouse, record/mark/home keys) | verified live in the cell | 76 + 24 tests; bag → HDF5 → `success=true` on `output/episodes/teleop_20261010_180355` |
| Training (dataset, LeRobot export, ACT-lite/BC, eval) | verified on CPU, one episode only (pipeline proof, not a policy) | 31 tests; `output/train/smoke_act/eval/replay_stage1_teleop.png` |
| Use-case packages (interfaces, motion, tasks, bringup) | verified live: `check_stack` 12/12, a PickBag ran all 6 stages on the left arm | 115 tests; result `unverified` (no Stage 5 scorer yet) |
| SLAM + Nav2 (mobile G2, 2 lidars, slam_toolbox, Nav2) | verified: map saved; 2/2 Nav2 goals succeeded, 0.10 m and 0.084 m final error | `ros_ws/src/g2cell_slam/maps/`, `output/slam/nav_result_*.json`, `output/renders/g2_slam_nav.mp4` |
| G2 SDK | none public; models gathered | `docs/G2_MODELS.md` |

Open issues, most important first:
1. ~~**Non-robot objects don't move in the render.**~~ Fixed 2026-10-11 by the object pose relay
   (`OBJ_RELAY=1`, see "Latest"). A grasp still has to physically move the bag: the scripted grasp is symbolic.
2. **Bags are rigid placeholders.** Deformable bags are Stage 4.
3. **Reach:** the right arm can't reach the pick table, and the left arm only reaches about
   Y ≤ 0.85. Keep bags on the near half or let the base drive about 0.3 m.
4. **The conveyor stop-gate joint is disabled.** Genie Sim refuses to start with it, so Stage 3 needs another way to drive the gate.
5. **Lidar mounts are estimates.** The official G2 files have none. Odometry is ground truth.
6. **The image needs a rebuild on Paperspace** to get pip h5py, spacenav, slam/nav2. Locally they were installed
   into the running container; a local image build needs about 50 GB free.
7. **Arm self-collision is off in Genie Sim.** The motion planners enforce 0.20 m head clearance instead.

## Overnight run (2026-10-11 → morning)

Scope: the sim environment, training and teleop packages, and video renders of the G2
in the test environment. After that succeeds: use-case packages (ROS 2, SLAM, and the rest
needed for use-case development).

| Track | Runs as | Needs the sim? | Output |
|---|---|---|---|
| A. Cell scene + renders — **done**: `output/renders/g2cell_{overview,head,demo}.mp4` (43.8 s each), Stage 2 checks in PLAN | sim track (only one process may run/restart the sim) | yes | `assets/cell/` scene composed of all Stage 2 assets with the G2, Stage 2 checks in sim, a scripted demo motion, MP4 renders in `output/renders/` |
| B. Learning package — **done** (31 tests; CPU smoke-train loss 0.456→0.055 on 1 episode, train=val) | offline | no (CPU only) | `ros_ws/src/g2cell_learning`: HDF5 → LeRobot-style dataset, ACT-lite/BC trainer, eval/replay, tests, CPU smoke-train on recorded episodes |
| C. Teleop package v2 — **done**: 76 + 24 tests; integration-tested in the cell scene (arm moves track within 0.0012 rad, home key, record start/stop → bag → host HDF5 → mark_episode success). The container needs pip h5py (Paperspace image) for in-container conversion | offline | no (unit tests; integration later) | SpaceMouse device, record start/stop + episode reset keys, one launch file |
| D. G2 SDK packages | done (research) | no | **No public G2 SDK.** The G2's "Genie 02 GDK" installs only from the robot (`curl http://10.42.1.101:8849/install.sh`, PC on 10.42.1.x, Ubuntu 22.04 + ROS 2 Humble). It provides `genie_msgs`, `agibot_gdk` (Python) and `libgdk_adapter.so` (x86_64 + aarch64). Manual: `../g2_sdk/genie02_gdk_manual_v2.1.0/`. Sim interfaces (`geniesim_msg`, Jazzy) differ from the real-robot `genie_msgs`, so a bridge is needed later. |
| E. Use-case packages (only after A–C pass) — **ROS 2 part built (2026-10-11), not yet run against the sim**: `g2cell_interfaces` (3 msg, 3 srv, 2 action), `g2cell_motion` (MoveArm server, 46 tests), `g2cell_tasks` (PickBag server + cell_manager skeleton, 55 tests), `g2cell_bringup` (launch + check_stack, 7 tests); `g2cell_demo` refactored onto g2cell_motion (7 tests pass); all 8 g2cell_* build in the container; `scripts/offline_smoke.py` 14/14 against a fake robot on an isolated ROS domain. Open: live-sim run of motion_server/PickBag, grasp heuristic validation, rack-lane reach. `g2cell_slam` **done 2026-10-11** (mobile scene, map, 2/2 Nav2 goals; see "Use-case: SLAM/Nav") | parallel tracks | lidar/mobile scene: yes, serially | `g2cell_interfaces`, `g2cell_bringup`, `g2cell_slam` (slam_toolbox / Nav2 on Jazzy in the image, G2 lidar in a mobile scene), task skeletons |

Morning report: what passed (with evidence), what failed and why, and where the videos are.

### Use-case: SLAM/Nav

Mobile scene `ros_ws/config/scenes/scene_g2cell_mobile.yaml` (same `cell.usda`, `pin_base_to_world: false`, start
(3.4, 0) facing -X), started with `scripts/mobile_up.sh`. Package `ros_ws/src/g2cell_slam` (README there).
Evidence is in `output/slam/` and `output/renders/g2_slam_nav.{mp4,png}`.

- [x] Mobile G2 base drives from a velocity command.
      Interface: `/cmd_twist` (Twist) -> `servo_node` (genie_sim_controllers, `default_servo:=GeneralServo`) -> `/cmd_4ws` -> engine.
      Measured: 0.2 m/s forward, then 0.83 m in about 4 s; spin 75 deg in 3 s with 3 cm or less drift. The default MPC servo arcs or
      stalls on spin, so it is not used. `vy` is ignored (non-holonomic use).
- [x] Odometry and TF from the engine: `/odom` at 99 Hz and `odom -> base_footprint -> base_link`. This is ground truth (odom == world).
      `fake_slam:=false` removes the engine's identity `map -> odom`.
- [x] LiDARs publish: `/lidar/livox_front` and `/lidar/livox_back` at about 4 Hz each, about 14k points per scan.
      Ranges match the room: rear wall 2.0 m, side walls 4.9 m, front wall 7.8 m, ceiling 3.97 m (`output/slam/sensor_check.txt`).
      - The official G2 USD has **no** lidar prims, so the mounts are **estimated**: (+-0.40, 0, 0.18) on base_link.
      - The prims are injected into the staged robot USD by `scripts/inject_lidars.py`.
      - The lidar launcher config is `config/launchers/launcher_ovrtx_isaac_physx_lidar.yaml`.
      - The relay was extended with `--lidar-body base_link`.
- [x] Packages installed in the container (and added to `docker/Dockerfile.cell`): ros-jazzy-slam-toolbox 2.8.5,
      ros-jazzy-pointcloud-to-laserscan 2.0.2, ros-jazzy-navigation2 1.3.13, ros-jazzy-nav2-bringup 1.3.13 (+ apt deps).
- [x] Mapping: `cloud_merger` -> `/lidar/merged_full` (8 Hz) -> `pointcloud_to_laserscan` -> slam_toolbox (online async).
      Scripted loop: 98 s, 22.8 m, at most 0.35 m/s.
      - Map: `ros_ws/src/g2cell_slam/maps/factory_room.{pgm,yaml}` (0.05 m).
      - Wall coverage within 0.1 m of the true line: W 99 %, N 99 %, S 100 %, E 64 % (the east wall is on the map edge).
      - All fixtures fall inside their ground-truth outlines (`output/slam/factory_room_preview.png`).
- [x] Nav2 (AMCL + RPP controller) on the saved map. Initial pose taken from ground truth.
      - Open floor -> in front of the conveyor (1.65, -0.80, 180 deg): **succeeded in 11.9 s**. Final error 0.10 m / 0.9 deg;
        AMCL error 0.022 m.
      - Around the cell -> rack side (-2.0, 0.3, 0 deg): **succeeded in 37.7 s**. Error 0.084 m / 4.0 deg;
        AMCL error 0.025 m (`output/slam/nav_result_*.json`).
- [x] Render: `output/renders/g2_slam_nav.mp4` (37.7 s, 1500x540, H.264). Free camera beside the live map: mapping at 4.5x,
      then the second Nav2 goal at 2.5x. Still: `g2_slam_nav.png`; contact sheet: `output/slam/video_contact.png`.
- [ ] Real-robot fidelity:
      - Odometry is ground truth (no drift).
      - Lidar mounts and profile are estimates; the lidar is idealised (instant scans, no noise).
      - Two lidars cut render fps to 11.6 on the 8 GB card.
      - `use_sim_time` must stay false (the engine stamps wall time).
- [ ] Not yet run on Paperspace; not combined with MoveIt (mobile scene runs without move_group).

## Known risks

- Deformables: Genie Sim ships no PhysX deformable scene; Newton cloth is marked experimental.
  Stage 4 will be our own work on the PhysX deformable schema in Isaac Sim 6.0.
- 8 GB VRAM here: expect only low-res, real-time checks locally; real runs on Paperspace.
- The 6.0 pin may still drift from what upstream tests. If the RT Engine breaks, try 6.0.0.1.
