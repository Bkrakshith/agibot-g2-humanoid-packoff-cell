# g2cell_datagen

Generates training episodes without a person at the keyboard. For each seed it:

1. Samples the randomisation.
2. Resets the scene.
3. Starts `g2cell_recorder`.
4. Runs the PickBag task.
5. Stops the recorder and marks the outcome.
6. Writes `episode.json`.

Rerunning the same command resumes the batch: seeds that are done are skipped. Operator mode wraps
keyboard teleop the same way. It automates reset, recording and marking, and the person does the task.

**State (2026-10-11, evening): pilot batch `pilot_v1` generated live, 20/20 episodes `success` (bag verified in the
pack carton by the scorer), reset `teleport`, sticky grasp on.** Evidence and numbers: `output/readiness/READINESS.md`.
Labels are good-only for now: the left arm's transfer to the rack tote failed planning at all 35 offline sweep points.

## What is randomised (cheap now)

Everything comes from `random.Random(seed)`, so an episode can be replayed from its seed alone
(`config/datagen.yaml`, `randomize.py`):

| factor | range | source |
|---|---|---|
| which bag is picked | Bag_01 / Bag_02 / Bag_03 | the 3 rigid placeholders in `cell.usda` |
| label → target | good 70 % → pack carton, reject 30 % → rack tote L1_1 | g2cell_tasks sort rule (`tasks.yaml`) |
| bag centre | X −0.25..0.15, Y 0.52..0.80 and x − y ≥ −0.85 (cell frame; the far corner is cut, measured with `tools/perf/reach_sweep.py`) | reachable part of the spawn zone: zone X −0.665..0.235 × Y 0.48..0.98 (assets/cell/README.md Layout v2), reach band X −0.4..0.35, left arm Y ≤ 0.85 (`left_table_max_y`) |
| bag yaw | −90..90° (the box bag looks the same after a 180° turn) | spec "random yaw" |
| other bags | parked on the floor behind the table at (−0.40, 1.40) and (−0.10, 1.40): out of reach, so PickBag "any" never takes them | |

Rejection sampling keeps every corner of the rotated 0.21 × 0.30 m bag at least 0.02 m inside the table
footprint, and keeps it 0.03 m or more from any parked bag. `validate_config` refuses a range that leaves
the spawn zone, the reach band or the left arm's Y limit. It also refuses a label whose target no arm can
reach (e.g. `sample` → L2_1). The tests cover all of these checks.

Not randomised yet: lighting, camera jitter, table and conveyor offsets, bag graphics, stiffness and
flips, and the lane shuffle. Only one rack tote (L1_1) is reachable, so there is nothing to shuffle.

## Reset: how the scene is put back

**Default since the pilot: `teleport`.** With the object pose relay (`OBJ_RELAY=1`, the stage1_up.sh default) the
runner moves every bag (picked bag at its sampled pose and yaw, the others to the parking spots) and puts the pack
carton back at (0.60, 0, 0.82) through `/g2cell/object_poke` (velocities zeroed), waits `settle_s`, and checks the
poses the relay reports (1 cm / 3°; `scene_io.py`). ~5.7 s per episode including homing. After the task, the bag's
resting place is scored (`placement_of` / `score`: carton interior, tote, table, floor, conveyor), so a completed
motion becomes `success` / `drop` / `wrong_target` / `overflow` / `grasp_failed` (verified) instead of `unverified`.
The grasp itself is the engine's sticky grasp (`GRASP_ASSIST=1`, g2cell_bringup `object_relay.GraspAssist`): the
omnipicker cannot physically hold the 0.21 m rigid box.

The older modes:

Genie Sim's RT engine has **no reset or teleport service**. At 6ca11c7 its ROS node only subscribes to
`/joint_command`, `/cmd_4ws` and the viewer camera pose (`genie_sim_engine/src/realtime_ros_node.cpp`).
rlinf's `ResetScene` exists only in the MuJoCo RL env. So there are two modes (`--reset`):

| mode | what happens per episode | cost | renders show |
|---|---|---|---|
| `none` (default) | The arms are homed through MoveArm, and `pick_bag_server` restarts with the episode's `tasks.yaml` (randomised bag poses and labels). The physical bags stay where `cell.usda` put them. | ~5 s | the built bag poses (the grasp is symbolic anyway: Stage 2 notes) |
| `restart` | Writes `/geniesim_assets/scenes/g2cell/datagen/<batch>/<ep>.usda`, a thin layer that sublayers `cell.usda` and overrides `/World/Bags/*` xforms, plus a scene YAML pointing at it. Then `tmux kill-session -t g2cell`, `stage1_up.sh <variant.yaml>` (`PROFILE=collect`), and a wait for `/joint_states`. | ~1 min warm | the randomised poses (the physics stage and the render stage both open the layer) |

The override layer was checked with usd-core against the real `cell.usda` (test
`test_override_on_the_real_cell_usda`). A Genie Sim launch on such a layer has **not** been tried yet.

Bags still do not move in the render after a grasp; Stage 3 needs a pose relay for that. Until then the
outcome is `unverified` (no scorer). The runner maps it to HDF5 `success=unknown`;
`--unverified-as true` labels completed motions as successes. `success`/`regrasp` → `true`;
`plan_failed`, `grasp_failed`, `drop`, `wrong_target`, `unreachable` → `false`.

## Commands (workstation, physics 100 Hz)

The runner must **not** run inside the `g2cell` tmux session, because `--reset restart` kills that
session. Use `docker exec`, or a separate tmux session in the container.

```bash
# 0. build (once after edits)
docker exec -u isaac-sim g2cell bash -lc 'source /workspace/devel/setup.bash && cd /g2cell_ws && colcon build --packages-select g2cell_datagen'
D="docker exec -it -u isaac-sim g2cell bash -lc"
SRC="source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash"

# 1. the sim: pinned cell scene, collect profile (head RGB-D 30 Hz). With PROFILE the sim is pinned to the
#    P-cores, so physics holds 100 Hz on the i9-14900K (docs/OPTIMISATION.md "Sim runtime")
docker exec -u isaac-sim g2cell bash -lc 'PROFILE=collect bash /g2cell_ws/scripts/stage1_up.sh /g2cell_ws/config/scenes/scene_g2cell.yaml'
#    do NOT start cell.launch.py / teleop: the runner starts motion_server + pick_bag_server itself
#    (exactly one /joint_command publisher). Stop any teleop first.

# 2. look at the plan (no ROS, no files)
$D "$SRC && ros2 run g2cell_datagen episode_runner --batch pick_v1 --seeds 0-49 --dry-run"

# 3. generate (resumable: rerun the same line after a crash or Ctrl+C)
$D "$SRC && ros2 run g2cell_datagen episode_runner --batch pick_v1 --seeds 0-49"
$D "$SRC && ros2 run g2cell_datagen episode_runner --batch pick_v1r --seeds 0-49 --reset restart --profile collect"

# 4. after a batch: status, convert bag-only episodes (if the container had no h5py), dataset card
$D "$SRC && ros2 run g2cell_datagen datagen_batch status /g2cell_output/episodes/pick_v1"
$D "$SRC && ros2 run g2cell_datagen datagen_batch finalize /g2cell_output/episodes/pick_v1"
$D "$SRC && ros2 run g2cell_datagen datagen_batch card /g2cell_output/episodes/pick_v1 --name pick_v1"
```

On the host, without ROS (`card` needs h5py + numpy; `finalize` needs a python with the recorder's
decoders, e.g. `.venv-learn` for the JPEG frames):

```bash
cd g2_packoff_cell
PYTHONPATH=ros_ws/src/g2cell_datagen:ros_ws/src/g2cell_tasks:ros_ws/src/g2cell_learning:ros_ws/src/g2cell_recorder \
  ../.venv-learn/bin/python -m g2cell_datagen.batch card output/episodes/pick_v1 --name pick_v1
```

Useful flags (`episode_runner --help`):

- `--n 50 --seed-start 100` as an alternative to `--seeds`.
- `--limit 5` runs at most 5 pending episodes now.
- `--no-retry-errors`: by default, error seeds are retried.
- `--max-consecutive-errors 3` stops the batch when the sim dies.
- `--task-timeout 600`.
- `--external-stack` uses an already running `pick_bag_server`. That server does not see the episode's
  `tasks.yaml`, so only the bag and target are randomised.
- `--no-home`, `--no-record`.
- `--node-ros-arg="-r /joint_command:=/offline/joint_command"` adds remaps for offline smoke runs.

### Operator in the loop

```bash
# window 1: the teleop (the single /joint_command publisher; do not use its r/n/f record keys)
docker exec -it -u isaac-sim g2cell bash -lc "$SRC && ros2 run g2cell_teleop keyboard_teleop --ros-args -p use_sim_time:=true"
# window 2: the runner prints where the bag is and where it goes, records, and asks s/f/d/q
$D "$SRC && ros2 run g2cell_datagen episode_runner --batch teleop_v1 --seeds 0-9 --mode operator"
```

In operator mode the runner starts no servers and does not home the arms. Answer `s` for success or `f`
for failure; the outcome is marked in the HDF5. `d` discards the episode and keeps the seed pending.
`q` discards and quits. Use `--reset none` here: `restart` would kill the teleop window together with the
sim. With `none`, the operator sees the printed target pose and places the bag. The bag prims do not move
in this mode.

## Output

```
output/episodes/<batch>/batch.json                     args, config + tasks snapshot, one entry per run
output/episodes/<batch>/ep_s00007/episode.json         seed, randomisation, reset info, outcome, success,
                                                       task result, recording result, timings (reset_s,
                                                       record_start_s, task_s, stop_s, total_s), status, error
output/episodes/<batch>/ep_s00007/tasks.yaml           what pick_bag_server believed
output/episodes/<batch>/ep_s00007/bag/ + ep_s00007.hdf5   recorder output (compact defaults: JPEG head RGB,
                                                       compressedDepth 5 Hz, MCAP zstd_fast)
output/episodes/<batch>/_crashed/ep_s00007.1/          leftovers of an interrupted attempt (never reused)
output/episodes/<batch>/_logs/                         motion_server / pick_bag_server / sim restart logs
output/episodes/<batch>/dataset_card.{json,md}         g2cell_learning dataset card + "datagen" section
```

`status` values:

- `running`: written before anything moves. Still there after a crash, so the seed reruns.
- `done`: the episode finished.
- `error`: a reset, recorder, timeout or `aborted`/`invalid` failure. Retried on the next run.
- `discarded`: operator mode. The seed stays pending.
- `interrupted`: Ctrl+C. The bag is closed and the seed reruns.

If the container python has no h5py, the recorder exits 3 with the bag written. The episode still counts
as `done`; run `finalize` to convert it and to apply the mark from `episode.json`.

## Training on the server

```bash
# workstation -> server (HDF5 + episode.json + card; MCAP bags only with --with-bags)
scripts/sync_to_server.sh --push-dataset pick_v1            # --dry-run to preview
# equivalent rsync line
rsync -az --partial --exclude bag/ --exclude '*.mcap' --exclude _crashed/ \
  output/episodes/pick_v1/ paperspace:g2/g2_packoff_cell/output/episodes/pick_v1/
# on the server: g2cell_learning (configs/server_a6000.yaml)
python -m g2cell_learning.train --episodes 'output/episodes/pick_v1/ep_s*/ep_s*.hdf5' ...
```

## First live run (done 2026-10-11)

Fixed on the way (details in `output/readiness/READINESS.md`): the installed recorder was stale (raw 1.4 GB
bags); depth throttle relay rejected `rate_hz:=5` (int); the approach from home threw the pack carton (fixture
check in the planner + `clear` pose); the grasp never lifted the bag (sticky grasp); reset `none` left the bags
where they were (`teleport`); unreachable far corner and rack transfers (region cut, good-only labels).

## Tests

```bash
cd ros_ws/src/g2cell_datagen
../.venv-tools/bin/python -m pytest -q test     # 93 passed
```

The tests cover: randomisation ranges, reachability, footprint and overlap checks; determinism and resume
per seed; config validation; the runner state machine with a fake backend (happy path, task, reset and
recording failures, infra outcomes, Ctrl+C, consecutive-error abort, operator verdicts); the recorder
adapter against a fake `record_episode.sh` (SIGINT, conversion, marking, deferred conversion); the
USD/YAML scene variants (including the real `cell.usda` through usd-core); process groups; the CLIs; and
`sync_to_server.sh --push-dataset` with a fake ssh and rsync.
