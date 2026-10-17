# g2cell_learning

Imitation-learning tools for the G2 pack-off cell: episode HDF5 → torch dataset or LeRobot v2
folder → ACT-lite / BC-MLP training → offline evaluation. Plain Python. ROS is not needed;
`setup.py` + `package.xml` (ament_python) let it also build in the ROS workspace.

## Data flow

```
keyboard teleop (g2cell_teleop) ──► /joint_command ─┐
Genie Sim (head RGB/depth, free cam, /joint_states) ┴─► ros2 bag record (MCAP)
   └─► g2cell_recorder.bag_to_hdf5 ──► episode.hdf5 (raw streams + /aligned 30 Hz view)
         ├─► prepare_cache ──► uint8 memmap frames ──► dataset.CachedEpisodeDataset (GPU training)
         ├─► dataset.EpisodeDataset            torch Dataset straight from the HDF5 (eval, CPU)
         ├─► export_lerobot                    LeRobot v2.0-style folder (parquet + mp4)
         └─► dataset_card                      per-batch card (.json + .md)
   train.py (ACT-lite | BC-MLP) ──► checkpoints, loss.csv ──► eval_offline.py ──► MSE + replay PNG
```

### What the dataset returns

All data comes from the HDF5 `/aligned` group (30 Hz, nearest-previous sample per stream).

| key | shape | content |
|---|---|---|
| `observation.images.head_front_rgb` | [3,H,W] float in [0,1] (`EpisodeDataset`) or uint8 (`CachedEpisodeDataset`; the policy converts on the device) | head RGB, resized (default 128×160); `--extra-cameras free_rgb` adds the free camera |
| `observation.state` | [16] | `/joint_states` positions of the 16 joints `/joint_command` commands (7+gripper per arm), **selected by name** |
| `action` | [K,16] | `/joint_command` at aligned steps `t+offset … t+offset+K-1` (default offset 1 = the next K commands, K = 20) |
| `action_is_pad` | [K] bool | true past the episode end; padded rows repeat the last action |
| `episode_index`, `frame_index` | scalar | |

State and action are z-scored with stats from the train split (`norm_stats.json`, also stored
in every checkpoint). A joint that never moves gets std = 0.01 (`std_floor`), not ~0. Joints
not yet commanded at the start of an episode take their measured position ("hold"). Episodes
with no `/joint_command` (such as `it_test1`) are rejected by the dataset and skipped by the
exporter; the dataset card still lists them.

### Policies (`policy.py`)

- **ACT-lite** (`--policy act`): CNN feature map per camera (shared backbone) → 1×1 projection +
  2-D sine positions + camera embedding, plus one proprio token → transformer encoder; K learned
  queries → transformer decoder → linear head. L1 loss on non-padded chunk steps. It differs from
  ACT: **no CVAE latent**, no temporal ensembling, small default size.
- **BC-MLP** (`--policy bc`): global-pooled CNN features + state → MLP → the whole chunk.
- Backbones: `small` (4-layer CNN, default, CPU-friendly) or `resnet18` (torchvision,
  `weights=None`, trained from scratch, so no download).

## Setup (CPU venv)

```bash
conda create -y -p ../.venv-learn python=3.11   # from the work folder next to this repository
../.venv-learn/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
../.venv-learn/bin/pip install numpy h5py pyarrow pytest av matplotlib
```

The venv is about 1.6 GB. All commands below run from `g2_packoff_cell/` with:

```bash
export CUDA_VISIBLE_DEVICES=""                       # CPU only on this workstation
export PYTHONPATH=ros_ws/src/g2cell_learning
PY=../.venv-learn/bin/python
```

## Commands

```bash
# tests (synthetic fixtures + one integration test on the real stage1_teleop file if present)
$PY -m pytest ros_ws/src/g2cell_learning/test -q

# dataset card for a batch
$PY -m g2cell_learning.dataset_card output/episodes/*/*.hdf5 -o output/datasets/stage1_card --batch-name stage1

# LeRobot v2.0-style export (repeat --camera for more videos)
$PY -m g2cell_learning.export_lerobot output/episodes/stage1_teleop/stage1_teleop.hdf5 \
    -o output/datasets/stage1_lerobot --task "stage 1 keyboard teleop" \
    --camera head_front_rgb=head --camera free_rgb=free

# train (real runs: many episodes, a val split, more steps)
$PY -m g2cell_learning.train --episodes 'output/episodes/*/ep*.hdf5' --out-dir output/train/act1 \
    --policy act --steps 20000 --batch-size 32 --val-fraction 0.2 --cache-dir output/cache --num-workers 4
#   resume:  --resume output/train/act1/checkpoints/last.pt --steps 40000   (or --resume auto)
#   options: --policy bc | --backbone resnet18 | --image-size 224 224 | --extra-cameras free_rgb
#            --chunk-size 20 | --action-offset 1 | --loss l1|mse | --threads 8 | --grad-accum 2
#            --amp auto|bf16|fp16|off | --compile | --no-channels-last | --early-stop-patience 8
#            --ckpt-minutes 10 | --config file.yaml (any flag as key: value; CLI overrides)

# frame cache on its own (train builds it on demand when --cache-dir is set)
$PY -m g2cell_learning.prepare_cache output/episodes/*/*.hdf5 --cache-dir output/cache --image-size 128 160

# GPU server (RTX A6000 48 GB, 8 vCPU): estimate first, then train
$PY -m g2cell_learning.train --config ros_ws/src/g2cell_learning/configs/server_a6000.yaml --dry-run
$PY -m g2cell_learning.train --config ros_ws/src/g2cell_learning/configs/server_a6000.yaml

# offline eval (defaults to the checkpoint's val split)
$PY -m g2cell_learning.eval_offline --ckpt output/train/act1/checkpoints/last.pt --out-dir output/train/act1/eval
```

Train outputs: `config.json`, `split.json` (split by **episode**), `norm_stats.json`, `loss.csv`
(`step,train_loss,val_loss,lr,elapsed_s,samples_per_s`, logged every `--eval-every` steps), and
`checkpoints/step_XXXXXX.pt`, `last.pt` and `best.pt` (lowest val loss). Each checkpoint holds the model,
optimizer, GradScaler and RNG states, config, split, joint names and norm stats. The val loss uses the
same fixed subset of val frames every time, so successive values are comparable. Eval writes `metrics.json` (`action_mse` = first predicted step
vs. the recorded command, in rad²; `chunk_mse` = all non-padded chunk steps; per-joint and
per-episode values) and `replay_<episode>.png`. Each PNG plots both grippers and the 4
most-moving arm joints, predicted against recorded.

## Smoke run (2026-10-11, CPU, 8 threads)

| | |
|---|---|
| Export | `output/datasets/stage1_lerobot/`: 1 episode, 583 frames, head + free mp4 (h264) |
| Card | `output/datasets/stage1_card.{json,md}` (stage1_teleop + it_test1) |
| Train | `output/train/smoke_act/`: ACT-lite, small CNN, 2.1 M params, 128×160, K=20, batch 16, lr 3e-4, 400 steps, 55 s |
| Loss (L1, normalised) | train 0.456 (first 25 steps) → 0.055 (last 25); val 0.365 → 0.040 |
| Eval | `output/train/smoke_act/eval/`: action MSE 6.4e-4 rad², chunk MSE 3.8e-4 rad² |

**The smoke run trains and validates on the same single episode** (`--train-equals-val`). The
numbers show that the pipeline runs end to end and that the model can fit (memorise) one
demonstration. They say nothing about generalisation. In the replay the step-0 prediction runs
slightly ahead of the recorded steps, because the chunk loss pulls it toward future commands.

## Limitations

- **One short episode** (20 s, left arm only, right arm constant). Real training needs tens to
  hundreds of demonstrations with success labels, a real episode-level val split, and the
  Paperspace GPU (`--device cuda`, larger images and backbone, 10⁴–10⁵ steps).
- Offline MSE is open-loop. It does not measure task success; closed-loop rollouts in Genie Sim
  are still to be built (a ROS node that runs the policy and publishes `/joint_command`).
- ACT-lite has no CVAE and no temporal ensembling. Depth is not used yet.
- Without `--cache-dir`, `EpisodeDataset` decodes every aligned frame into RAM as float32 at every start
  (about 0.7 s and 60 MB per 20 s episode, per process). That is fine for a few episodes. For real
  datasets use `--cache-dir`: frames are decoded once, stored as uint8 memmaps (about 15 MB per episode
  at 128×160) and shared by all DataLoader workers through the page cache.
- The camera runs at about 13.5 Hz, so the 30 Hz aligned view repeats each frame 2–3 times.

### LeRobot format caveats (written from memory; lerobot was not installed to check)

- Layout, `info.json` keys (`codebase_version: "v2.0"`, `data_path`/`video_path` templates,
  `chunks_size: 1000`, `splits`, `features` with `dtype/shape/names`), `episodes.jsonl`,
  `tasks.jsonl` and the parquet columns follow LeRobot v2.0 as I remember it. Unverified:
  - the video feature metadata key (`info` vs. `video_info`);
  - `names: {"motors": [...]}` vs. a flat list;
  - whether vectors should be `fixed_size_list` (they are written as `list<float32>`);
  - the exact `stats.json` shapes (image stats are `[3][1][1]`, low-dim per joint; no stats for
    the index columns).
- `episodes.jsonl` carries extra keys (`source`, `scene`, `seed`, `success`) for traceability.
- Videos are h264/yuv420p with GOP 2 by default. LeRobot's default is AV1 (`--vcodec libsvtav1`
  is available).
- v2.1 moved to per-episode `episodes_stats.jsonl`, and v3.0 changed the layout again. Run
  LeRobot's own loader/converter once before relying on this export.

## Server training (GPU)

`configs/server_a6000.yaml` holds the defaults for the paid A6000 box. Paths are relative to
`g2_packoff_cell/`.

| setting | value | why |
|---|---|---|
| `cache_dir` | `output/cache` | uint8 memmaps: no per-sample gzip inflate or resize; ~33–41k samples/s from the loader on 8 CPU cores here |
| `num_workers` / `pin_memory` / `persistent_workers` / `prefetch_factor` | 6 / true / true / 4 | 8 vCPU: 6 loader workers + the main process |
| `batch_size` × `grad_accum` | 256 × 1 | 48 GB; raise `grad_accum` rather than `batch_size` when images get larger |
| `amp` | auto | bf16 autocast on Ampere, fp16 + GradScaler where bf16 is unsupported, fp32 on CPU |
| `channels_last`, `compile` | true, false | try `--compile` after the dry run |
| `ckpt_minutes` | 10 | `last.pt` at least every 10 min; SIGTERM/SIGINT also save it |
| `resume` | auto | restarting the same command after a preemption continues from `last.pt` |
| `early_stop_patience` | 8 evals | stops after 4000 steps without val improvement; `best.pt` keeps the best |

`--dry-run` builds the cache and the model, then times the loader alone and `--bench-steps` training steps.
It writes `out_dir/dry_run.json` with samples/s, s/step, estimated epoch and run time, peak GPU memory, and
the cost when `--price-per-hour` is set. It also flags a loader-bound setup. Nothing is trained or
checkpointed. On this CPU-only workstation the dry run reports ~130–175 samples/s (the small CNN is
compute-bound). The A6000 figure has not been measured; run the dry run there before committing hours.
