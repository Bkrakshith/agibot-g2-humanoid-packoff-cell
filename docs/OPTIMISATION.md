# Optimisation

How the G2 cell is made cheaper and faster to build, deploy and run. Each section covers one area
(sections are delimited by `<!-- BEGIN/END section: ... -->` markers).

<!-- BEGIN section: Build & deploy -->
## Build & deploy

Goal: spend billed A6000 hours only on running the sim. Images are built on a cheap CPU machine,
pushed to a **private** registry, and pulled on the GPU box. The GPU box never builds.

### Recommended flow

```
 workstation ──sync──► CPU builder ──build_and_push──► private registry (GHCR / Docker Hub)
      │                                                     │
      └──────sync──► GPU box (A6000) ◄──pull_on_server──────┘   session.sh up|collect|showcase|down
```

| Step | Where | Command | Billed as |
|---|---|---|---|
| 1 | workstation | `HOST=builder scripts/sync_to_server.sh` (project + assets, ~0.5 GB) | — |
| 2 | CPU builder | `scripts/server_setup.sh --builder` (disk grow, Docker + buildx, overlay2, git-lfs, genie_sim @ `6ca11c7` + LFS) | CPU |
| 3 | CPU builder | `docker login ghcr.io` (once), then `NAMESPACE=<user> scripts/build_and_push.sh` | CPU |
| 4 | workstation | `scripts/sync_to_server.sh` (to the `paperspace` alias) | — |
| 5 | GPU box | `scripts/server_setup.sh` → `sudo reboot` → `scripts/server_setup.sh` (self-test) | GPU |
| 6 | GPU box | `docker login ghcr.io` (read-only token), `NAMESPACE=<user> scripts/pull_on_server.sh` | GPU |
| 7 | GPU box | `sudo scripts/idle_shutdown.sh install`, then `scripts/session.sh collect` | GPU |
| 8 | GPU box / workstation | `scripts/session.sh down --poweroff`; `scripts/sync_to_server.sh --pull-output` | — |

The builder can be a Paperspace CPU machine or any Linux box with Docker and ≥ 150–200 GB disk. This
workstation (48 GB free) cannot hold a build. No GPU is needed to build: nothing in either Dockerfile
touches CUDA, and the genie_sim/g2cell colcon builds are CPU-only.

**Registry.** `scripts/build_and_push.sh` pushes one repository, `<registry>/<namespace>/g2cell`, with
tag prefixes, so a Docker Hub free account (one private repo) also works:

| Tag | Content | Hash input (no git needed) |
|---|---|---|
| `geniesim-3.2.0-isaac6.0.1.0-<h>` | base: Isaac Sim 5.1 + pip Isaac Sim 6.0.1.0 + ROS 2 Jazzy | genie_sim commit, Isaac Sim 6 version, `docker/Dockerfile.geniesim` |
| `cell-<h>` | base + cell deps (dev workflow: workspaces built in the container) | base hash + `docker/Dockerfile.cell` |
| `baked-<h>-<w>`, `baked-<h>` | cell + both ROS workspaces prebuilt | + every file in `ros_ws/src` |
| `*-latest` | moving tags | — |

The hashes come from file contents (`scripts/lib/g2_images.sh`), so the builder and the GPU box compute
the same tags from the same synced files. `build_and_push.sh` skips a target whose tag is already in
the registry; `pull_on_server.sh` pulls the exact tag and falls back to `baked-<h>` when only
`ros_ws/src` changed. In that case `docker/cell.sh up` seeds the prebuilt genie_sim workspace and asks
for a `colcon build` of `/g2cell_ws` (~2 min). `--print-tags` shows the tags without building.

**Credentials.** Neither script stores any. Log in once per machine:

```bash
# GHCR: classic PAT; builder needs write:packages, the GPU box only read:packages
echo "$GHCR_PAT" | docker login ghcr.io -u <github-user> --password-stdin
# Docker Hub: access token, and create <user>/g2cell as a PRIVATE repo before the first push
docker login -u <hub-user>
```

`docker login` saves the token base64-encoded in `~/.docker/config.json` unless a credential helper is
set up. Use a read-only token on the GPU box and `docker logout ghcr.io` when the project ends. Keep the
package private: it contains the NVIDIA Isaac Sim base, which may not be redistributed publicly. New
GHCR packages are private by default; check `github.com/<user>?tab=packages` after the first push.

**Why the push is shaped this way**

- *buildx `docker-container` builder, pushed straight from its cache.* Nothing is loaded into the
  builder's image store, so the disk holds one copy, not two.
- *zstd layers* (`--compression zstd`; gzip is available). They unpack several times faster than gzip
  on the GPU box, and unpacking ~56 GB is most of the pull time. The base push also recompresses
  NVIDIA's own layers. Docker ≥ 23 pulls zstd.
- *Smaller base layers.* `docker/Dockerfile.geniesim` was one 25.7 GB pip layer for Isaac Sim 6 and
  one 9.5 GB layer for the Genie Sim deps. GHCR refuses layers over 10 GB, and a single blob that big
  is fragile to push. The pip install is now resolved once (`pip --dry-run --report`, same flags), then
  installed in `ISAACSIM_LAYERS=4` round-robin chunks with `--no-deps`. The deps install is one layer
  per interpreter. The installed files are the same. This is the only change from upstream besides
  the 6.0.1.0 pin.

### Fallback: build on the GPU box

When no builder or registry is available, run `scripts/server_setup.sh` and then
`BAKED=1 docker/build_all.sh` on the GPU box (about 1.5 h of billed GPU time). Or use
`docker save | zstd` / `docker load` from another machine (docs/PAPERSPACE.md).

### Docker daemon config (server and builder)

`scripts/server_setup.sh` merges this into `/etc/docker/daemon.json` before Docker first starts, and keeps
the keys that `nvidia-ctk` adds:

```json
{
  "features": {"containerd-snapshotter": false},
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "3"},
  "max-concurrent-downloads": 6,
  "max-concurrent-uploads": 4,
  "default-runtime": "nvidia",
  "runtimes": {"nvidia": {"args": [], "path": "nvidia-container-runtime"}}
}
```

- *`containerd-snapshotter: false`.* Uses the classic overlay2 store. The containerd store (the default
  on fresh Docker 29 installs) keeps both the packed and the unpacked layers: locally `docker image ls`
  shows 82.3 GB of disk for 26.4 GB of content. Set it before the first pull; images pulled under the
  other store are not visible after the switch.
- *Log rotation.* Stops a chatty container from filling the disk.
- *`default-runtime: nvidia`.* Set by `nvidia-ctk runtime configure --set-as-default`. `cell.sh` still
  passes `--gpus all`.

### What `scripts/server_setup.sh` does (idempotent; rerun after the reboot)

1. Grows the root filesystem when the disk was resized. It finds the root LV, its PV and that PV's
   partition (`/dev/xvda` 3 on Paperspace; override with `ROOT_DISK`/`ROOT_PART`). It runs `growpart`
   only if `growpart --dry-run` reports room, then `pvresize`, then `lvextend -l +100%FREE` only if the
   VG has free extents. It runs `resize2fs` only if the filesystem is > 2 GB smaller than its LV. With
   nothing to grow it changes nothing. Ubuntu's installer gives the root LV only part of the VG, so the
   `lvextend` step can help even without a disk resize.
2. Installs the base packages, then writes the Docker daemon config. Docker CE comes from Docker's apt
   repo, with buildx.
3. GPU mode: installs nvidia-container-toolkit and registers the nvidia runtime as the default.
4. GPU mode: creates the persistent cache dirs under `CACHE_ROOT` (default `~/docker/isaac-sim`, the
   same as `cell.sh`). The Kit/OV shader cache lives there, so only the very first sim launch on a
   machine compiles shaders. The cache survives container restarts and machine stop/start.
5. Clones genie_sim at the pinned commit and pulls LFS, via `bootstrap_host.sh` with
   `NO_GPU=1 SKIP_DOCKER=1 SKIP_ASSETS=1`. The assets come from the workstation sync.
6. GPU mode: installs `nvidia-driver-580` via `ubuntu-drivers install nvidia:580`, which uses prebuilt
   signed modules when Ubuntu has them and DKMS otherwise; it falls back to plain apt, or the
   graphics-drivers PPA with `DRIVER_PPA=1`. It then **exits with code 3 and asks for a reboot**
   (`REBOOT=1` reboots by itself). On the rerun it checks the version: it warns if the version is not the
   workstation's 580.178.04 and fails below 580.
7. GPU mode self-test: `docker run --rm --gpus all ubuntu:22.04 nvidia-smi`.

### Cost controls

**Idle power-off** (`scripts/idle_shutdown.sh`, a systemd timer, every 5 min). Install it with
`sudo scripts/idle_shutdown.sh install [--minutes 30] [--gpu-threshold 10] [--ignore-ssh]`. It
powers the machine off after N minutes (default 30) in which **all** of these held:

- no GPU above the threshold;
- no established SSH connection, Foxglove tunnels included;
- no sim process in the `g2cell` container (checked with `docker top`, without exec);
- no docker pull, push, build, load or save running.

Commands:

- `status`: shows the current verdict.
- `pause` / `resume`: hold it off until the next reboot, for unattended jobs.
- `disable` / `enable`: stop or start the timer.
- `uninstall`: removes it completely.

The log is in `journalctl -u g2cell-idle-shutdown`. The SSH test means an open laptop session keeps
the machine on. Close SSH when you leave, or install with `--ignore-ssh`.

**Paperspace auto-shutdown** is the backstop for everything else, for example a sim left running. Set it
on the machine's page in the Paperspace console under *Settings → Auto-Shutdown* (timeouts such as 1 h,
8 h, 1 day, 1 week). Use 8 h for working sessions. *Unverified here:* whether the timer counts from
start or from idleness on Linux machines. Treat it as wall-clock time. Also check, after the first
idle power-off, that the console shows the machine as **Off**, which stops compute billing. The disk
is billed while the machine is off either way.

**`scripts/session.sh`** starts only what the task needs:

| Command | Starts |
|---|---|
| `up` | the container; the ROS workspaces are seeded from the baked image or built if missing. No sim. Use for editing and offline tests |
| `collect [scene]` | `up` + `stage1_up.sh` with `PROFILE=collect` on `scene_g2cell.yaml` |
| `showcase [--mobile] [scene]` | `up` + `PROFILE=showcase`; `--mobile`: `mobile_up.sh` with `WITH_MOVEIT=1` (drive-in / packing demo) |
| `down [--poweroff]` | stops the tmux stack and the container; `--poweroff` also powers the machine off |
| `status` | uptime (= billed time), GPU use, disk, image, container, tmux windows, idle-shutdown verdict |

`PROFILE` is passed through unchanged to `ros_ws/scripts/stage1_up.sh`, which decides what each render
profile starts. From the workstation: `SESSION_HOST=paperspace scripts/session.sh status`.

### Image size impact (estimates; the new cell image has not been built yet)

| Part | Uncompressed | Compressed |
|---|---|---|
| base (unchanged contents; 4+2 layers instead of 1+1) | ~56 GB | ~26 GB |
| cell apt: navigation2 + nav2-bringup (~150–250 MB with behaviortree/costmap plugins), slam-toolbox (~60–100 MB with ceres/suitesparse), foxglove-bridge (~15 MB), pointcloud-to-laserscan, spacenav, tmux, mcap plugin (few MB each) | +0.25–0.45 GB | +0.1–0.15 GB |
| pip h5py (bundles libhdf5) | ~+15 MB | ~+6 MB |
| baked: genie_sim devel (129 MB) + devel_build without `*.o` (~270 MB) + g2cell install (12 MB) | ~+0.4 GB | ~+0.12 GB |

Merging the apt layers and cleaning the lists saves roughly 50–100 MB of apt list files per extra
layer. The cell image adds under 1 % to the pull either way.

**Jazzy package names.** These were already apt-installed and running in the local container:
`ros-jazzy-foxglove-bridge`, `ros-jazzy-slam-toolbox` (2.8.5), `ros-jazzy-pointcloud-to-laserscan`
(2.0.2), `ros-jazzy-navigation2` and `ros-jazzy-nav2-bringup` (1.3.13). Not apt-queried here but
standard: `ros-jazzy-spacenav` (joystick_drivers; g2cell_teleop's README names it),
`ros-jazzy-rosbag2-storage-mcap` (part of rosbag2 since Iron, normally already in `ros-jazzy-desktop`)
and `tmux` (noble). If one is missing, the apt layer fails early on the builder, not on the GPU box.

### First GPU session: hour-by-hour plan

Phase A, CPU builder (not GPU-billed; run it the day before or while doing other work):

| Time | Machine | Step | Duration |
|---|---|---|---|
| 0:00 | Paperspace console | create a CPU machine: Ubuntu 22.04, ≥ 8 vCPU, 200 GB disk | 5 min |
| 0:05 | workstation | `HOST=builder scripts/sync_to_server.sh` | 2 min |
| 0:07 | builder | `scripts/server_setup.sh --builder` | 10–15 min |
| 0:20 | builder | `docker login`; `NAMESPACE=<user> scripts/build_and_push.sh` (base: pull 7.6 GB NVIDIA base, apt + ROS, pip Isaac Sim 6 ~10–15 GB of wheels, ~56 GB of layers; then cell; then baked = 2 colcon builds) | 1.5–2.5 h, mostly network-bound |
| ~2:30 | builder | `docker buildx prune -af`; stop or delete the builder | 2 min |

Phase B, GPU box (billed from power-on):

| Time | Machine | Step | Duration |
|---|---|---|---|
| 0:00 | console | start the A6000 machine (disk already resized to ~200 GB) | 1–2 min |
| 0:02 | workstation | `scripts/sync_to_server.sh` | 1–2 min |
| 0:04 | GPU box | `scripts/server_setup.sh`: disk grow, Docker, toolkit, genie_sim + LFS, driver 580 → exit 3 | 12–18 min |
| 0:20 | GPU box | `sudo reboot`, then rerun `scripts/server_setup.sh` (self-test) | 4–5 min |
| 0:25 | GPU box | `docker login`; `scripts/pull_on_server.sh` (download ~26 GB, unpack ~56 GB); meanwhile `sudo scripts/idle_shutdown.sh install` | 15–25 min |
| 0:45 | GPU box | `scripts/session.sh collect`: container up, workspaces seeded, first launch compiles shaders (cached afterwards) | 8–12 min |
| 0:55 | GPU box + laptop | Stage 1 check on Paperspace (open in PLAN): teleop, Foxglove app over `ssh -L 8765:...`, one recorded episode | 30 min |
| 1:25 | GPU box | real work: data collection / `session.sh showcase` renders | as planned |
| end | both | `scripts/session.sh down --poweroff`; `scripts/sync_to_server.sh --pull-output` | 2 min |

Later sessions: start → `scripts/session.sh collect`. The image, genie_sim and the shader cache persist
on the disk, so the sim is up in ~3–5 min. When a Dockerfile changes, rerun `build_and_push.sh` on the
builder and `pull_on_server.sh` on the GPU box. Only the changed layers download: a `Dockerfile.cell`
change is ~0.2 GB.

### Estimated GPU-hours saved

| | Build on GPU box (fallback) | Build off-box (recommended) |
|---|---|---|
| setup + driver + reboot | ~0.4 h | ~0.4 h |
| image | build ~1.2–1.6 h (local build took > 1 h), plus up to ~10 min of colcon in the container | pull ~0.3–0.4 h, no colcon |
| first sim launch | ~0.15 h | ~0.15 h |
| **GPU time before the first useful minute** | **~1.9–2.3 h** | **~0.8–1.0 h** |

That saves **≈ 1.1–1.4 GPU-hours on the first session**, or about $2–3 at an A6000 rate of ~$1.9/h
(check the current price). A later base-image change saves ~1.2 h per rebuild; a cell-only change saves
~10 min. The larger saving is the idle power-off: a machine forgotten overnight costs 12–16 h (~$25–30).
<!-- END section: Build & deploy -->

> **Note (main session, 2026-10-11):** `docker/Dockerfile.geniesim` is kept as the **proven** single-layer
> build (upstream `Dockerfile.5.1` + the Isaac Sim 6.0.1.0 pin; it built successfully on the workstation).
> The layer-split variant needed for registries with a 10 GB per-layer limit (GHCR) is
> `docker/Dockerfile.geniesim.split`. It is used only by `scripts/build_and_push.sh` and has **not been built yet**.
> Direct builds on the GPU box (`docker/build_geniesim.sh`, `build_all.sh`) use the proven file.

<!-- BEGIN section: Data & training -->
## Data & training

Goal: small recordings, fast bag → HDF5 conversion, and a training loop that keeps the A6000 busy.
All numbers below were measured on this workstation with the CPU only (no sim, no GPU), on the real
Stage 1 episode (`output/episodes/stage1_teleop`, 19.7 s). Timings were pinned to 8 cores
(`taskset -c 0-7`) to match the 8-vCPU server.

### Recording size (`record_episode.sh`)

Bytes per second per topic, from the existing bags (all three show the same rates):

| topic | before | after (default) | how |
|---|---|---|---|
| `/genie_sim/free_camera_rgb` 1280×720 rgba8, 13.5 Hz | 49.7 MB/s | 0 (off) | `--free-cam` to record it: 1.49 MB/s as JPEG |
| `/genie_sim/head_front_camera_rgb` 640×400 rgba8, 13.5 Hz | 13.8 MB/s | 0.66 MB/s | `.../image_raw/compressed` (JPEG q95; mean abs error 0.44/255, p99 3/255) |
| `/genie_sim/head_front_camera_depth` 640×400 32FC1, 13.5 Hz | 13.8 MB/s | 0.33 MB/s | `.../compressedDepth` (PNG of quantised inverse depth, max error 0.71 mm at ≤ 2.7 m) re-published at ≤ 5 Hz (4.3 Hz measured) by `throttle_relay` |
| `/tf` 212 Hz | 0.61 MB/s | ~0.2 MB/s | MCAP chunk zstd (lossless) |
| `/joint_states` 97 Hz, `/joint_command` 48 Hz | 0.21 + 0.03 MB/s | ~0.10 + 0.015 MB/s | full rate, chunk zstd |
| **total on disk** | **78.2 MB/s (1.54 GB / 20 s)** | **1.09 MB/s (21.5 MB / 20 s), 72× smaller** | with `--free-cam`: 2.46 MB/s |

- **Topics.** Genie Sim's render node publishes every camera through image_transport, so `compressed`,
  `compressedDepth`, `theora` and `zstd` all exist (`output/foxglove.log`). JPEG via `compressed` is
  already used by `record_cams`. Raw float depth barely compresses with zstd (13.8 → 11.4 MB/s), so depth
  goes through `compressedDepth`.
- **Flags.** `--rgb compressed|zstd|raw`, `--depth compressedDepth|zstd|raw|off`, `--depth-rate HZ|full`,
  `--free-cam`, `--compression zstd_fast|zstd_small|none` (MCAP `--storage-preset-profile`). `--full-fidelity`
  restores the raw streams at full rate (still chunk-compressed, which is lossless: 78 → ~25 MB/s).
  `--dry-run` prints the topic list and the command.
- **How the "after" column was measured.** The recorded raw frames were re-encoded offline with Pillow
  (JPEG q95, PNG level 9, the plugins' defaults) and the MCAP python writer (zstd chunks). The rates were
  then measured on that bag. OpenCV's encoder and rosbag2's `zstd_fast` will differ by a few percent.
  **Not yet recorded live from the sim.** Check the first live episode with `ros2 bag info`.

### Conversion (`bag_to_hdf5`)

Profile of the old converter on the Stage 1 bag: 14.7 s, of which 82 % was HDF5's single-threaded gzip
filter on the image writes (11.7 s), 1.1 s MCAP read + CDR decode, and 0.9 s RGBA → RGB.

Optimised: streaming writes (no per-frame resize, geometric growth), one frame per chunk, and frames
decoded and shuffle+deflated in a process pool. The main process writes the ready chunks with
`H5Dwrite_chunk`, so the files are ordinary gzip/shuffle datasets. Peak RSS is ~150 MB.

| Stage 1 bag (raw topics, 1.54 GB) | time | HDF5 size |
|---|---|---|
| before: serial, gzip 4 + shuffle | 14.7 s | 317.4 MB |
| **after: 7 workers, gzip 4 (default)** | **3.3 s (4.4×)** | **317.4 MB (every dataset equal)** |
| after: gzip 1 / gzip 2 / gzip 3 / gzip 6 | 2.6 / 3.0 / 3.4 / 7.1 s | 367 / 355 / 334 / 294 MB |
| after: lzf (HDF5 filter, main thread) | 5.5 s | 522 MB |
| after: none | 2.5 s | 1214 MB |
| after: gzip 1, workers 0 / 3 | 10.8 / 4.3 s | 367 MB |
| **default-profile bag (JPEG + compressedDepth 5 Hz, 21.5 MB)** | **0.9 s** (3.3 s serial) | **71 MB** |

Serial per-stream filter benchmark (h5py, 266 frames each; size / write time):

| filter | head RGB 640×400 | depth 32FC1 | free RGB 1280×720 |
|---|---|---|---|
| none | 204 MB / 0.07 s | 272 MB / 0.25 s | 736 MB / 0.5 s |
| lzf (+shuffle on depth) | 74 MB / 0.6 s | 120 MB / 0.9 s | 326 MB / 2.4 s |
| gzip 1 (+shuffle) | 56 MB / 1.2 s | 107 MB / 2.8 s | 203 MB / 5.1 s |
| gzip 6 | 45 MB / 5.2 s | 198 MB / 7.1 s (no shuffle) | 149 MB / 24.9 s |

The choice: gzip level 4. LZF is about 2× faster than gzip 1 serially, but it runs inside HDF5's
single-threaded filter. gzip parallelises across the workers. With workers, gzip 4 is faster than LZF
(3.3 s against 5.5 s) and 40 % smaller. JPEG-sourced frames are noisier and compress less (71 MB for 20 s of head RGB + 5 Hz
depth), which is why the HDF5 is larger than the bag. The HDF5 keeps decoded frames so the schema and
every reader stay unchanged.

### Training throughput (`prepare_cache.py`, `CachedEpisodeDataset`, `train.py`)

`prepare_cache` decodes, resizes (the same antialiased bilinear as before) and rounds each distinct
aligned frame once. It writes them into `<cache>/<episode>-<hash>/<cam>.npy` (uint8 [U,3,h,w], memory-
mapped) plus a step → frame LUT, `lowdim.npz` and per-episode moments. The train split's
normalisation stats are combined exactly from those moments; a test checks them against
`compute_norm_stats`. The hash covers path, size, mtime and image size, so a re-converted episode is
re-cached automatically. Stage 1 at 128×160: 241 frames, 15 MB, 1.3 s with 8 workers.

DataLoader throughput, CPU, 8 cores, batch 64, 128×160, K = 20 (samples/s at 0 / 4 / 8 workers):

| dataset | start-up | samples/s |
|---|---|---|
| `EpisodeDataset(cache_images=False)`: per-sample HDF5 read + gzip inflate + resize | 0 s | 413 / 989 / 1 285 |
| `EpisodeDataset` (old default): float32 frames decoded into RAM at every start | 0.7 s and ~60 MB per episode, per process | ~19 000 (8 000–19 000 across runs) |
| **`CachedEpisodeDataset`**: uint8 memmap | **0 s; 15 MB per episode, shared** | **33 000–41 000** |

The cache gives 30–90× the per-sample HDF5 path and 2× the RAM path. It needs no decode at start-up and
4× less memory, so it scales to hundreds of episodes on a 44 GB box: the RAM path would need ~24 GB per
process for 200 episodes at 160×256. Images now cross the bus as uint8, and the policy converts them on
the device.

Training step on this CPU (ACT-lite, small CNN, batch 32): ~130–175 samples/s with or without the
cache. The CPU is compute-bound, so the loader gain does not show here. `--compile` gave 252 against 175
samples/s at d_model 64 on CPU. **Expected on the A6000, not measured:** ResNet-18 at 160×256 in bf16
should train at a few thousand samples/s. That is above what the old per-sample path delivers
(≤ 1.3 k/s, so the GPU would sit mostly idle) and well below the cache path (~35 k/s on 8 cores). Run
`--dry-run` on the server: it flags `loader_bound` when the loader is under 1.2× the train rate.

Training loop changes (`train.py`, all flags also in the YAML config):
- `--device auto`;
- `--amp auto`: bf16 autocast, fp16 + GradScaler where bf16 is unsupported, fp32 on CPU;
- `--compile` (optional), `--channels-last`, `--grad-accum N`;
- DataLoader `num_workers` / `pin_memory` / `persistent_workers` / `prefetch_factor`;
- the loss is kept on the device, so there is no per-step `.item()` sync;
- `last.pt` every `--ckpt-minutes` (atomic write, with optimizer, scaler and RNG states), and on
  SIGTERM/SIGINT;
- `--resume auto`;
- a fixed val subset, `best.pt`, and early stopping (`--early-stop-patience`);
- `samples_per_s` in `loss.csv`.

### Server command

```bash
cd g2_packoff_cell
python -m g2cell_learning.train --config ros_ws/src/g2cell_learning/configs/server_a6000.yaml --dry-run \
    --price-per-hour 1.9          # samples/s, est. epoch + run time and cost, peak GPU memory
python -m g2cell_learning.train --config ros_ws/src/g2cell_learning/configs/server_a6000.yaml
```

Defaults in `server_a6000.yaml`:
- ACT-lite on ResNet-18 at 160×256, batch 256, lr 3e-4, 30k steps, bf16, channels_last;
- cache in `output/cache`; 6 loader workers with pinned memory, persistent workers and prefetch 4;
- `last.pt` every 10 min and `--resume auto`, so a preempted run loses ≤ 10 min when restarted with
  the same command;
- early stop after 8 evals (4 000 steps) without val improvement.
### Pilot dataset (2026-10-11, live, workstation)

`pilot_v1`: 20 datagen episodes, 20/20 `success` (scored from the relay's bag pose), mean wall 42.4 s per episode
(reset + homing 5.7 s, task 32.7 s incl. planning), bags 97 MB/episode (head JPEG 30 Hz + depth 5 Hz + tf), HDF5
~410 MB/episode (decoded frames, gzip 4; 45 s to convert all 20 with 8 workers). Frame cache 128×160: 21 318 frames,
1.3 GB, 19 s. CPU train (ACT-lite small, batch 16): 150–158 samples/s. Details: `output/readiness/READINESS.md`.
<!-- END section: Data & training -->

<!-- BEGIN section: Motion planning -->
## Motion planning

Goal: arm planning for the packing demo and the PickBag tasks should be **deterministic** and take
**seconds, not minutes**, so that no billed GPU time goes to IK retries. All numbers were measured
offline against a standalone `move_group` (KDL, timeout 0.1 s) in the g2cell container on
ROS_DOMAIN_ID 42, without the sim. The replays (`tools/perf/ik_replay.py`, `tools/perf/task_replay.py`)
run the demo's and motion_server's own planning code. Breakdown before the change:
`output/perf/ik_baseline.md`.

### Root cause

Most of the time went into **failed /compute_ik calls, each burning KDL's full 0.1 s timeout**. In the
demo, 85 % of 2.5-3.7 k calls failed. `IkHelper.solve` followed a failed seeded attempt with up to 11
random-jitter attempts and no early exit, using the unseeded global RNG. The demo also re-solved the place
chains for every pick candidate, and it fixture-checked the whole 12-waypoint path for every (pick,
place) pair at 41 ms per sample. Random jitter made the search order, and so its cost, change from run
to run (278-455 s in the replay, 4-8 min live).

### What changed

| Change | Where | Effect |
|---|---|---|
| **Local IK first**: batched damped-least-squares on the staged `robot.urdf` (numpy only, no pinocchio needed in the container). It enforces the genie KDL plugin's coupled joint-6/7 constraints (`coupled_constraints.yaml`, margin 0.1); without them, 25 % of local answers were rejected by MoveIt. Each answer is then **confirmed by one /compute_ik call seeded with it**. KDL converges at once (~1 ms), so the result is MoveIt's. | `g2cell_motion/kinematics.py`, `IkHelper.init_local_ik/solve` | 38/38 and 66/66 confirmed in the demo replay; local FK = /compute_fk to 7e-16 m |
| **Deterministic seeds**: seed = previous solution / home; restarts come from a fixed-seed RNG (`np.random.default_rng(0)`, `random.Random(0)`), never the global one | `kinematics.ArmIk`, `planner.ik_seed_sequence` | identical plans run to run |
| **Bounded budgets, early exit**: MoveIt fallback limited to `ik_fallback_attempts` per waypoint, first success wins. `attempts=0` really means no calls. `PlanBudget(max_solves, max_seconds)` in `plan_chain_waist` stops with reason `budget`. The demo's 3x retry loop is now `plan_retries: 1` plus one bounded MoveIt pass only if the local pass finds nothing. | `ik_client.py`, `planner.py`, `packing_demo.plan_arm` | no 0.1 s timeouts in the demo |
| **Each IK chain solved once** (memoised lazy chains) and **pruned search**: the path is checked as pick part A, waist swing B and place part C. The check is a minimum over segment pairs, so A+B+C gives exactly the old result. A failing pick part skips all place work, and the first plan in the old order (bag, wp, o1, wd, o2) is still the one returned. A property test checks this against brute force on 200 random problems. | `g2cell_demo/search.py`, `packing_demo.plan_left` | 7 path checks instead of 67-171 |
| **Cheaper clearance**: the URDF is parsed once, only the arm and head chains are evaluated, a bounding-sphere broad phase skips (link, box) pairs (the box SDF is 1-Lipschitz, so the result stays exact), the head check reuses the same kinematics pass, and the path check stops at the first sample below the margin | `g2cell_demo/env_check.py` | 41 → 0.34 ms per sample, 595 → 6.9 ms per path; same values (tests compare against the old code, real meshes in the container) |
| **Local FK for head clearance** in motion_server instead of one /compute_fk per sample | `motion_server._clearance` | |
| **Plan cache** (below) | `g2cell_motion/plan_cache.py` | repeated runs ~0.1 s |

Unchanged: waist-first policy, orientation order, twist/flip filters, 0.20 m head clearance, fixture
margin 0.015 m and idle-arm check.

### Results

Packing demo planning (left: bag → carton, right: tote L1_1), same replay, same machine:

| | before (3 runs) | after, cold (5 runs) | after, plan cache hit |
|---|---|---|---|
| left arm | 245.9-424.5 s | 4.37-4.63 s | 0.06 s |
| right arm | 26.5-32.2 s | 1.18-1.28 s | 0.04 s |
| **total per demo** | **278-455 s** | **5.6-5.9 s (~50-80x)** | **0.1 s** |
| /compute_ik calls | 2.6-3.8 k (85 % failing at 0.1 s) | 104 (all confirmations, ~1 ms) | 0 |
| path checks | 92-197 | 19 | 2 (re-validation) |
| success | 3/3 | 5/5 | 5/5 |
| identical plan every run | final plan yes, search effort no | **yes (5/5 bit-identical paths)** | yes |

The cold plan differs from the old one: left waist +69/0 deg, orientation (0,0)/(-90,40), fixture
clearance 0.068 m (old: 0.018 m); right orientation (45,40) (old: (180,0)). The search order is the
same; the old randomised KDL failed or twisted on these earlier candidates, and the local IK solves them.
Every plan still passes the same fixture, head, idle-arm and twist checks. **The new plans have not
been executed in the sim yet**, so check them on the next sim run.

PickBag tasks (motion_server `_plan`, left arm, Bag_01..03 → carton, 12-18 goals per episode):

| | before (3 episodes) | after, cold (5 episodes) | after, cache (episodes 2-3) |
|---|---|---|---|
| planning per episode | 75.6-77.1 s | 33.7-36.7 s | 0.0 s |
| goals planned OK | 9/12 | 13/18 | 13/18 |
| bags that completed every stage | 1 (Bag_01) | 1 (Bag_01); Bag_02 now passes approach and lift but fails the grasp | same |
| distinct outcomes across episodes | 3 of 3 | 1 of 5 | 1 |

The task episode time is dominated by goals that fail (Bag_03 approach, Bag_02 transfer). With
`ik_fallback_attempts: 0` the same 13/18 plan in 18 s; the default is 1 as insurance. Deterministic
failures are cached too, so a repeated episode plans in ~0 s.

### Fixture check (2026-10-11, PLAN_VERSION 3)

The first live datagen episode showed the approach from home (hand 0.10 m over the pack carton, waist turning
toward the table) sweeping the gripper through the carton's +Y wall: the carton was thrown 0.5 m. The planner only
checked head clearance. Now every sampled pose along a path also keeps the arm links and fingertips
`fixture_margin` (0.04 m) from the `cell_colliders.json` boxes within 1.6 m (bags excluded), and PickBag first moves
to the named pose `clear` (gripper-down, (0.50, 0.25, 1.30), waist held) when it starts from home. Verified live:
the carton stayed at (0.6, 0, 0.82) in the probe and the 20 pilot episodes. Bumping PLAN_VERSION drops old cached
plans (the first live run had replayed pre-fix plans from the cache). Known residue: 12/20 pilot episodes fail the
retract planning (head clearance 0.20 m) after the release; recovery still brings the arm home.

### Plan cache

- `packing_demo` parameter `plan_cache` (default `/g2cell_output/plan_cache/packing_demo.json`, `""` =
  off). Key: targets, rest/head state, orientation list, margins, waist axis and planner version. The
  base pose is stored with the plan. Within 1 mm / 0.1 deg of the cached base the path is reused after
  one fixture/head re-check (`hit`). Further away, only the cached (waist, orientation) choice is
  re-solved at the new base (`hint`). If that fails, the full search runs. `result.planning` in
  `packing_demo.json` records which path was taken and the timings.
- motion_server parameter `plan_cache` (default `/g2cell_output/plan_cache/motion_server.json`; set in
  `g2cell_bringup/config/cell_params.yaml`). Key: goal fields + start state rounded to 1e-4 rad +
  planner parameters. It stores successes and exhausted searches, but not budget stops or missing
  services. The tasks get it through MoveArm.
- **Invalidation**: each cache file carries a layout key = sha256 of the module-level UPPERCASE constants
  of `assets/cell/build.py` (read via the AST, so comments do not count) plus the content of the exported
  `cell_colliders.json`. A file written under another key or `PLAN_VERSION` is dropped on load. In the
  container build.py is not mounted, so the colliders hash, which build.py rewrites, does the
  invalidation there. Set `layout_file` or `G2CELL_BUILD_PY` to include build.py.

### Parameters

`packing_demo`: `local_ik` (true), `ik_confirm` (true), `ik_fallback_attempts` (3, used only in the
fallback pass), `plan_retries` (1), `plan_cache`, `layout_file`.
`motion_server`: `local_ik`, `ik_confirm`, `ik_fallback_attempts` (1), `plan_max_ik_solves` (600),
`plan_max_seconds` (60), `plan_cache`, `layout_file`, `urdf`, `colliders`.

### Tests

`g2cell_motion` 80 → 102: `test_kinematics` (FK, IK round trip, bit-identical repeats, coupled
constraints), `test_budget` (seed determinism, budget/early exit, zero attempts, same inputs → same
plan), `test_plan_cache` (key stability, layout-hash invalidation, version bump, unwritable/corrupt
files). `g2cell_demo` 13 → 23: `test_search` (pruned search = brute force), `test_env_check_fast`
(fast checker = old checker). tasks 62, teleop 84 and bringup 7 are unchanged and pass.

### Reproduce

```bash
# own tmux session, private domain; never on the sim's domain
docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash && ros2 launch genie_sim_moveit wbc.launch.py arm:=crs gripper:=omnipicker use_ros2_control:=false'
docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && ros2 run tf2_ros static_transform_publisher --frame-id odom --child-frame-id base_link'
cp tools/perf/*.py output/perf/
docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/ik_replay.py --mode after --runs 5 --tag after_cold'
#   --mode baseline needs the pre-change packages; --cache FILE for cache runs; task_replay.py likewise
```
### PickBag transfer fix (2026-10-11)

Symptom (A6000, `output/a6000/pickbag.log`): Bag_01 ran approach, grasp and lift, then
`transfer: planning failed: ik_failed (code 2)`, recover ran, and the outcome was `plan_failed` after 29.3 s.
On the workstation only Bag_01 had ever completed every stage. Reproduced offline with
`tools/perf/pickbag_replay.py`. The replay runs the real `PickPlaceMachine` and `MotionRunner` and
motion_server's own `_plan`, starting from the scene's init state (head 40/28 deg), against a standalone
move_group on domain 42. Its failure message and stage sequence match the live run exactly.

**Root cause.** Found with `tools/perf/transfer_diag.py`, `column_diag.py` and `transfer_bag02_diag.py`
(IK per waypoint x waist yaw x 24 orientations).

1. **Waist sequencing at transfer.** The pick happens with the torso turned toward the table (Bag_01: waist
   78.5 deg). The transfer goal (via (0.325, 0.31, 1.19) and then 0.17 m above the carton) used
   `waist_mode: absolute 0` for **every** point. With the torso straight, the via point is beside the
   shoulder and has **0/24** IK orientations at waist 0, against 6 to 12 at 30 to 60 deg. Reach is the only
   reason: no solution was rejected by the twist filter, and the coupled joint-6/7 rows do not come into it.
   The carton point itself solves at waist 0 (7/24). The carton rim height (1.07 now, 1.044 before) does not
   matter: 7 vs 8 orientations.
2. **Pick column not planned as a unit (Bag_02).** The approach chose its orientation and waist (39 deg)
   alone. The descent with that orientation then had no IK. Each move also re-ran waist_first, so the torso
   swung 78 -> 48 -> 78 deg while the hand was at the bag. Bag_02's column only solves tilted (yaw 90, tilt
   20, elbow up) at waist 15 to 50 deg. The fallback waist profiles stepped 30 deg and tried only the 8
   untilted orientations, so they never found it.
3. **Bag_02 transfer.** From that elbow-up posture the transfer at waist 0 passes **0.198 m** from the head
   (limit 0.20). With the waist ending at 30 deg it clears.
4. **Non-determinism.** With `ik_fallback_attempts: 1`, every failed local IK made one randomised 0.1 s KDL
   call. In 5 replays Bag_02 took two different transfer paths (21 s vs 31 s).
5. **Not the cause:** plan-cache staleness (the replay starts with an empty cache and fails the same way);
   tasks.yaml Layout v2 drop points (correct). `PLAN_VERSION` was still bumped to 2, so old caches are dropped.
6. **Recovery.** In the replay, `home` from the lift pose plans and runs, so recovery reached home there.
   The live A6000 timing (3.15 s from the last IK failure to the result, about gripper 0.8 s + home 2.3 s)
   suggests home was also commanded live. The head frame showing the extended arm cannot be tied to a
   time, so the cause of the extended arm is **unverified**. Three real hazards were fixed:
   - recovery stopped before `home` when `open gripper` failed. A BUSY reply does that, and BUSY follows a
     client timeout: `motion_timeout` 60 s = `plan_max_seconds` 60 s, and the canceled goal keeps planning
     on the server.
   - `home` was a single joint-space swing from wherever the hand was, with no straight-up exit first.
   - there was no fallback when `home` was refused.

**Fix.**

| Change | Where |
|---|---|
| `absolute` waist with via points: the via yaws turn evenly from the current waist to the target (Bag_01 via at 39 deg), then fallbacks all-at-target and turn-last | `g2cell_motion/planner.waist_profiles` |
| MoveArm `lookahead` points: the moves that come next, planned with the same orientation at the final waist yaw and head-checked, but not executed. Failure reason `lookahead_ik_failed` | `MoveArm.action`, `planner.plan_chain_waist`, `motion_server`, `client.move_tool` |
| Fallback waist profiles: 15 deg steps, all 24 orientations (`waist_fallback_step_deg`, `waist_fallback_orients`) | `motion_server`, `cell_params.yaml` |
| `ik_fallback_attempts` 1 -> 0 (local IK + MoveIt confirmation only; deterministic, faster) | `motion_server`, `cell_params.yaml` |
| Approach plans with lookahead [grasp, lift]. Grasp, lift, place, retract and back-off hold the waist (`fixed`). Transfer has lookahead [release, retract] and tries: pick orientation -> searched -> `waist_first` | `g2cell_tasks/runner.py` (pure; moved out of `pick_bag_server`) |
| Recover: open (BUSY/no-answer retried) -> 8 cm straight up from the last pose (tolerated) -> home; if refused, arm home with the waist held, then the waist. `ok` only at home | `runner.MotionRunner.recover` |
| `motion_timeout` 60 -> 120 s (> planning budget + motion) | `pick_bag_server`, `cell_params.yaml` |
| Smoke test runs motion_server with `local_ik:=false`, `plan_cache:=''` (the fake /compute_ik cannot confirm URDF IK; never write the real cache) | `ros_ws/scripts/offline_smoke.py` |

**Before / after** (offline replay, same move_group, empty plan cache per bag; `output/perf/pickbag_replay_{before1,after}.json`):

| bag | before (live config) | after, 5 runs |
|---|---|---|
| Bag_01 | `plan_failed` at transfer (ik_failed), 9.6 s planning | **motion complete** (`unverified`, no scorer) 5/5, 3.76-3.83 s, ends at home |
| Bag_02 | `plan_failed` at grasp (descent ik_failed), 7.1 s | **motion complete** 5/5, 15.0-15.3 s (pick at waist 54 deg, place at waist 30 deg), ends at home |
| Bag_03 (Y 0.90) | `unreachable` in 0.0 s (reach table, no planning) | `unreachable` in 0.0 s, unchanged |
| identical goal sequence across runs | yes (1 run) | **yes, 5/5 for every bag** (with fallback 1: no, Bag_02 differed) |

Recovery check (`pickbag_replay_recover_unreachable_crate.json`): the carton was moved out of reach
(`crate_xy` 1.30). Transfer then fails after its 3 tries (19 s / 25 s per bag). Recovery opens the
gripper, rises 8 cm from the lift pose and goes home: final arm = home and waist 0 for both bags.
The offline smoke test (fake robot) passes 14/14. Tests: g2cell_motion 102 -> 108, g2cell_tasks 62 -> 77
(`test_runner.py`: stage goals, transfer fallbacks, recovery incl. BUSY and the arm-first home); the demo,
teleop, bringup, recorder and learning suites are unchanged and pass.

Not yet run in the sim. The replay jumps to each planned end state and checks the head only, not the
fixtures. The motion_server path check is head clearance only. Bag_01's via (X 0.325) is 0.075 m from the pack
carton's -X wall in plan, so watch the swept path on the next sim run.

```bash
cp tools/perf/{pickbag_replay,transfer_diag,column_diag}.py output/perf/
docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/pickbag_replay.py --runs 5 --tag after'
```
<!-- END section: Motion planning -->

<!-- BEGIN section: Sim runtime -->
## Sim runtime

Goal: more useful sim output per billed GPU-hour. That means fewer wasted renders, a physics loop that
holds 100 Hz, and no shader recompiles after restarts. **Measured** below means measured on the
workstation: RTX 3070 8 GB, i9-14900K (8 P-cores + 16 E-cores), scene `scene_g2cell.yaml` (pinned cell),
`NO_MOVEIT=1`. **Extrapolated** means not run on an A6000. Raw data is in `output/perf/` (`baseline.json`,
`baseline.md`, `profiles.json`, `exp_*.json`, stills in `output/perf/stills/`). Every number was produced
the same way: `python3 tools/sim_perf.py --label X [--watch-startup]`, a 30 s window with the sim already up.

### Profiles

Pick one with `PROFILE=...` on `stage1_up.sh` or `mobile_up.sh`; `scripts/session.sh` passes it through.
Without `PROFILE` you get the old upstream render path, unchanged.

| Profile | Renders | Use for |
|---|---|---|
| `collect` | head RGB + depth 640x400 at **30 Hz**, no FreeCam | teleop data collection and policy evaluation (same images, and the recorder's `/aligned` group is 30 Hz); SLAM on the mobile scene |
| `teleop` | head RGB + depth 640x400 at **15 Hz**, no FreeCam | driving or debugging without recording; lowest GPU load |
| `showcase` | head RGB + depth + FreeCam 1280x720, 30 Hz target, upstream render mode | videos (`record_cams`) |

Overrides: `RENDER_FPS=20`, `FREECAM=on|off`, `RENDER_MODE=raster|rtpt|pathtrace|offline`, `LIDAR=0|1`,
`SIM_CPUS=0-7|auto|none`. In `mobile_up.sh` with a profile, the lidars render for the SLAM stack and are off
for the packing stack (`WITH_MOVEIT=1`, closed loop on ground-truth `/odom`).

### Before / after (measured, 3070)

| | default (before) | collect | teleop | showcase | default + P-core pin only |
|---|---|---|---|---|---|
| render FPS (target) | 12.8 (30) | **30.0** (30) | 15.0 (15) | 12.8 (30) | 12.9 (30) |
| GPU frame time | 78 ms | 22 ms | 24 ms | 78 ms | 77 ms |
| head camera "map" (= GPU wait) | 71 ms | 17.7 ms | 19.5 ms | 71.6 ms | 71.5 ms |
| head RGB / depth / FreeCam Hz | 12.7 / 12.8 / 12.8 | 30.0 / 30.0 / – | 15.0 / 15.0 / – | 12.8 / 12.8 / 12.8 | 12.8 / 12.7 / 12.7 |
| GPU util / power | 91 % / 156 W | **55 % / 108 W** | 36 % / 65 W | 92 % / 156 W | 92 % / 156 W |
| VRAM (engine + render) | 4.1 GB (1.4 + 1.9) | 3.6 GB (1.4 + 1.3) | 3.5 GB | 4.4 GB (1.4 + 2.2) | 4.1 GB |
| physics Hz, overruns/500 | 98.7, 0.6 | **100.0, 0** | 99.8, 0.5 | 100.0, 0 | 99.8, 0.3 |
| tick step p99 / interval p99 | 9.4 / 21.1 ms | 5.9 / 11.0 ms | 5.6 / 11.2 ms | 5.7 / 10.6 ms | 5.5 / 10.2 ms |
| physics budget mean / worst | 93 % / 22 ms | 84 % / 11 ms | 82 % / 12 ms | 81 % / 11 ms | 81 % / 12 ms |
| "loop fell behind" per 30 s | 1–3 | 0 | 0 | 0 | 0 |
| CPU engine / render (cores) | 3.66 / 1.28 | 3.25 / 1.18 | 3.24 / 0.67 | 3.22 / 1.21 | 3.23 / 1.20 |
| first frame, warm caches | 14.9 s | 17.4 s | 17.4 s | 17.4 s | 15.4 s |

Mobile scene (`mobile_up.sh`, SLAM stack):

| | default | `PROFILE=collect` (lidar on) | `PROFILE=collect LIDAR=0` |
|---|---|---|---|
| render FPS, GPU frame | 10.5, 96 ms | 30.0, 31 ms | 30.0, 22 ms |
| lidar per sensor | 3.6 Hz | 10.0 Hz (nominal) | off |
| GPU util / VRAM | 92 % / 4.8 GB | 77 % / 3.9 GB | 53 % / 3.5 GB |
| physics Hz, fell-behind per 30 s | 97.7, 1 | 99.9, 0 | 100.0, 0 |

Startup (measured):

| | first rendered frame | engine loop running |
|---|---|---|
| all caches cold (first run on a machine) | 167 s (default), 207 s (collect) | ~180 s |
| fresh container, host caches kept, Kit cache lost (every `cell.sh up` before this change) | 20 s | ~85–90 s |
| warm | 15–17 s | 13–14 s |

`collect` reaches its first frame about 2.5 s later than the default. Its render node waits for this launch's
`assemble_scene` output before it starts. The two cold numbers are both dominated by shader compiles running
in parallel and vary from run to run. A cold start is a one-off once the caches persist.

### What changed (files)

- `ros_ws/scripts/render_profile.py` (new) starts upstream's `genie_sim_render_node` with our settings. The
  upstream launch file appends `render_fps: 30.0` after the launcher YAML's params, so the YAML cannot change
  it. The script:
  - waits for this launch's `manifest.json` and `render_layer.usda`;
  - writes `manifest_<profile>.json` and `render_layer_<profile>.usda` next to them. Only the camera list
    changes (the FreeCam is dropped when off) and the `omni:rtx:rendermode` token;
  - execs the node with the same node name and remaps as upstream, plus `render_fps` and `lidar`.

  Upstream genie_sim is untouched. Scene, cameras, intrinsics, topics and frame ids are identical.
- `ros_ws/config/launchers/launcher_ovrtx_isaac_physx_norender.yaml` (new): the upstream launcher with
  `renders: []`.
- `ros_ws/scripts/stage1_up.sh`: adds `PROFILE`, a render tmux window that logs to `render.log`, and
  `RENDER_MODE` / `RENDER_FPS` / `FREECAM` / `LIDAR`. The FreeCam aiming helper only starts when the FreeCam
  renders. With a profile it also:
  - sets `ISAACSIM_VIEWPORT_UPDATES=0`: the engine's Kit viewport is never shown headless; cost only;
  - sets `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`;
  - pins the sim with `SIM_CPUS=auto`: on a hybrid CPU it pins sim + render to the fast cores, on uniform
    cores it does nothing.

  The script also writes `perf_t0` for startup timing.
- `ros_ws/scripts/mobile_up.sh`: `PROFILE` and `LIDAR`, as described in Profiles above.
- `docker/cell.sh`: mounts `$CACHE_ROOT/kitcache` → `isaacsim/kit/cache` and `$CACHE_ROOT/optix` →
  `/var/tmp/OptixCache_isaac-sim`, and sets `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`. It takes effect at
  the next `cell.sh up`. The running container was not recreated, because other processes were running in
  it. `~/docker/isaac-sim/kitcache` and `optix` were seeded from the warm container.
- Measurement tools: `tools/sim_perf.py` (host), `ros_ws/scripts/perf_probe.py` (container) and
  `ros_ws/scripts/img_diff.py` (still comparison).

### Findings

1. **The head-camera "readback" of 60–75 ms is the GPU frame time.** The render node steps all render
   products, then maps them in manifest order. The first `ovrtx_map` blocks until the whole frame is done:
   head RGB + depth plus the 1280x720 FreeCam, which has 3.6x the head camera's pixels. The memcpy is 0.6 ms
   for all cameras, and depth/32FC1 costs nothing measurable. With the FreeCam listed first, the 74 ms moves
   to it and the head camera maps in 0.4 ms. Dropping the FreeCam takes the frame from 78 ms to 22 ms, so
   the head camera reaches its 30 Hz target.
2. **There was no path tracing to switch off.**
   - The "RealTimePathTracing" in the log is the engine's Kit viewport. Headless, it renders 0 of 500 ticks.
   - OVRTX renders `RaytracedLighting`, `RealTimePathTracing` (what upstream authors) and
     `Real-Time Path-Tracing` at the same cost, with the same image. The mean pixel difference to the
     baseline still is 0.8–1.3 (p99 5), the same as frame-to-frame noise.
   - `PathTracing` really is different: progressive, blotchy at real-time rates (mean difference 18.5,
     sharpness 290 vs 840), and 37 ms for the head camera alone. `Minimal` renders black frames.

   `collect` therefore uses `RaytracedLighting` explicitly and loses no image quality. `showcase` keeps the
   upstream token, so the videos look exactly as before.
3. **"loop fell behind" is a CPU problem.**
   - Each 10 ms physics tick spends ~4.1–4.9 ms in the PhysX solver and ~4.0–4.8 ms publishing clock,
     joint_states, `/tf_render` (56 bodies) and odom from Python. That leaves under 1 ms of slack.
   - On this hybrid CPU, ticks that run on E-cores overrun. Pinning to the P-cores alone gives 0 warnings
     and halves step p99 (9.4 → 5.5 ms) and interval p99 (21 → 10 ms), even with the GPU at 91 %.
   - GPU load is not the cause. Neither are the helpers: the relay uses 0.10–0.13 cores, `free_cam_pose`
     under 0.005. Unpinned runs varied with other load on the box (86–99 Hz).
4. **Engine CPU** is ~3.3 cores even though physics needs about half a core:
   - the main loop busy-waits: ~0.9 core;
   - Kit's `1 Render Async` thread spins at ~1 core, even with viewport updates off;
   - 24 `carb.tasking` workers use ~4 % each.

   None of this can be changed from our configs: upstream sets `limit_cpu_threads` and the Kit args. It
   matters on a server with 8 vCPUs (see below).
5. **Caches.**
   - The OVRTX shaders are compiled into the driver's Vulkan cache, `~/.cache/nvidia` (`GLCache`, ~330 MB).
     It is host-mounted already. Without it the first frame takes 167 s.
   - Kit runs in portable mode, so its cache lives in `isaacsim/kit/cache` (DerivedDataCache + shader
     caches, 0.4–0.9 GB) inside the container layer. Every `cell.sh up` / `session.sh down` + `up` lost it
     and paid ~75 s of Kit startup again.
   - Verified with throwaway containers (`--network none`): with `$CACHE_ROOT/kitcache` mounted,
     SimulationApp startup goes from 76.1 s cold to **4.9 s in a new container**; without the mount it is
     72.8 s.
   - The OptiX cache (~9 MB) is mounted too, for completeness.
6. **Lidar** renders at the render node's rate. The default mobile stack rendered at 10.5 fps, so each lidar
   published 3.6 Hz. `collect` gives the nominal 10 Hz, and with `LIDAR=0` it saves 24 % GPU and 0.4 GB
   VRAM.

Not done:

- A low-rate FreeCam alongside a 30 Hz head camera. One render node steps every product at one rate, so it
  would need a second OVRTX renderer: +1.3 GB VRAM, ~55 ms of 3070 GPU time per FreeCam frame. Use
  `showcase` when you need the FreeCam.
- Lowering the RTX sample counts or effects. That would change the images.

### Cache directories that must stay on persistent disk (server)

Everything is under `CACHE_ROOT` (default `~/docker/isaac-sim`; `scripts/server_setup.sh` uses the same):

| dir | mounted at | size | saves |
|---|---|---|---|
| `cache/main/nvidia` | `~/.cache/nvidia` | ~0.3 GB | ~150 s of OVRTX shader compile on every first frame |
| `kitcache` | `/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache` | 0.4–0.9 GB | ~75 s of Kit startup per container (re)creation |
| `cache/computecache`, `optix`, `cache/main/{ov,warp}` | CUDA JIT, OptiX, OV/warp | < 50 MB | small |

Driver caches are keyed to the driver and GPU, so the first session on the A6000 always compiles once
(~3 min). Later sessions start warm (~15–20 s to the first frame), but only if `CACHE_ROOT` is on the
machine's persistent disk.

### A6000 expectations (extrapolated; nothing here was run on an A6000)

- **GPU.** The A6000 is the same Ampere generation as the 3070, with 84 vs 46 SMs and RT cores, 768 vs
  448 GB/s, and similar clocks. Assume ~1.6–1.8x the render throughput. Then:
  - `collect` 22 ms → ~12–14 ms per frame: 30 Hz at ~30–35 % GPU;
  - `teleop` ~15–20 %;
  - `showcase` 78 ms → ~45–50 ms: ~20–22 fps for the FreeCam videos (12.8 here). Raise `RENDER_FPS` only
    if the measured frame time allows it.
- **VRAM.** 3.5–4.4 GB of 48 GB, so VRAM is not a limit.
- **CPU is the likely limit.** Paperspace lists 8 vCPUs for its single-A6000 machine type; check it in the
  console. The stack uses about:
  - engine 3.2–3.7 cores;
  - render 0.7–1.3;
  - relay 0.1;
  - MoveIt up to ~1 while planning;
  - Foxglove / recorder / teleop extra.

  That is ~5–6.5 of 8 vCPUs. The physics tick needs ~8–9 ms of single-thread time per 10 ms on a 5.7 GHz
  P-core. A server vCPU (Xeon/EPYC at 2.x–3.x GHz) may need noticeably more, and then the 100 Hz loop runs
  slower than real time.
- **First thing to do on the server:** run `python3 tools/sim_perf.py --label a6000_collect --watch-startup`
  after `PROFILE=collect stage1_up.sh`. If `physics.budget_mean_pct` exceeds ~90 %, or `actual_hz` stays
  below 99, you need a machine with more or faster vCPUs, or upstream changes to the engine's
  publish/busy-wait. `SIM_CPUS=auto` does nothing on uniform cores.
- **Cost:**
  - `collect` delivers 30 head frames/s instead of 12.8, i.e. 2.3x the camera frames per billed hour at the
    same physics rate;
  - persistent caches save ~75 s per container start, plus ~3 min on the first launch after a re-image;
  - at ~55 % of a 3070, `collect` should also fit a smaller RTX card (A4000/A5000 class; extrapolated).
    `showcase` is where the A6000 pays off.
### Object pose relay cost (2026-10-11)

`OBJ_RELAY=1` (default) adds one engine post-step hook. It samples 14 object rigid bodies at 25 Hz, sends only
the bodies that changed and a 2 s keyframe, and publishes on its own thread. On the 3070 with `PROFILE=collect`:
0.17 ms per sample, +0.05 ms per physics tick on average (max 0.23 ms), about 0.4 % of one core. Engine CPU,
render fps and frame time (30.0 fps, 22.8 ms) and the physics step (4.45 ms, 0 overruns) are unchanged against
`OBJ_RELAY=0`. The render node applies 14 extra xforms at most every 40 ms, which does not show in its frame time.
Details are in `ros_ws/src/g2cell_bringup/README.md`; data is in `output/readiness/relay/perf_*.json`.

<!-- END section: Sim runtime -->
