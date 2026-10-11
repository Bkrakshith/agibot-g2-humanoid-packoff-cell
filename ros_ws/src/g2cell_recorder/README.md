# g2cell_recorder

Records teleop episodes as MCAP rosbags and converts each one to an HDF5 episode file. The schema
is at the top of `g2cell_recorder/episode.py`.

| file | role |
|---|---|
| `scripts/record_episode.sh` | `ros2 bag record` with the chosen topic profile, then `bag_to_hdf5` |
| `throttle_relay.py` | republishes a topic at a lower rate without deserialising it (depth at 5 Hz) |
| `bag_reader.py` | reads a bag as `(topic, t_ns, msg)` through rosbag2_py (container) or `mcap` (host) |
| `image_codec.py` | decodes raw, `compressed` (JPEG/PNG), `compressedDepth` and `zstd` image payloads |
| `episode.py` | alignment, the HDF5 schema, and the streaming parallel writer |
| `bag_to_hdf5.py`, `mark_episode.py` | the CLIs |

## Recording

```bash
ros2 run g2cell_recorder record_episode.sh ep001 --scene scene_g2cell --seed 0   # cheap defaults
ros2 run g2cell_recorder record_episode.sh ep002 --free-cam                       # + overview camera
ros2 run g2cell_recorder record_episode.sh ep003 --full-fidelity                  # raw, lossless
ros2 run g2cell_recorder record_episode.sh ep004 --dry-run                        # print topics + command
```

| flag | default | effect |
|---|---|---|
| `--rgb compressed\|zstd\|raw` | `compressed` | head (and free) RGB over image_transport: JPEG q95, lossless zstd, or raw |
| `--depth compressedDepth\|zstd\|raw\|off` | `compressedDepth` | head depth: PNG of quantised inverse depth (< 1 mm error below 3 m), zstd, raw, none |
| `--depth-rate HZ\|full` | `5` | throttles depth through `throttle_relay` (topic `/g2cell/rec/head_front_camera_depth/<transport>`) |
| `--free-cam` | off | also records `/genie_sim/free_camera_rgb` with the `--rgb` transport |
| `--compression zstd_fast\|zstd_small\|none` | `zstd_fast` | MCAP chunk compression (`--storage-preset-profile`); lossless, read transparently |
| `--full-fidelity` | | `--rgb raw --depth raw --depth-rate full --free-cam` |

`/joint_states`, `/joint_command`, `/tf` and `/tf_static` are always recorded at full rate. Measured on the
Stage 1 episode (20 s), the defaults cut the recording from 78 MB/s to about 1.1 MB/s on disk (1.54 GB to
~22 MB). Per-topic numbers are in [docs/OPTIMISATION.md](../../../docs/OPTIMISATION.md) "Data & training".
The `compressed`, `compressedDepth` and `zstd` topics are published by Genie Sim's render node through
image_transport. The defaults were sized by re-encoding the recorded raw frames offline. They have not
yet been recorded live from the sim.

## Conversion

```bash
bag_to_hdf5 /g2cell_output/episodes/ep001/bag -o .../ep001.hdf5 [--compression gzip|lzf|none]
            [--level 4] [--workers N] [--aligned-images] [--backend auto|rosbag2|mcap]
```

- Every transport of a camera maps to the same HDF5 stream (`DEFAULT_TOPICS`). The file always holds
  decoded frames, so the datasets and the LeRobot export do not care how the episode was recorded. The
  group attrs `transport` and `encoding` record the source. If one bag carries two transports of the same
  camera, the first topic seen wins and the other is reported as ignored.
- Frames are decoded and gzip-compressed in `--workers` processes (default: cores - 1, at most 8). The
  main process writes the finished chunks directly (`H5Dwrite_chunk`), one frame per chunk, with the
  file growing as it streams. On the Stage 1 bag this takes 3.3 s instead of 14.7 s and gives the same
  317 MB file (every dataset equal). A bag recorded with the defaults converts in 0.9 s to 71 MB.
- `--compression lzf` runs HDF5's filter in one thread. It is faster than gzip only without workers, and the
  files are about 1.6x larger. `--level 1` is about 20 % faster and 15 % larger than 4.
- JPEG decoding uses cv2 (present in the container) or Pillow. PNG and compressedDepth fall back to a
  pure-numpy decoder when neither is installed; it is slow, about 0.5 s per frame. `.venv-tools` has
  neither library, so JPEG bags are converted in the container or in a python that has Pillow.

## Tests

```bash
../../../../.venv-tools/bin/python -m pytest -q test     # 63 passed, 1 skipped (JPEG: no Pillow)
../../../../.venv-learn/bin/python -m pytest -q test     # runs the JPEG test; skips the mcap ones
```
