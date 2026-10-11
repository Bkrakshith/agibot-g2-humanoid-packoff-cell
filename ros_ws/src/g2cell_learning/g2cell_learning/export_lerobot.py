"""Convert episode HDF5 files to a LeRobot v2.0-style dataset folder (no lerobot dependency).

    root/
      meta/info.json            codebase_version "v2.0", fps, features, path templates, totals
      meta/episodes.jsonl       {"episode_index", "tasks": [...], "length"} per episode
      meta/tasks.jsonl          {"task_index", "task"}
      meta/stats.json           mean/std/min/max per feature (v2.0 kept global stats here)
      data/chunk-000/episode_000000.parquet
            observation.state  list<float32>[J]   measured positions of the commanded joints
            action             list<float32>[J]   /joint_command (latest command at the frame)
            timestamp float32 (frame_index / fps), frame_index, episode_index, index, task_index
      videos/chunk-000/observation.images.<key>/episode_000000.mp4

The layout is reproduced from memory of LeRobot v2.0 (see README "LeRobot format caveats").

    python -m g2cell_learning.export_lerobot ep*.hdf5 -o output/datasets/run --task "..."
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import load_episode_arrays

CODEBASE_VERSION = "v2.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_CAMERAS = {"head_front_rgb": "head"}


def write_video(path: Path, frames_iter, fps: int, height: int, width: int,
                codec: str = "libx264", crf: int = 23, gop: int = 2) -> int:
    """Encode uint8 RGB frames to mp4 (yuv420p). Odd sizes are padded by one pixel."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    h2, w2 = height + height % 2, width + width % 2
    n = 0
    with av.open(str(path), "w") as c:
        s = c.add_stream(codec, rate=Fraction(fps))
        s.width, s.height, s.pix_fmt = w2, h2, "yuv420p"
        if codec in ("libx264", "libsvtav1"):
            s.options = {"crf": str(crf), "g": str(gop)}
        for fr in frames_iter:
            if fr.shape[:2] != (h2, w2):
                fr = np.pad(fr, ((0, h2 - fr.shape[0]), (0, w2 - fr.shape[1]), (0, 0)),
                            mode="edge")
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr),
                                                           format="rgb24")):
                c.mux(pkt)
            n += 1
        for pkt in s.encode():
            c.mux(pkt)
    return n


def _aligned_frames(f, cam):
    ds = f[f"images/{cam}/data"]
    idx = f[f"aligned/images/{cam}/index"][:]
    last, frame = -1, None
    for j in idx:
        if j != last:
            frame, last = ds[int(j)], int(j)
        yield frame


def _stats(x: np.ndarray) -> dict:
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "min": x.min(0).tolist(), "max": x.max(0).tolist()}


def _image_stats(f, cam, n=20) -> dict:
    ds = f[f"images/{cam}/data"]
    idx = f[f"aligned/images/{cam}/index"][:]
    pick = np.unique(idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)])
    px = np.stack([ds[int(i)] for i in pick]).reshape(-1, 3).astype(np.float64) / 255.0
    to = lambda v: [[[float(c)]] for c in v]  # noqa: E731  (LeRobot keeps [C,1,1])
    return {"mean": to(px.mean(0)), "std": to(px.std(0)), "min": to(px.min(0)),
            "max": to(px.max(0))}


def export_lerobot(paths, root, task: str = "teleop demonstration", fps: int = 30,
                   cameras: dict[str, str] | None = None, robot_type: str = "agibot_g2",
                   vcodec: str = "libx264", overwrite: bool = False) -> dict:
    """Write the dataset; returns the info.json dict. Episodes without actions are skipped."""
    cameras = dict(DEFAULT_CAMERAS if cameras is None else cameras)
    root = Path(root)
    if root.exists():
        if not overwrite and any(root.iterdir()):
            raise FileExistsError(f"{root} exists and is not empty (use overwrite=True)")
        if overwrite:
            shutil.rmtree(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    episodes, all_state, all_action = [], [], []
    names, video_shapes, img_stats = None, {}, {}
    index = 0
    for p in map(str, paths):
        try:
            arr = load_episode_arrays(p, names)
        except ValueError as e:
            print(f"[export] skip {p}: {e}", file=sys.stderr)
            continue
        names = names or arr.joint_names
        ei = len(episodes)
        chunk = ei // CHUNKS_SIZE
        T = len(arr.t_ns)
        if abs(float(arr.attrs.get("aligned_rate_hz", fps)) - fps) > 1e-6:
            raise ValueError(f"{p}: aligned rate {arr.attrs.get('aligned_rate_hz')} != fps {fps}")
        table = pa.table({
            "observation.state": pa.array(arr.state.astype(np.float32).tolist(),
                                          type=pa.list_(pa.float32())),
            "action": pa.array(arr.action.astype(np.float32).tolist(),
                               type=pa.list_(pa.float32())),
            "timestamp": pa.array((np.arange(T) / fps).astype(np.float32)),
            "frame_index": pa.array(np.arange(T, dtype=np.int64)),
            "episode_index": pa.array(np.full(T, ei, dtype=np.int64)),
            "index": pa.array(np.arange(index, index + T, dtype=np.int64)),
            "task_index": pa.array(np.zeros(T, dtype=np.int64)),
        })
        dp = root / DATA_PATH.format(episode_chunk=chunk, episode_index=ei)
        dp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dp)
        with h5py.File(p, "r") as f:
            for cam, key in cameras.items():
                shape = f[f"images/{cam}/data"].shape[1:]
                vk = f"observation.images.{key}"
                if video_shapes.setdefault(vk, shape) != shape:
                    raise ValueError(f"{p}: {cam} frame size {shape} != {video_shapes[vk]}")
                vp = root / VIDEO_PATH.format(episode_chunk=chunk, video_key=vk,
                                              episode_index=ei)
                n = write_video(vp, _aligned_frames(f, cam), fps, shape[0], shape[1], vcodec)
                assert n == T, (n, T)
                if vk not in img_stats:
                    img_stats[vk] = _image_stats(f, cam)
        episodes.append({"episode_index": ei, "tasks": [task], "length": T,
                         "source": p, "scene": arr.attrs.get("scene", ""),
                         "seed": arr.attrs.get("seed", -1),
                         "success": arr.attrs.get("success", "unknown")})
        all_state.append(arr.state)
        all_action.append(arr.action)
        index += T

    n_ep = len(episodes)
    codec_name = {"libx264": "h264", "libsvtav1": "av1"}.get(vcodec, vcodec)
    motors = {"motors": list(names or [])}
    J = len(names or [])
    features = {
        "observation.state": {"dtype": "float32", "shape": [J], "names": motors},
        "action": {"dtype": "float32", "shape": [J], "names": motors},
    }
    for vk, shape in video_shapes.items():
        features[vk] = {
            "dtype": "video", "shape": [int(shape[0]), int(shape[1]), 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": float(fps), "video.height": int(shape[0]),
                     "video.width": int(shape[1]), "video.channels": 3,
                     "video.codec": codec_name, "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"),
                  ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": n_ep,
        "total_frames": index,
        "total_tasks": 1,
        "total_videos": n_ep * len(video_shapes),
        "total_chunks": (n_ep - 1) // CHUNKS_SIZE + 1 if n_ep else 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    with (root / "meta/episodes.jsonl").open("w") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    stats = {}
    if all_state:
        s, a = np.concatenate(all_state), np.concatenate(all_action)
        stats = {"observation.state": _stats(s), "action": _stats(a), **img_stats}
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="+", help="episode .hdf5 files")
    p.add_argument("-o", "--root", required=True)
    p.add_argument("--task", default="teleop demonstration")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", action="append", default=None, metavar="CAM=KEY",
                   help="HDF5 camera -> video key, e.g. head_front_rgb=head (repeatable); "
                        "default head_front_rgb=head")
    p.add_argument("--vcodec", default="libx264", help="libx264 (default) | libsvtav1")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    cams = dict(c.split("=", 1) for c in a.camera) if a.camera else None
    info = export_lerobot(a.episodes, a.root, a.task, a.fps, cams, vcodec=a.vcodec,
                          overwrite=a.overwrite)
    print(json.dumps({k: info[k] for k in ("total_episodes", "total_frames", "total_videos")}))


if __name__ == "__main__":
    main()
