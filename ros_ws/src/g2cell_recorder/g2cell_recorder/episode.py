"""ROS-free episode logic: message -> array conversion, time alignment and the HDF5 schema.

Messages are duck-typed (anything with the sensor_msgs field names works), so this module runs
with rosbag2_py messages in the container, mcap_ros2 messages on the host, or plain test objects.
Image streams may arrive raw (sensor_msgs/Image) or through an image_transport plugin
(sensor_msgs/CompressedImage: compressed JPEG/PNG, compressedDepth, zstd); see image_codec.py.
Either way the HDF5 holds decoded frames, so readers do not care how a bag was recorded.

HDF5 layout (one file per episode; all times are int64 ns, bag receive time unless noted):

    /                         attrs: schema_version, source_bag, scene, seed, success,
                                     start_time_ns, end_time_ns, duration_s, aligned_rate_hz,
                                     stream_rates_json, topics_json,
                                     image_compression, image_compression_level
    /images/<cam>/data        uint8   [N,H,W,3] (alpha dropped)  one frame per chunk,
                                      gzip (default level 4) | lzf | none
    /images/<cam>/t_ns, header_t_ns           attrs: topic, encoding, transport, rate_hz
    /depth/<cam>/data         float32 [N,H,W]  (metres; NaN = no return) one frame per chunk,
                                      byte-shuffle + the same compression
    /joint_states/positions|velocities|efforts float64 [N,J]   attrs: names, topic, rate_hz
    /joint_command/actions    float64 [N,J] (NaN where a message did not command a joint)
    /aligned/t_ns             common clock at aligned_rate_hz over the window every stream covers
    /aligned/<stream>/index   int64 [M] nearest-previous sample of that stream
    /aligned/joint_states/positions|velocities|efforts, /aligned/joint_command/actions
                              [M,J] values at index, NaNs forward-filled per joint
    /aligned/images|depth/<cam>/data  only with aligned_images=True (otherwise use index)

Speed: frames are decoded and (for gzip) byte-shuffled + deflated in worker processes
(`workers`), and the main process writes the finished chunks with H5Dwrite_chunk, so the
single-threaded HDF5 filter pipeline is bypassed. Files are standard gzip/shuffle datasets.
"""
from __future__ import annotations

import json
import os
import zlib
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA_VERSION = 1

COMPRESSIONS = ("gzip", "lzf", "none")
DEFAULT_COMPRESSION = "gzip"
DEFAULT_GZIP_LEVEL = 4

_HEAD = "/genie_sim/head_front_camera_rgb/image_raw"
_FREE = "/genie_sim/free_camera_rgb/image_raw"
_DEPTH = "/genie_sim/head_front_camera_depth/image_raw"
# throttled copies of the depth stream written by throttle_relay (record_episode.sh --depth-rate)
DEPTH_RELAY_PREFIX = "/g2cell/rec/head_front_camera_depth/"
DEPTH_RELAY_TOPIC = DEPTH_RELAY_PREFIX + "compressedDepth"

# topic -> (kind, key). Several transports of one camera map to the same stream; if a bag holds
# more than one of them, the first topic seen wins and the others are reported as ignored.
DEFAULT_TOPICS: dict[str, tuple[str, str]] = {
    "/joint_states": ("joint_states", "joint_states"),
    "/joint_command": ("joint_command", "joint_command"),
    _HEAD: ("image", "head_front_rgb"),
    _HEAD + "/compressed": ("image", "head_front_rgb"),
    _HEAD + "/zstd": ("image", "head_front_rgb"),
    _FREE: ("image", "free_rgb"),
    _FREE + "/compressed": ("image", "free_rgb"),
    _FREE + "/zstd": ("image", "free_rgb"),
    _DEPTH: ("depth", "head_front"),
    _DEPTH + "/compressedDepth": ("depth", "head_front"),
    _DEPTH + "/zstd": ("depth", "head_front"),
    DEPTH_RELAY_TOPIC: ("depth", "head_front"),
    DEPTH_RELAY_PREFIX + "zstd": ("depth", "head_front"),
    DEPTH_RELAY_PREFIX + "image_raw": ("depth", "head_front"),
}

# ---------------------------------------------------------------- conversions

# raw-message helpers live in image_codec (re-exported here for existing callers)
from g2cell_recorder.image_codec import (  # noqa: E402
    decode_payload,
    depth_to_float32,
    frame_payload,
    image_to_rgb,
    payload_encoding,
)


def joint_matrix(names_list: list[list[str]], values_list: list[list[float]]):
    """Per-message (names, values) -> (union of names in first-seen order, float64 [N,J]).

    Joints a message does not mention (or empty value lists) are NaN.
    """
    names: list[str] = []
    col: dict[str, int] = {}
    for ns in names_list:
        for n in ns:
            if n not in col:
                col[n] = len(names)
                names.append(n)
    mat = np.full((len(names_list), len(names)), np.nan)
    for i, (ns, vs) in enumerate(zip(names_list, values_list)):
        for n, v in zip(ns, vs):
            mat[i, col[n]] = v
    return names, mat


def ffill_nan(mat: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down each column (rows = time)."""
    if mat.size == 0:
        return mat.copy()
    valid = ~np.isnan(mat)
    idx = np.where(valid, np.arange(mat.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = mat[idx, np.arange(mat.shape[1])[None, :]]
    return out


# ---------------------------------------------------------------- alignment

def aligned_grid(stream_times: dict[str, np.ndarray], rate_hz: float) -> np.ndarray:
    """Fixed-rate clock (int64 ns) from the latest first sample to the earliest last sample."""
    if not rate_hz or rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    ts = [np.asarray(t) for t in stream_times.values() if len(t)]
    if not ts:
        return np.zeros(0, dtype=np.int64)
    t0 = max(int(t[0]) for t in ts)
    t1 = min(int(t[-1]) for t in ts)
    if t1 < t0:
        return np.zeros(0, dtype=np.int64)
    period = 1e9 / rate_hz
    n = int(np.floor((t1 - t0) / period + 1e-9)) + 1
    return t0 + np.round(np.arange(n) * period).astype(np.int64)


def previous_index(t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the latest sample with t <= grid time (t sorted ascending)."""
    idx = np.searchsorted(np.asarray(t), np.asarray(grid), side="right") - 1
    if len(idx) and idx.min() < 0:
        raise ValueError("grid starts before the first sample")
    return idx.astype(np.int64)


# ---------------------------------------------------------------- conversion

def _stamp_ns(msg) -> int:
    try:
        s = msg.header.stamp
        return int(s.sec) * 1_000_000_000 + int(s.nanosec)
    except AttributeError:
        return -1


def _rate(t: np.ndarray) -> float:
    return float((len(t) - 1) * 1e9 / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 0.0


# ---------------------------------------------------------------- frame encoding (workers)

def _check_compression(compression: str, level) -> int | None:
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS}, got {compression!r}")
    if compression != "gzip":
        return None
    level = DEFAULT_GZIP_LEVEL if level is None else int(level)
    if not 0 <= level <= 9:
        raise ValueError("gzip level must be 0..9")
    return level


def _shuffle_bytes(arr: np.ndarray) -> bytes:
    """HDF5 shuffle filter: byte k of every element grouped together (k = 0..itemsize-1)."""
    a = np.ascontiguousarray(arr)
    if a.dtype.itemsize == 1:
        return a.tobytes()
    return np.ascontiguousarray(a.reshape(-1).view(np.uint8)
                                .reshape(-1, a.dtype.itemsize).T).tobytes()


def encode_frame(payload: dict, compression: str, level: int | None):
    """Decode one image payload; for gzip also shuffle + deflate it into a ready HDF5 chunk.

    Returns (shape, dtype str, chunk bytes or None, array or None, encoding, transport).
    Runs in worker processes, so it only takes and returns picklable values.
    """
    arr = decode_payload(payload)
    chunk = None
    if compression == "gzip":
        chunk = zlib.compress(_shuffle_bytes(arr), level)
        arr_out = None
    else:
        arr_out = arr
    return (arr.shape, arr.dtype.str, chunk, arr_out, payload_encoding(payload),
            payload["transport"])


def _warm(_=None):  # starts every worker before HDF5/bag files are opened
    return os.getpid()


# ---------------------------------------------------------------- streams

class _ImageStream:
    def __init__(self, grp, topic: str, depth: bool, compression: str, level):
        self.grp, self.topic, self.depth = grp, topic, depth
        self.compression, self.level = compression, level
        self.ds = None
        self.n = 0
        self.t: list[int] = []
        self.th: list[int] = []
        self.encoding = ""
        self.transport = ""
        self.create_kw: dict = {}

    def note(self, t_ns: int, msg):
        self.t.append(t_ns)
        self.th.append(_stamp_ns(msg))

    def _create(self, shape, dtype):
        dtype = np.dtype(dtype)
        kw = dict(chunks=(1, *shape))
        if self.compression != "none":
            kw["compression"] = self.compression
            if self.compression == "gzip":
                kw["compression_opts"] = self.level
            kw["shuffle"] = dtype.itemsize > 1
        self.create_kw = kw
        self.ds = self.grp.create_dataset("data", shape=(0, *shape), maxshape=(None, *shape),
                                          dtype=dtype, **kw)

    def write(self, shape, dtype, chunk, arr, encoding, transport):
        if self.ds is None:
            self._create(shape, dtype)
            self.encoding, self.transport = encoding, transport
        elif tuple(shape) != self.ds.shape[1:]:
            raise ValueError(f"{self.topic}: frame size changed {self.ds.shape[1:]} -> {shape}")
        if self.n >= self.ds.shape[0]:  # grow geometrically, trimmed in finish()
            self.ds.resize(max(16, 2 * self.ds.shape[0]), axis=0)
        if chunk is not None:
            self.ds.id.write_direct_chunk((self.n,) + (0,) * len(shape), chunk)
        else:
            self.ds[self.n] = arr
        self.n += 1

    def finish(self) -> np.ndarray:
        if self.n != len(self.t):
            raise RuntimeError(f"{self.topic}: {self.n} frames written for {len(self.t)} stamps")
        self.ds.resize(self.n, axis=0)
        t = np.asarray(self.t, dtype=np.int64)
        self.grp.create_dataset("t_ns", data=t)
        self.grp.create_dataset("header_t_ns", data=np.asarray(self.th, dtype=np.int64))
        self.grp.attrs.update(topic=self.topic, encoding=self.encoding,
                              transport=self.transport, rate_hz=_rate(t))
        return t


class _JointStream:
    def __init__(self, topic: str):
        self.topic = topic
        self.t: list[int] = []
        self.th: list[int] = []
        self.names: list[list[str]] = []
        self.pos: list[list[float]] = []
        self.vel: list[list[float]] = []
        self.eff: list[list[float]] = []

    def add(self, t_ns: int, msg):
        self.t.append(t_ns)
        self.th.append(_stamp_ns(msg))
        self.names.append(list(msg.name))
        self.pos.append(list(msg.position))
        self.vel.append(list(msg.velocity))
        self.eff.append(list(msg.effort))


def _str_attr(names: list[str]):
    return np.array(names, dtype=object) if names else np.array([], dtype="S1")


class _FramePipeline:
    """Ordered frame encoding: inline (workers=0) or in a process pool with a bounded window."""

    def __init__(self, workers: int, compression: str, level):
        self.compression, self.level = compression, level
        self.pool = None
        self.pending: deque = deque()
        self.window = 4 * max(workers, 1)
        if workers > 0:
            import concurrent.futures as cf
            import multiprocessing as mp
            # forkserver: no fork() of a threaded parent; workers import this module once
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods()
                                 else "spawn")
            if ctx.get_start_method() == "forkserver":
                ctx.set_forkserver_preload(["g2cell_recorder.episode"])
            self.pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            list(self.pool.map(_warm, range(workers)))

    def submit(self, stream: _ImageStream, payload: dict):
        if self.pool is None:
            stream.write(*encode_frame(payload, self.compression, self.level))
            return
        self.pending.append((stream, self.pool.submit(encode_frame, payload,
                                                      self.compression, self.level)))
        while len(self.pending) > self.window:
            self._pop()

    def _pop(self):
        stream, fut = self.pending.popleft()
        stream.write(*fut.result())

    def drain(self):
        while self.pending:
            self._pop()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None


def convert_messages(messages: Iterable[tuple[str, int, object]], out_path, source: str,
                     scene: str = "", seed: int = -1, aligned_rate_hz: float = 30.0,
                     aligned_images: bool = False, success: str = "unknown",
                     topics: dict[str, tuple[str, str]] | None = None,
                     compression: str = DEFAULT_COMPRESSION, compression_level=None,
                     workers: int = 0) -> dict:
    """Write one HDF5 episode from (topic, receive_time_ns, msg) tuples in time order.

    compression: image/depth datasets, gzip (level 4 default) | lzf | none.
    workers: processes that decode (and gzip) frames; 0 = all in this process.
    Returns a small summary dict. Writes to <out>.partial and renames on success.
    """
    import h5py  # local import: keeps the pure helpers usable without h5py

    if success not in ("true", "false", "unknown"):
        raise ValueError("success must be true, false or unknown")
    level = _check_compression(compression, compression_level)
    topics = DEFAULT_TOPICS if topics is None else topics
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = _FramePipeline(int(workers), compression, level)  # fork before opening files
    try:
        with h5py.File(tmp, "w") as f:
            summary = _write(f, messages, topics, aligned_rate_hz, aligned_images, pipe,
                             compression, level)
            f.attrs.update(
                schema_version=SCHEMA_VERSION, source_bag=str(source), scene=scene,
                seed=int(seed), success=success, aligned_rate_hz=float(aligned_rate_hz),
                start_time_ns=summary["start_time_ns"], end_time_ns=summary["end_time_ns"],
                duration_s=(summary["end_time_ns"] - summary["start_time_ns"]) / 1e9,
                stream_rates_json=json.dumps(summary["rates_hz"]),
                topics_json=json.dumps({k: list(v) for k, v in topics.items()}),
                image_compression=compression,
                image_compression_level=-1 if level is None else level)
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        pipe.close()
    summary["path"] = str(out_path)
    return summary


def _copy_frames(src, dst, idx):
    """dst[i] = src[idx[i]], copying stored chunks as-is when both use one frame per chunk."""
    direct = src.chunks == (1, *src.shape[1:]) and dst.chunks == src.chunks
    zeros = (0,) * (src.ndim - 1)
    last, frame = -1, None
    for i, j in enumerate(idx):
        j = int(j)
        if direct:
            if j != last:
                mask, frame = src.id.read_direct_chunk((j,) + zeros)
                last = j
            dst.id.write_direct_chunk((i,) + zeros, frame, filter_mask=mask)
        else:
            if j != last:
                frame, last = src[j], j
            dst[i] = frame


def _write(f, messages, topics, rate_hz, aligned_images, pipe, compression, level) -> dict:
    images: dict[str, _ImageStream] = {}
    owner: dict[str, str] = {}          # stream path -> topic that feeds it
    ignored: dict[str, int] = {}
    joints: dict[str, _JointStream] = {}
    t_min, t_max = None, None
    for topic, t_ns, msg in messages:
        spec = topics.get(topic)
        if spec is None:
            continue
        kind, key = spec
        if kind in ("image", "depth"):
            path = f"{'images' if kind == 'image' else 'depth'}/{key}"
        elif kind in ("joint_states", "joint_command"):
            path = kind
        else:
            raise ValueError(f"unknown stream kind {kind!r} for {topic}")
        if owner.setdefault(path, topic) != topic:
            ignored[topic] = ignored.get(topic, 0) + 1
            continue
        t_ns = int(t_ns)
        t_min = t_ns if t_min is None else min(t_min, t_ns)
        t_max = t_ns if t_max is None else max(t_max, t_ns)
        if kind in ("image", "depth"):
            if path not in images:
                images[path] = _ImageStream(f.require_group(path), topic, kind == "depth",
                                            compression, level)
            images[path].note(t_ns, msg)
            pipe.submit(images[path], frame_payload(msg, kind == "depth"))
        else:
            if kind not in joints:
                joints[kind] = _JointStream(topic)
            joints[kind].add(t_ns, msg)
    pipe.drain()
    if t_min is None:
        raise ValueError("bag contains none of the configured topics")

    times: dict[str, np.ndarray] = {}
    rates: dict[str, float] = {}
    for path, s in images.items():
        times[path] = s.finish()
        rates[path] = _rate(times[path])

    lowdim: dict[str, dict[str, np.ndarray]] = {}
    for kind, s in joints.items():
        g = f.create_group(kind)
        t = np.asarray(s.t, dtype=np.int64)
        names, pos = joint_matrix(s.names, s.pos)
        fields = {"actions": pos} if kind == "joint_command" else {
            "positions": pos,
            "velocities": joint_matrix(s.names, s.vel)[1],
            "efforts": joint_matrix(s.names, s.eff)[1],
        }
        for k, v in fields.items():
            g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        g.create_dataset("t_ns", data=t)
        g.create_dataset("header_t_ns", data=np.asarray(s.th, dtype=np.int64))
        g.attrs.update(names=_str_attr(names), topic=s.topic, rate_hz=_rate(t))
        times[kind], rates[kind], lowdim[kind] = t, _rate(t), fields

    grid = aligned_grid(times, rate_hz)
    al = f.create_group("aligned")
    al.create_dataset("t_ns", data=grid)
    al.attrs.update(rate_hz=float(rate_hz), method="nearest-previous",
                    streams=_str_attr(sorted(times)))
    for path, t in times.items():
        g = al.create_group(path)
        idx = previous_index(t, grid) if len(grid) else np.zeros(0, dtype=np.int64)
        g.create_dataset("index", data=idx)
        if path in lowdim:
            for k, v in lowdim[path].items():
                g.create_dataset(k, data=ffill_nan(v)[idx])
        elif aligned_images:
            src = f[path]["data"]
            dst = g.create_dataset("data", shape=(len(idx), *src.shape[1:]), dtype=src.dtype,
                                   **images[path].create_kw)
            _copy_frames(src, dst, idx)
    return {"start_time_ns": t_min, "end_time_ns": t_max, "rates_hz": rates,
            "counts": {p: int(len(t)) for p, t in times.items()}, "aligned_len": int(len(grid)),
            "transports": {p: s.transport for p, s in images.items()},
            "topics_used": {p: owner[p] for p in times},
            "ignored_topics": ignored, "image_compression": compression}
