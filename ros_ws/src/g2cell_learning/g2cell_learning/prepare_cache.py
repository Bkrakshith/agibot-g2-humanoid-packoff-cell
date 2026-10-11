"""Decode + resize every aligned camera frame once into uint8 memory-mapped .npy files.

Training then reads the cache (``dataset.CachedEpisodeDataset``): no gzip inflate, no resize and
no float conversion per sample, and DataLoader workers share the frames through the page cache.

    python -m g2cell_learning.prepare_cache output/episodes/*/*.hdf5 --cache-dir output/cache \
        --image-size 128 160 [--extra-cameras free_rgb] [--workers 8]

Layout, one directory per (episode file version, image size):

    <cache-dir>/<episode stem>-<hash>/
        meta.json            source path/size/mtime, length T, joint names, cameras, attrs,
                             per-episode moments of state/action (n, mean, M2, min, max) and
                             per-camera RGB mean/std, so train-split stats need no HDF5 read
        lowdim.npz           t_ns [T], state [T,J], action [T,J] (float64, as load_episode_arrays)
        <cam>.npy            uint8 [U,3,h,w]  each distinct source frame once (camera < 30 Hz)
        <cam>_lut.npy        int32 [T]        aligned step -> row of <cam>.npy
    <cache-dir>/norm_stats.json    stats over all episodes given on the command line

The hash covers the episode's absolute path, size and mtime, the image size and the cache
version, so a re-converted episode or a new image size gets a new directory; cameras are added
to an existing directory on demand. Frames are resized exactly like ``dataset.resize_frames``
(bilinear, antialiased) and rounded to uint8 (|error| <= 0.5/255).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .dataset import HEAD_CAMERA, NormStats, load_episode_arrays, resize_frames

CACHE_VERSION = 1
BLOCK = 64     # unique frames per work item (~2.7 MB each for 720p sources)


# ---------------------------------------------------------------- keys and stats

def cache_dir_for(path, cache_root, image_size: Sequence[int], cameras=None) -> Path:
    """Directory for this episode file version and image size (cameras share it)."""
    p = Path(path).resolve()
    st = p.stat()
    key = f"{CACHE_VERSION}|{p}|{st.st_size}|{st.st_mtime_ns}|{tuple(int(x) for x in image_size)}"
    return Path(cache_root) / f"{p.stem}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def _moments(x: np.ndarray) -> dict:
    mean = x.mean(0)
    return {"n": int(len(x)), "mean": mean.tolist(), "m2": ((x - mean) ** 2).sum(0).tolist(),
            "min": x.min(0).tolist(), "max": x.max(0).tolist()}


def _combine(moms: list[dict]):
    """Chan et al. parallel combination -> (mean, population std, min, max)."""
    n = 0
    mean = m2 = mn = mx = None
    for m in moms:
        nb, mb, m2b = m["n"], np.asarray(m["mean"]), np.asarray(m["m2"])
        if n == 0:
            n, mean, m2 = nb, mb, m2b
            mn, mx = np.asarray(m["min"]), np.asarray(m["max"])
            continue
        delta = mb - mean
        tot = n + nb
        mean = mean + delta * nb / tot
        m2 = m2 + m2b + delta ** 2 * n * nb / tot
        n = tot
        mn, mx = np.minimum(mn, m["min"]), np.maximum(mx, m["max"])
    return mean, np.sqrt(m2 / n), mn, mx


def load_cached_stats(dirs, std_floor: float = 1e-2) -> NormStats:
    """NormStats over the given cache dirs; equals dataset.compute_norm_stats on the sources."""
    metas = [json.loads((Path(d) / "meta.json").read_text()) for d in dirs]
    names = metas[0]["joint_names"]
    for m in metas[1:]:
        if m["joint_names"] != names:
            raise ValueError(f"{m['source']}: joint names differ from {metas[0]['source']}")
    sm, ss, smin, smax = _combine([m["moments"]["state"] for m in metas])
    am, asd, amin, amax = _combine([m["moments"]["action"] for m in metas])
    return NormStats(list(names), sm, np.maximum(ss, std_floor), am, np.maximum(asd, std_floor),
                     smin, smax, amin, amax)


# ---------------------------------------------------------------- work items

def _init_worker():
    import torch
    torch.set_num_threads(1)


def _resize_block(h5path: str, cam: str, src_idx: np.ndarray, mm_path: str, offset: int,
                  image_size) -> int:
    """Read frames src_idx (sorted, unique) of one camera, resize, write rows offset.. of mm."""
    import torch
    with h5py.File(h5path, "r") as f:
        ds = f[f"images/{cam}/data"]
        raw = ds[src_idx[0]:src_idx[-1] + 1][src_idx - src_idx[0]]
    x = resize_frames(raw, tuple(image_size)).mul_(255.0).round_().clamp_(0, 255)
    mm = np.load(mm_path, mmap_mode="r+")
    mm[offset:offset + len(src_idx)] = x.to(torch.uint8).numpy()
    mm.flush()
    del mm
    return len(src_idx)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _plan_episode(path, cache_root, image_size, cameras, force):
    """Create/refresh the episode's lowdim + metadata; return (dir, meta, tasks, finals)."""
    path = Path(path).resolve()
    d = cache_dir_for(path, cache_root, image_size)
    meta_p = d / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() and not force else None
    if meta is None:
        arr = load_episode_arrays(path)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "lowdim.npz", t_ns=arr.t_ns, state=arr.state, action=arr.action)
        st = path.stat()
        attrs = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                 for k, v in arr.attrs.items()}
        meta = {"cache_version": CACHE_VERSION, "source": str(path), "size": st.st_size,
                "mtime_ns": st.st_mtime_ns, "length": int(len(arr.t_ns)),
                "image_size": [int(x) for x in image_size], "joint_names": arr.joint_names,
                "cameras": {}, "attrs": attrs,
                "moments": {"state": _moments(arr.state), "action": _moments(arr.action)}}
    tasks, finals = [], []
    todo = [c for c in cameras if c not in meta["cameras"]]
    if todo:
        with h5py.File(path, "r") as f:
            for cam in todo:
                key = f"aligned/images/{cam}/index"
                if key not in f or f"images/{cam}/data" not in f:
                    raise ValueError(f"{path}: camera {cam!r} not recorded")
                idx = f[key][:]
                uniq, lut = np.unique(idx, return_inverse=True)
                tmp = d / f"{cam}.partial.npy"
                h, w = (int(x) for x in image_size)
                np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                          shape=(len(uniq), 3, h, w)).flush()
                for off in range(0, len(uniq), BLOCK):
                    tasks.append((str(path), cam, uniq[off:off + BLOCK], str(tmp), off,
                                  tuple(image_size)))
                finals.append((cam, tmp, lut.astype(np.int32), int(len(uniq))))
    return d, meta, tasks, finals


def _image_stats(mm_path: Path) -> dict:
    mm = np.load(mm_path, mmap_mode="r")
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for i in range(0, len(mm), 256):
        x = np.asarray(mm[i:i + 256], dtype=np.float64) / 255.0
        s += x.sum((0, 2, 3))
        s2 += (x ** 2).sum((0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
    mean = s / max(n, 1)
    return {"mean": mean.tolist(), "std": np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 0)).tolist()}


def prepare_cache(paths, cache_root, image_size=(128, 160), cameras=(HEAD_CAMERA,),
                  workers: int = 0, force: bool = False, verbose: bool = False) -> list[Path]:
    """Build (or reuse) the cache for each episode; returns the cache dirs in input order."""
    t0 = time.time()
    plans = [_plan_episode(p, cache_root, image_size, tuple(cameras), force) for p in paths]
    tasks = [t for _, _, ts, _ in plans for t in ts]
    if tasks:
        if workers > 0:
            import concurrent.futures as cf
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods()
                                 else "spawn")
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                        initializer=_init_worker) as pool:
                for _ in pool.map(_resize_block, *zip(*tasks)):
                    pass
        else:
            for t in tasks:
                _resize_block(*t)
    out = []
    for d, meta, _, finals in plans:
        for cam, tmp, lut, n in finals:
            os.replace(tmp, d / f"{cam}.npy")
            np.save(d / f"{cam}_lut.npy", lut)
            meta["cameras"][cam] = {"unique_frames": n, **_image_stats(d / f"{cam}.npy")}
        if finals or not (d / "meta.json").exists():
            _write_json(d / "meta.json", meta)
        out.append(d)
    if verbose:
        n_frames = sum(f[3] for _, _, _, fs in plans for f in fs)
        print(f"[cache] {len(out)} episodes, {n_frames} frames decoded in {time.time() - t0:.1f}s "
              f"-> {cache_root}", flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="+")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--image-size", nargs=2, type=int, default=(128, 160), metavar=("H", "W"))
    p.add_argument("--camera", default=HEAD_CAMERA)
    p.add_argument("--extra-cameras", nargs="*", default=())
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    p.add_argument("--std-floor", type=float, default=1e-2)
    a = p.parse_args(argv)
    cams = (a.camera, *a.extra_cameras)
    dirs = prepare_cache(a.episodes, a.cache_dir, tuple(a.image_size), cams, a.workers,
                         a.force, verbose=True)
    load_cached_stats(dirs, a.std_floor).save(Path(a.cache_dir) / "norm_stats.json")
    size = sum(f.stat().st_size for d in dirs for f in d.iterdir())
    print(f"[cache] {size / 1e6:.1f} MB in {len(dirs)} dirs; norm_stats.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
