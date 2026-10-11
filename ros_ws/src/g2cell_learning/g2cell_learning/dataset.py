"""Torch dataset over g2cell_recorder episode HDF5 files (schema: g2cell_recorder/episode.py).

Everything is read from the 30 Hz ``/aligned`` view:

* proprio  = ``/aligned/joint_states/positions`` restricted to the joints that
  ``/joint_command`` commands, selected **by name** (joint_states has 46 joints in a different
  order; joint_command has the 16 arm + gripper joints).
* action   = ``/aligned/joint_command/actions`` (latest command at each step, forward-filled).
  Joints not yet commanded at the start of an episode (leading NaN) take the measured position,
  i.e. "hold where you are".
* images   = ``/images/<cam>/data[ /aligned/images/<cam>/index ]``, resized, float in [0, 1].

Item ``t`` holds the observation at aligned step ``t`` and the ACT-style chunk
``action[t + offset : t + offset + K]``. Steps past the episode end repeat the last action and are
flagged in ``action_is_pad``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

HEAD_CAMERA = "head_front_rgb"


def _names(attr) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in attr]


@dataclass
class EpisodeArrays:
    path: str
    t_ns: np.ndarray            # [T] int64
    joint_names: list[str]      # commanded joint names (action/state column order)
    state: np.ndarray           # [T, J] float64
    action: np.ndarray          # [T, J] float64
    attrs: dict = field(default_factory=dict)


def load_episode_arrays(path, joint_names: Sequence[str] | None = None) -> EpisodeArrays:
    """Low-dim aligned arrays of one episode, with state columns matched to command names."""
    path = str(path)
    with h5py.File(path, "r") as f:
        if "aligned/joint_command/actions" not in f or "joint_command" not in f:
            raise ValueError(f"{path}: no /aligned/joint_command/actions (episode has no "
                             "joint_command actions; was it recorded without teleop?)")
        cmd_names = _names(f["joint_command"].attrs["names"])
        st_names = _names(f["joint_states"].attrs["names"])
        act = f["aligned/joint_command/actions"][:].astype(np.float64)
        pos = f["aligned/joint_states/positions"][:].astype(np.float64)
        t_ns = f["aligned/t_ns"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    names = list(joint_names) if joint_names is not None else cmd_names
    missing = [n for n in names if n not in st_names or n not in cmd_names]
    if missing:
        raise ValueError(f"{path}: joints {missing} missing from joint_states or joint_command")
    state = pos[:, [st_names.index(n) for n in names]]
    action = act[:, [cmd_names.index(n) for n in names]]
    nan = np.isnan(action)
    action[nan] = state[nan]          # uncommanded so far -> hold measured position
    state = np.nan_to_num(state)
    action = np.nan_to_num(action)
    return EpisodeArrays(path, t_ns, names, state, action, attrs)


# ---------------------------------------------------------------- normalisation

@dataclass
class NormStats:
    joint_names: list[str]
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None

    def normalize_state(self, x):
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, x):
        return (x - self.action_mean) / self.action_std

    def unnormalize_action(self, x):
        return np.asarray(x) * self.action_std + self.action_mean

    def to_dict(self) -> dict:
        out = {"joint_names": list(self.joint_names)}
        for k in ("state_mean", "state_std", "action_mean", "action_std",
                  "state_min", "state_max", "action_min", "action_max"):
            v = getattr(self, k)
            out[k] = None if v is None else np.asarray(v, dtype=float).tolist()
        return out

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        kw = {k: (None if v is None else np.asarray(v, dtype=np.float64))
              for k, v in d.items() if k != "joint_names"}
        return cls(joint_names=list(d["joint_names"]), **kw)

    @classmethod
    def load(cls, path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text()))


def compute_norm_stats(paths, std_floor: float = 1e-2,
                       joint_names: Sequence[str] | None = None) -> NormStats:
    """Mean/std/min/max of state and action over all aligned steps of all episodes.

    Joints that never move (e.g. the idle arm in a one-arm demo) get std = ``std_floor`` so the
    normalised value stays ~0 instead of blowing up.
    """
    eps = [load_episode_arrays(p, joint_names) for p in paths]
    names = eps[0].joint_names
    for e in eps[1:]:
        if e.joint_names != names:
            raise ValueError(f"{e.path}: joint names differ from {eps[0].path}")
    s = np.concatenate([e.state for e in eps])
    a = np.concatenate([e.action for e in eps])
    return NormStats(names, s.mean(0), np.maximum(s.std(0), std_floor),
                     a.mean(0), np.maximum(a.std(0), std_floor),
                     s.min(0), s.max(0), a.min(0), a.max(0))


# ---------------------------------------------------------------- dataset

@dataclass
class DatasetConfig:
    image_size: tuple[int, int] = (128, 160)        # (H, W) after resize
    camera: str = HEAD_CAMERA
    extra_cameras: tuple[str, ...] = ()              # e.g. ("free_rgb",)
    chunk_size: int = 20                             # K
    action_offset: int = 1                           # chunk starts at t + offset ("next" actions)
    normalize: bool = True
    cache_images: bool = True                        # decode + resize every frame once, in RAM
    std_floor: float = 1e-2

    @property
    def cameras(self) -> tuple[str, ...]:
        return (self.camera, *self.extra_cameras)


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 [N,H,W,3] -> float32 [N,3,h,w] in [0,1] (bilinear, antialiased)."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False,
                          antialias=True)
    return x.clamp_(0.0, 1.0)


class EpisodeDataset(Dataset):
    """Concatenation of the aligned steps of one or more episodes."""

    def __init__(self, paths, config: DatasetConfig | None = None,
                 stats: NormStats | None = None):
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise ValueError("no episode files given")
        self.cfg = config or DatasetConfig()
        self.episodes = [load_episode_arrays(p) for p in self.paths]
        names = self.episodes[0].joint_names
        for e in self.episodes[1:]:
            if e.joint_names != names:
                raise ValueError(f"{e.path}: joint names differ")
        self.joint_names = names
        if stats is None and self.cfg.normalize:
            stats = compute_norm_stats(self.paths, self.cfg.std_floor)
        self.stats = stats
        self._index = [(ei, t) for ei, e in enumerate(self.episodes) for t in range(len(e.t_ns))]
        self._img_index: list[dict[str, np.ndarray]] = []
        with_files = [h5py.File(p, "r") for p in self.paths]
        try:
            for f in with_files:
                d = {}
                for cam in self.cfg.cameras:
                    key = f"aligned/images/{cam}/index"
                    if key not in f:
                        raise ValueError(f"{f.filename}: camera {cam!r} not recorded")
                    d[cam] = f[key][:]
                self._img_index.append(d)
            self._cache: list[dict[str, torch.Tensor]] | None = None
            if self.cfg.cache_images:
                self._cache = [self._load_all_frames(f, d) for f, d in
                               zip(with_files, self._img_index)]
        finally:
            for f in with_files:
                f.close()
        self._files: dict[int, h5py.File] = {}

    # -- images
    def _load_all_frames(self, f, idx_by_cam) -> dict[str, torch.Tensor]:
        out = {}
        for cam, idx in idx_by_cam.items():
            ds = f[f"images/{cam}/data"]
            uniq = np.unique(idx)
            frames = []
            for start in range(0, len(uniq), 32):   # bounded RAM for 720p frames
                sel = uniq[start:start + 32]
                frames.append(resize_frames(ds[sel[0]:sel[-1] + 1][sel - sel[0]],
                                            self.cfg.image_size))
            small = torch.cat(frames)
            lut = {int(u): i for i, u in enumerate(uniq)}
            # store frames per source frame plus a per-step lookup
            out[cam] = (small, torch.tensor([lut[int(i)] for i in idx]))
        return out

    def _file(self, ei: int) -> h5py.File:
        f = self._files.get(ei)
        if f is None:  # opened lazily per process (h5py handles do not survive fork)
            f = self._files[ei] = h5py.File(self.paths[ei], "r")
        return f

    def _frame(self, ei: int, t: int, cam: str) -> torch.Tensor:
        if self._cache is not None:
            frames, lut = self._cache[ei][cam]
            return frames[lut[t]]
        src = int(self._img_index[ei][cam][t])
        raw = self._file(ei)[f"images/{cam}/data"][src]
        return resize_frames(raw[None], self.cfg.image_size)[0]

    # -- torch API
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, t = self._index[i]
        ep = self.episodes[ei]
        n, k, off = len(ep.t_ns), self.cfg.chunk_size, self.cfg.action_offset
        steps = np.arange(t + off, t + off + k)
        pad = steps >= n
        act = ep.action[np.minimum(steps, n - 1)]
        state = ep.state[t]
        if self.cfg.normalize:
            act = self.stats.normalize_action(act)
            state = self.stats.normalize_state(state)
        item = {
            "observation.state": torch.as_tensor(state, dtype=torch.float32),
            "action": torch.as_tensor(act, dtype=torch.float32),
            "action_is_pad": torch.as_tensor(pad, dtype=torch.bool),
            "episode_index": torch.tensor(ei, dtype=torch.long),
            "frame_index": torch.tensor(t, dtype=torch.long),
        }
        for cam in self.cfg.cameras:
            item[f"observation.images.{cam}"] = self._frame(ei, t, cam)
        return item

    def __del__(self):
        for f in getattr(self, "_files", {}).values():
            try:
                f.close()
            except Exception:
                pass


# ---------------------------------------------------------------- cached dataset

class CachedEpisodeDataset(Dataset):
    """Same items as EpisodeDataset, read from prepare_cache.py output (memory-mapped uint8).

    Images come out as **uint8** [3,h,w] (4x less to copy, pin and transfer); the policy
    converts them to float on the device. Memmaps are opened lazily in each process, so the
    dataset pickles cheaply into DataLoader workers, which then share frames via the page cache.
    """

    def __init__(self, cache_dirs, config: DatasetConfig | None = None,
                 stats: NormStats | None = None):
        self.dirs = [Path(d) for d in cache_dirs]
        if not self.dirs:
            raise ValueError("no cache dirs given")
        self.cfg = config or DatasetConfig()
        self.metas = [json.loads((d / "meta.json").read_text()) for d in self.dirs]
        names = self.metas[0]["joint_names"]
        for d, m in zip(self.dirs, self.metas):
            if m["joint_names"] != names:
                raise ValueError(f"{d}: joint names differ")
            if list(m["image_size"]) != list(self.cfg.image_size):
                raise ValueError(f"{d}: cached at {m['image_size']}, config wants "
                                 f"{list(self.cfg.image_size)}")
            missing = [c for c in self.cfg.cameras if c not in m["cameras"]]
            if missing:
                raise ValueError(f"{d}: camera(s) {missing} not in the cache "
                                 "(run prepare_cache with them)")
        self.joint_names = list(names)
        self.paths = [m["source"] for m in self.metas]
        if stats is None and self.cfg.normalize:
            from .prepare_cache import load_cached_stats
            stats = load_cached_stats(self.dirs, self.cfg.std_floor)
        self.stats = stats
        self._state, self._action, self._lut = [], [], []
        for d in self.dirs:
            low = np.load(d / "lowdim.npz")
            s, a = low["state"], low["action"]
            if self.cfg.normalize:
                s, a = stats.normalize_state(s), stats.normalize_action(a)
            self._state.append(np.ascontiguousarray(s, dtype=np.float32))
            self._action.append(np.ascontiguousarray(a, dtype=np.float32))
            self._lut.append({c: np.load(d / f"{c}_lut.npy") for c in self.cfg.cameras})
        lens = [len(s) for s in self._state]
        self._ep = np.repeat(np.arange(len(lens)), lens).astype(np.int64)
        self._t = np.concatenate([np.arange(n) for n in lens]).astype(np.int64)
        self._mms: dict = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mms"] = {}          # never pickle memmaps (that would copy every frame)
        return d

    def _mm(self, ei: int, cam: str) -> np.ndarray:
        m = self._mms.get((ei, cam))
        if m is None:
            m = self._mms[(ei, cam)] = np.load(self.dirs[ei] / f"{cam}.npy", mmap_mode="r")
        return m

    def __len__(self) -> int:
        return len(self._ep)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, t = int(self._ep[i]), int(self._t[i])
        act_all = self._action[ei]
        n, k, off = len(act_all), self.cfg.chunk_size, self.cfg.action_offset
        steps = np.arange(t + off, t + off + k)
        pad = steps >= n
        item = {
            "observation.state": torch.from_numpy(self._state[ei][t].copy()),
            "action": torch.from_numpy(act_all[np.minimum(steps, n - 1)]),
            "action_is_pad": torch.from_numpy(pad),
            "episode_index": torch.tensor(ei, dtype=torch.long),
            "frame_index": torch.tensor(t, dtype=torch.long),
        }
        for cam in self.cfg.cameras:
            row = int(self._lut[ei][cam][t])
            item[f"observation.images.{cam}"] = torch.from_numpy(np.array(self._mm(ei, cam)[row]))
        return item
