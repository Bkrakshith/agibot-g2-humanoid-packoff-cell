#!/usr/bin/env python3
"""Episode HDF5 (+ optional policy checkpoint) -> replay .npz for replay_node.py (RViz).

Run with the learning venv (h5py; torch only when --checkpoint is given):

    CUDA_VISIBLE_DEVICES= ../.venv-learn/bin/python tools/viz/export_replay.py \
        output/episodes/stage1_teleop/stage1_teleop.hdf5 \
        --checkpoint output/train/smoke_act -o /tmp/replay.npz

The .npz holds plain numpy arrays only (readable by the host /usr/bin/python3 numpy):

    js_names [J] str           joint_states names (46)
    js_t     [N] float64 s     raw /joint_states time, relative to the aligned start
    js_pos   [N,J] float64     raw /joint_states positions (~100 Hz)
    al_t     [M] float64 s     /aligned clock (30 Hz), relative to its first step
    al_pos   [M,J] float64     /aligned/joint_states/positions (NaN forward-filled, NaN->0)
    cmd_names [K] str          commanded joint names (16)
    al_cmd   [M,K] float64     recorded command (NaN -> measured position, as in training)
    has_policy  bool
    pol_pos  [M,J] float64     recorded al_pos with the K commanded joints replaced by the
                               policy's step-0 prediction for that step (open loop, like
                               eval_offline: prediction made at step t-offset targets step t;
                               the first `offset` steps keep the recorded command)
    pol_cmd  [M,K] float64     the prediction itself
    meta_json str              episode / checkpoint info
Mimic gripper joints are NOT fixed up here; replay_node applies the URDF mimic rules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # g2_packoff_cell/
sys.path.insert(0, str(ROOT / "ros_ws/src/g2cell_learning"))


def _names(attr):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in attr]


def _ffill(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    for j in range(a.shape[1]):
        col = a[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            col[:] = 0.0
            continue
        idx = np.where(ok, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        col[: np.argmax(ok)] = col[np.argmax(ok)]      # leading NaN -> first valid value
    return a


def _resolve_ckpt(p: str) -> Path:
    p = Path(p)
    if p.is_dir():
        for c in ("checkpoints/best.pt", "checkpoints/last.pt", "best.pt", "last.pt"):
            if (p / c).exists():
                return p / c
        raise FileNotFoundError(f"no best.pt/last.pt under {p}")
    return p


def predict(ckpt_path: Path, episode: str, batch_size: int = 64):
    """Step-0 predicted command per aligned step [M,K] (unnormalised, rad) + info."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from g2cell_learning.dataset import EpisodeDataset, NormStats
    from g2cell_learning.eval_offline import predict_episode
    from g2cell_learning.train import TrainConfig, build_from_checkpoint

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{**ckpt["config"], "image_size": tuple(ckpt["config"]["image_size"]),
                         "extra_cameras": tuple(ckpt["config"]["extra_cameras"])})
    stats = NormStats.from_dict(ckpt["norm_stats"])
    model = build_from_checkpoint(ckpt).eval()
    ds = EpisodeDataset([episode], cfg.dataset_config(), stats)
    if ds.joint_names != ckpt["joint_names"]:
        raise ValueError(f"{episode}: commanded joints differ from the checkpoint")
    pred_n, _, _ = predict_episode(model, ds, batch_size)
    pred = stats.unnormalize_action(pred_n)             # [M,Kchunk,J]
    rec = ds.episodes[0].action
    off = cfg.action_offset
    M = len(rec)
    pred0 = rec.copy()
    pred0[off:] = pred[: M - off, 0] if off else pred[:, 0]
    mse = float(((pred0[off:] - rec[off:]) ** 2).mean())
    info = {"checkpoint": str(ckpt_path), "step": int(ckpt["step"]),
            "policy": ckpt["config"]["policy"], "action_offset": off,
            "action_mse_rad2": mse,
            "trained_on_this_episode": str(episode) in map(str, ckpt["split"]["train"])
            or Path(episode).name in [Path(p).name for p in ckpt["split"]["train"]]}
    return ds.joint_names, rec, pred0, info


def export(episode: str, out: str, checkpoint: str | None = None) -> dict:
    with h5py.File(episode, "r") as f:
        js_names = _names(f["joint_states"].attrs["names"])
        js_t_ns = f["joint_states/t_ns"][:] if "joint_states/t_ns" in f else None
        js_pos = f["joint_states/positions"][:].astype(np.float64)
        al_t_ns = f["aligned/t_ns"][:]
        al_pos = f["aligned/joint_states/positions"][:].astype(np.float64)
        cmd_names = _names(f["joint_command"].attrs["names"]) if "joint_command" in f else []
        al_cmd = (f["aligned/joint_command/actions"][:].astype(np.float64)
                  if "aligned/joint_command/actions" in f else np.zeros((len(al_t_ns), 0)))
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    if js_t_ns is None:
        raise ValueError(f"{episode}: /joint_states/t_ns missing")
    t0 = int(al_t_ns[0])
    js_t = (js_t_ns.astype(np.int64) - t0) / 1e9
    al_t = (al_t_ns.astype(np.int64) - t0) / 1e9
    js_pos = _ffill(js_pos)
    al_pos = _ffill(al_pos)
    # recorded command: NaN (not yet commanded) -> measured position (as in dataset.py)
    for k, n in enumerate(cmd_names):
        nan = np.isnan(al_cmd[:, k])
        al_cmd[nan, k] = al_pos[nan, js_names.index(n)]

    meta = {"episode": str(episode), "scene": str(attrs.get("scene", "")),
            "duration_s": float(al_t[-1]), "aligned_rate_hz": float(attrs.get("aligned_rate_hz", 30)),
            "success": str(attrs.get("success", ""))}
    data = dict(js_names=np.array(js_names), js_t=js_t, js_pos=js_pos, al_t=al_t,
                al_pos=al_pos, cmd_names=np.array(cmd_names), al_cmd=al_cmd,
                has_policy=np.array(False))
    if checkpoint:
        ck = _resolve_ckpt(checkpoint)
        names, rec, pred0, info = predict(ck, episode)
        if len(pred0) != len(al_t):
            raise ValueError(f"prediction length {len(pred0)} != aligned steps {len(al_t)}")
        pol_pos = al_pos.copy()
        for k, n in enumerate(names):
            pol_pos[:, js_names.index(n)] = pred0[:, k]
        pol_cmd = np.zeros_like(al_cmd)
        for k, n in enumerate(cmd_names):
            pol_cmd[:, k] = pred0[:, names.index(n)] if n in names else al_cmd[:, k]
        data.update(has_policy=np.array(True), pol_pos=pol_pos, pol_cmd=pol_cmd)
        meta["policy"] = info
    data["meta_json"] = np.array(json.dumps(meta))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", help="episode .hdf5")
    p.add_argument("-o", "--out", required=True, help="output .npz")
    p.add_argument("--checkpoint", help="run dir (uses checkpoints/best.pt|last.pt) or .pt file")
    a = p.parse_args(argv)
    meta = export(a.episode, a.out, a.checkpoint)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
