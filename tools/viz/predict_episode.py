"""Open-loop policy predictions for one episode -> .npz (run with .venv-learn, CPU).

Same convention as g2cell_learning.eval_offline: the policy sees the recorded observation at
aligned step t; its first predicted action targets the recorded command at t + action_offset.
Output arrays are indexed by aligned step (NaN where no prediction targets that step):

    pred0        [T, J] predicted command (rad), step-0 of the chunk, shifted by action_offset
    recorded     [T, J] recorded command (dataset convention: NaN -> measured state)
    joint_names  [J]
    meta_json    checkpoint, step, policy, action_offset, action_mse

    CUDA_VISIBLE_DEVICES="" .venv-learn/bin/python tools/viz/predict_episode.py \
        --ckpt output/train/smoke_act/checkpoints/last.pt --episode EP.hdf5 --out pred.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros_ws/src/g2cell_learning"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from g2cell_learning.dataset import EpisodeDataset, NormStats  # noqa: E402
from g2cell_learning.eval_offline import predict_episode  # noqa: E402
from g2cell_learning.train import TrainConfig, build_from_checkpoint  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{**ckpt["config"], "image_size": tuple(ckpt["config"]["image_size"]),
                         "extra_cameras": tuple(ckpt["config"]["extra_cameras"])})
    stats = NormStats.from_dict(ckpt["norm_stats"])
    model = build_from_checkpoint(ckpt).eval()
    ds = EpisodeDataset([a.episode], cfg.dataset_config(), stats)
    if ds.joint_names != ckpt["joint_names"]:
        raise SystemExit(f"{a.episode}: joint names differ from the checkpoint")
    pred_n, _, _ = predict_episode(model, ds)
    pred = stats.unnormalize_action(pred_n)                     # [T,K,J]
    rec = ds.episodes[0].action
    T, off = len(rec), cfg.action_offset
    pred0 = np.full_like(rec, np.nan)
    pred0[off:] = pred[: T - off, 0] if off else pred[:, 0]
    ok = ~np.isnan(pred0).any(1)
    mse = float(((pred0[ok] - rec[ok]) ** 2).mean())
    meta = {"checkpoint": str(a.ckpt), "step": int(ckpt["step"]), "policy": cfg.policy,
            "action_offset": off, "action_mse": mse,
            "train_episodes": list(ckpt.get("split", {}).get("train", []))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, pred0=pred0, recorded=rec, joint_names=np.array(ds.joint_names),
             meta_json=json.dumps(meta))
    print(json.dumps({"out": a.out, "T": T, "action_mse": mse}))


if __name__ == "__main__":
    main()
