"""Offline evaluation on held-out episodes: action MSE + open-loop replay plots.

"Open loop" = the policy sees the *recorded* observations at every step (it never drives the
robot), and its first predicted action is compared with the recorded command. This measures
imitation error only; it says nothing about closed-loop success (that needs the sim).

    python -m g2cell_learning.eval_offline --ckpt run/checkpoints/last.pt \
        --episodes ep_val.hdf5 --out-dir run/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import EpisodeDataset, NormStats
from .train import TrainConfig, build_from_checkpoint


def _pick_joints(names, actions, n_arm=4):
    """Both grippers + the n_arm arm joints that move most in the recorded actions."""
    grip = [i for i, n in enumerate(names) if "gripper" in n]
    arm = [i for i in np.argsort(-actions.std(0)) if i not in grip][:n_arm]
    return grip + sorted(int(i) for i in arm)


@torch.no_grad()
def predict_episode(model, ds: EpisodeDataset, batch_size=64):
    """Normalised predictions [T,K,J], targets [T,K,J], pad masks [T,K] for every frame."""
    preds, tgts, pads = [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    for b in loader:
        preds.append(model(b))
        tgts.append(b["action"])
        pads.append(b["action_is_pad"])
    return torch.cat(preds).numpy(), torch.cat(tgts).numpy(), torch.cat(pads).numpy()


def plot_replay(path, names, joints, recorded, predicted, dt, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(joints)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows), sharex=True, squeeze=False)
    t = np.arange(len(recorded)) * dt
    for ax, j in zip(axes.flat, joints):
        ax.plot(t, recorded[:, j], color="#1f4e79", lw=1.8, label="recorded command")
        ax.plot(t, predicted[:, j], color="#d9822b", lw=1.2, ls="--", label="predicted (step 0)")
        ax.set_title(names[j], fontsize=9)
        ax.set_ylabel("rad", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]", fontsize=8)
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(ckpt_path, episodes, out_dir, plot_joints=None, batch_size=64) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{**ckpt["config"], "image_size": tuple(ckpt["config"]["image_size"]),
                         "extra_cameras": tuple(ckpt["config"]["extra_cameras"])})
    stats = NormStats.from_dict(ckpt["norm_stats"])
    model = build_from_checkpoint(ckpt).eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ckpt["joint_names"]
    dt = 1.0 / 30.0

    sq_first, sq_chunk, n_first, n_chunk = 0.0, 0.0, 0, 0
    per_joint = np.zeros(len(names))
    per_ep = {}
    for ep in episodes:
        ds = EpisodeDataset([ep], cfg.dataset_config(), stats)
        if ds.joint_names != names:
            raise ValueError(f"{ep}: joint names differ from the checkpoint")
        pred_n, tgt_n, pad = predict_episode(model, ds, batch_size)
        pred = stats.unnormalize_action(pred_n)
        tgt = stats.unnormalize_action(tgt_n)
        valid = ~pad
        e2 = (pred - tgt) ** 2                                  # [T,K,J] in rad^2
        first_valid = valid[:, 0]
        ep_first = e2[first_valid, 0]                           # [T',J]
        sq_first += ep_first.sum()
        n_first += ep_first.size
        per_joint += ep_first.sum(0)
        sq_chunk += e2[valid].sum()
        n_chunk += e2[valid].size
        per_ep[ep] = {"action_mse": float(ep_first.mean()), "frames": int(len(ds))}

        rec = ds.episodes[0].action
        off = cfg.action_offset
        # predicted step-0 action at frame t targets recorded action[t+off]
        T = len(rec)
        pred0 = np.full_like(rec, np.nan)
        pred0[off:] = pred[: T - off, 0] if off else pred[:, 0]
        joints = plot_joints or _pick_joints(names, rec)
        plot_replay(out / f"replay_{Path(ep).stem}.png", names, joints, rec, pred0, dt,
                    f"Open-loop replay: {Path(ep).name} ({ckpt['config']['policy']}, "
                    f"step {ckpt['step']}); action MSE {per_ep[ep]['action_mse']:.2e} rad²")

    metrics = {
        "checkpoint": str(ckpt_path), "step": int(ckpt["step"]),
        "episodes": list(map(str, episodes)),
        "action_mse": float(sq_first / max(n_first, 1)),      # first predicted step, rad^2
        "chunk_mse": float(sq_chunk / max(n_chunk, 1)),       # all non-padded chunk steps
        "per_joint_mse": dict(zip(names, (per_joint / max(n_first // len(names), 1)).tolist())),
        "per_episode": per_ep,
        "train_equals_val": bool(set(map(str, episodes)) & set(ckpt["split"]["train"])),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", nargs="+", help="default: the checkpoint's val split")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--joints", nargs="*", type=int, help="action indices to plot")
    a = p.parse_args(argv)
    eps = a.episodes
    if not eps:
        eps = torch.load(a.ckpt, map_location="cpu", weights_only=False)["split"]["val"]
    m = evaluate(a.ckpt, eps, a.out_dir, a.joints or None)
    print(json.dumps({k: m[k] for k in ("action_mse", "chunk_mse", "train_equals_val")},
                     indent=2))


if __name__ == "__main__":
    main()
