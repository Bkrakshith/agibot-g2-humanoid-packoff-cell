"""Per-batch dataset card (JSON + Markdown) for a set of episode HDF5 files.

Reads only file attributes and small arrays, never image data, so it is fast on big batches.

    python -m g2cell_learning.dataset_card output/episodes/*/*.hdf5 -o output/datasets/card \
        --batch-name stage1
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def _names(attr):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in attr]


def _py(v):
    return v.item() if hasattr(v, "item") else v


def episode_summary(path) -> dict:
    with h5py.File(path, "r") as f:
        a = {k: _py(v) for k, v in f.attrs.items()}
        has_actions = "aligned/joint_command/actions" in f
        cams = {}
        for grp in ("images", "depth"):
            if grp in f:
                for cam in f[grp]:
                    ds = f[f"{grp}/{cam}/data"]
                    cams[f"{grp}/{cam}" if grp == "depth" else cam] = {
                        "shape": list(ds.shape[1:]), "frames": int(ds.shape[0]),
                        "rate_hz": float(_py(f[f"{grp}/{cam}"].attrs.get("rate_hz", 0.0)))}
        return {
            "path": str(path),
            "scene": str(a.get("scene", "")),
            "seed": int(a.get("seed", -1)),
            "success": str(a.get("success", "unknown")),
            "duration_s": float(a.get("duration_s", 0.0)),
            "aligned_rate_hz": float(a.get("aligned_rate_hz", 0.0)),
            "aligned_frames": int(f["aligned/t_ns"].shape[0]) if "aligned/t_ns" in f else 0,
            "stream_rates_hz": json.loads(a.get("stream_rates_json", "{}") or "{}"),
            "has_actions": bool(has_actions),
            "action_joint_names": _names(f["joint_command"].attrs["names"])
            if "joint_command" in f else [],
            "num_state_joints": len(_names(f["joint_states"].attrs["names"]))
            if "joint_states" in f else 0,
            "cameras": cams,
            "schema_version": int(a.get("schema_version", -1)),
        }


def build_card(paths, batch_name: str = "") -> dict:
    eps = [episode_summary(p) for p in paths]
    labelled = [e for e in eps if e["success"] in ("true", "false")]
    durs = np.array([e["duration_s"] for e in eps]) if eps else np.zeros(0)
    rate_keys = sorted({k for e in eps for k in e["stream_rates_hz"]})
    rates = {k: [e["stream_rates_hz"][k] for e in eps if k in e["stream_rates_hz"]]
             for k in rate_keys}
    cams = {}
    for e in eps:
        for c, v in e["cameras"].items():
            cams.setdefault(c, {"shape": v["shape"], "episodes": 0})
            cams[c]["episodes"] += 1
    act_names = next((e["action_joint_names"] for e in eps if e["action_joint_names"]), [])
    return {
        "batch_name": batch_name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "num_episodes": len(eps),
        "num_episodes_with_actions": sum(e["has_actions"] for e in eps),
        "success": {
            "labelled": len(labelled),
            "successes": sum(e["success"] == "true" for e in labelled),
            "rate": (sum(e["success"] == "true" for e in labelled) / len(labelled))
            if labelled else None,
            "unknown": len(eps) - len(labelled),
        },
        "duration_s": {
            "total": float(durs.sum()), "mean": float(durs.mean()) if len(durs) else 0.0,
            "min": float(durs.min()) if len(durs) else 0.0,
            "max": float(durs.max()) if len(durs) else 0.0,
        },
        "total_aligned_frames": sum(e["aligned_frames"] for e in eps),
        "aligned_rate_hz": sorted({e["aligned_rate_hz"] for e in eps}),
        "stream_rates_hz": {
            "mean": {k: float(np.mean(v)) for k, v in rates.items()},
            "min": {k: float(np.min(v)) for k, v in rates.items()},
            "max": {k: float(np.max(v)) for k, v in rates.items()},
        },
        "scenes": dict(sorted(Counter(e["scene"] for e in eps).items())),
        "seeds": [e["seed"] for e in eps],
        "cameras": cams,
        "action_dim": len(act_names),
        "action_joint_names": act_names,
        "schema_versions": sorted({e["schema_version"] for e in eps}),
        "episodes": eps,
    }


def card_markdown(card: dict) -> str:
    s = card["success"]
    rate = "n/a (no episode labelled)" if s["rate"] is None else \
        f"{s['rate']:.1%} ({s['successes']}/{s['labelled']} labelled; {s['unknown']} unknown)"
    d = card["duration_s"]
    lines = [
        f"# Dataset card: {card['batch_name'] or 'unnamed batch'}",
        "",
        f"Created {card['created']}.",
        "",
        "| field | value |", "|---|---|",
        f"| Episodes | {card['num_episodes']} ({card['num_episodes_with_actions']} with actions) |",
        f"| Success rate | {rate} |",
        f"| Duration | total {d['total']:.1f} s, mean {d['mean']:.1f} s "
        f"(min {d['min']:.1f}, max {d['max']:.1f}) |",
        f"| Aligned frames | {card['total_aligned_frames']} at "
        f"{', '.join(f'{r:g}' for r in card['aligned_rate_hz'])} Hz |",
        f"| Scenes | {', '.join(f'{k} ({v})' for k, v in card['scenes'].items())} |",
        f"| Seeds | {', '.join(map(str, card['seeds']))} |",
        f"| Action | {card['action_dim']} joints: {', '.join(card['action_joint_names'])} |",
        "",
        "## Stream rates (Hz, mean over episodes)", "",
        "| stream | mean | min | max |", "|---|---|---|---|",
    ]
    r = card["stream_rates_hz"]
    for k in r["mean"]:
        lines.append(f"| {k} | {r['mean'][k]:.1f} | {r['min'][k]:.1f} | {r['max'][k]:.1f} |")
    lines += ["", "## Cameras", "", "| camera | frame shape | episodes |", "|---|---|---|"]
    for c, v in card["cameras"].items():
        lines.append(f"| {c} | {'x'.join(map(str, v['shape']))} | {v['episodes']} |")
    lines += ["", "## Episodes", "",
              "| file | scene | seed | success | duration s | aligned frames | actions |",
              "|---|---|---|---|---|---|---|"]
    for e in card["episodes"]:
        lines.append(f"| {Path(e['path']).name} | {e['scene']} | {e['seed']} | {e['success']} | "
                     f"{e['duration_s']:.1f} | {e['aligned_frames']} | "
                     f"{'yes' if e['has_actions'] else 'no'} |")
    return "\n".join(lines) + "\n"


def write_card(card: dict, out_stem) -> Path:
    """Writes <out_stem>.json and <out_stem>.md; returns the .md path."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(card, indent=2))
    md = out_stem.with_suffix(".md")
    md.write_text(card_markdown(card))
    return md


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="+")
    p.add_argument("-o", "--out", required=True, help="output stem (writes .json and .md)")
    p.add_argument("--batch-name", default="")
    a = p.parse_args(argv)
    md = write_card(build_card(a.episodes, a.batch_name), a.out)
    print(md.read_text())


if __name__ == "__main__":
    main()
