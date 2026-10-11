"""Batch tools for a generated episode folder (no ROS needed; host or container).

    datagen_batch status   output/episodes/pick_v1        # done / error / outcome counts, per-episode lines
    datagen_batch finalize output/episodes/pick_v1        # convert bag-only episodes, apply marks from episode.json
    datagen_batch card     output/episodes/pick_v1 --name pick_v1   # dataset_card.{json,md}

`card` reuses g2cell_learning.dataset_card for the HDF5 side and adds a "datagen" section from the
episode.json files (status, outcomes, labels/targets, randomisation ranges, timings).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from g2cell_datagen.store import DONE, EpisodeStore


def datagen_section(store: EpisodeStore) -> dict:
    recs = store.records()
    done = [r for r in recs if r.get("status") == DONE]
    rnd = [r["randomisation"] for r in done if r.get("randomisation")]

    def rng(vals):
        return [min(vals), max(vals)] if vals else []
    totals = [r.get("timings", {}).get("total_s") for r in done if r.get("timings", {}).get("total_s") is not None]
    return {
        "episodes": len(recs),
        "status": dict(sorted(Counter(r.get("status", "?") for r in recs).items())),
        "outcome": dict(sorted(Counter(r.get("outcome", "") for r in done).items())),
        "success": dict(sorted(Counter(r.get("success", "") for r in done).items())),
        "mode": dict(sorted(Counter(r.get("mode", "") for r in done).items())),
        "label": dict(sorted(Counter(x["bag"]["label"] for x in rnd).items())),
        "bag_id": dict(sorted(Counter(x["bag"]["id"] for x in rnd).items())),
        "target": dict(sorted(Counter(x["target"] for x in rnd).items())),
        "randomisation_ranges": {"x": rng([x["bag"]["x"] for x in rnd]), "y": rng([x["bag"]["y"] for x in rnd]),
                                 "yaw_deg": rng([x["bag"]["yaw_deg"] for x in rnd])},
        "mean_total_s": (sum(totals) / len(totals)) if totals else None,
        "episodes_detail": [{"name": r["name"], "seed": r.get("seed"), "status": r.get("status"),
                             "outcome": r.get("outcome"), "success": r.get("success"),
                             "bag": (r.get("randomisation") or {}).get("bag"),
                             "target": (r.get("randomisation") or {}).get("target"),
                             "total_s": r.get("timings", {}).get("total_s")} for r in recs],
    }


def datagen_markdown(dg: dict) -> str:
    def kv(d):
        return ", ".join(f"{k} ({v})" for k, v in d.items()) or "-"
    rr = dg["randomisation_ranges"]
    mean = "-" if dg["mean_total_s"] is None else f"{dg['mean_total_s']:.1f} s"

    def span(v, f="{:+.3f}"):
        return f"{f.format(v[0])} .. {f.format(v[1])}" if v else "-"
    lines = ["", "## Scripted generation (g2cell_datagen)", "",
             "| field | value |", "|---|---|",
             f"| episode.json files | {dg['episodes']} |",
             f"| Status | {kv(dg['status'])} |",
             f"| Outcome (done) | {kv(dg['outcome'])} |",
             f"| Success attr (done) | {kv(dg['success'])} |",
             f"| Mode | {kv(dg['mode'])} |",
             f"| Bag / label / target | {kv(dg['bag_id'])}; {kv(dg['label'])}; {kv(dg['target'])} |",
             f"| Bag x / y (m, cell frame) | {span(rr['x'])} / {span(rr['y'])} |",
             f"| Bag yaw (deg) | {span(rr['yaw_deg'], '{:+.1f}')} |",
             f"| Mean episode wall time | {mean} |",
             "", "| episode | seed | status | outcome | success | bag | x | y | yaw | target | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in dg["episodes_detail"]:
        b = e.get("bag") or {}
        lines.append(f"| {e['name']} | {e['seed']} | {e['status']} | {e['outcome'] or ''} | {e['success'] or ''} | "
                     f"{b.get('id', '')} {b.get('label', '')} | {b.get('x', '')} | {b.get('y', '')} | "
                     f"{b.get('yaw_deg', '')} | {e.get('target') or ''} | {e.get('total_s') or ''} |")
    return "\n".join(lines) + "\n"


def cmd_card(a) -> int:
    from g2cell_learning.dataset_card import build_card, card_markdown  # noqa: PLC0415
    store = EpisodeStore(a.batch_dir)
    h5s = sorted(str(p) for p in Path(a.batch_dir).glob("ep_s*/ep_s*.hdf5"))
    card = build_card(h5s, a.name or Path(a.batch_dir).name)
    card["datagen"] = datagen_section(store)
    stem = Path(a.out) if a.out else Path(a.batch_dir) / "dataset_card"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(card, indent=2))
    md = stem.with_suffix(".md")
    md.write_text(card_markdown(card) + datagen_markdown(card["datagen"]))
    print(md)
    return 0


def cmd_status(a) -> int:
    store = EpisodeStore(a.batch_dir)
    for r in store.records():
        b = (r.get("randomisation") or {}).get("bag", {})
        print(f"{r['name']}  {r.get('status', '?'):11s} {r.get('outcome', '') or '-':13s} "
              f"success={r.get('success', '') or '-':7s} {b.get('id', '')} {b.get('label', '')} "
              f"{(r.get('timings') or {}).get('total_s', '')}  {r.get('error', '')[:80]}")
    print(json.dumps(store.summary(), indent=2))
    return 0


def _mark(h5: Path, success: str) -> None:
    try:
        from g2cell_recorder.mark_episode import mark  # noqa: PLC0415
    except ImportError:          # host without the recorder on the path
        def mark(path, s):
            import h5py  # noqa: PLC0415
            with h5py.File(path, "r+") as f:
                f.attrs["success"] = s
    mark(str(h5), success)


def cmd_finalize(a) -> int:
    store = EpisodeStore(a.batch_dir)
    conv = shlex.split(a.converter)
    bad = 0
    for r in store.records():
        if r.get("status") != DONE:
            continue
        name, success = r["name"], r.get("success") or "unknown"
        h5, bag = store.hdf5_path(name), store.episode_dir(name) / "bag"
        if not h5.exists():
            if not bag.exists():
                print(f"{name}: no bag, no hdf5: skipped")
                continue
            cmd = conv + [str(bag), "-o", str(h5), "--scene", r.get("scene", "scene_g2cell"),
                          "--seed", str(r["seed"]), "--success", success]
            print(f"{name}: convert ({' '.join(cmd)})")
            if a.dry_run:
                continue
            if subprocess.run(cmd, check=False).returncode != 0 or not h5.exists():
                print(f"{name}: conversion FAILED", file=sys.stderr)
                bad += 1
                continue
        elif success in ("true", "false"):
            if a.dry_run:
                print(f"{name}: mark success={success}")
                continue
            _mark(h5, success)
        if not a.dry_run:
            r = store.load(name)
            r.setdefault("recording", {})
            r["recording"].update(hdf5=str(h5), finalized=True)
            store.write(name, r)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="datagen_batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("batch_dir")
    c = sub.add_parser("card")
    c.add_argument("batch_dir")
    c.add_argument("--name", default="")
    c.add_argument("-o", "--out", default="", help="output stem (default <batch_dir>/dataset_card)")
    f = sub.add_parser("finalize")
    f.add_argument("batch_dir")
    f.add_argument("--converter", default=f"{sys.executable} -m g2cell_recorder.bag_to_hdf5",
                   help="bag_to_hdf5 command (default: this python -m g2cell_recorder.bag_to_hdf5)")
    f.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"status": cmd_status, "card": cmd_card, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
