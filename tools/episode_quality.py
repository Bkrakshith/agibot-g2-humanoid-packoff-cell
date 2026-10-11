#!/usr/bin/env python3
"""Per-episode quality checks for a datagen batch (host, .venv-learn): duration, gripper events, waist yaw at the
grasp, and how much of the (green) bag the head camera sees at grasp / lift (fraction of green pixels), so
episodes whose grasp happens outside the head camera's view can be flagged.

    episode_quality.py output/episodes/pilot_v1 [-o out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from episode_sheet import GRIPPERS, gripper_events


def green_fraction(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return float(((g > 90) & (g > r + 15) & (g > b + 40)).mean())   # bag film (~120, 145, 70)


def episode(path: str) -> dict:
    import h5py  # noqa: PLC0415
    with h5py.File(path, "r") as f:
        t0 = int(f.attrs["start_time_ns"])
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["joint_command"].attrs["names"]]
        q = f["joint_command/actions"][:, names.index(GRIPPERS["left"])]
        ct = (f["joint_command/t_ns"][:] - t0) / 1e9
        ev = gripper_events(ct, q)
        js = [n.decode() if isinstance(n, bytes) else str(n) for n in f["joint_states"].attrs["names"]]
        jt = (f["joint_states/t_ns"][:] - t0) / 1e9
        waist = f["joint_states/positions"][:, js.index("idx05_body_joint5")]
        it = (f["images/head_front_rgb/t_ns"][:] - t0) / 1e9
        img = f["images/head_front_rgb/data"]
        out = {"file": os.path.basename(path), "duration_s": round(float(f.attrs["duration_s"]), 2),
               "frames": int(img.shape[0]), "success": str(f.attrs.get("success", "")),
               "depth_frames": int(f["depth/head_front/data"].shape[0]) if "depth/head_front/data" in f else 0,
               "size_mb": round(os.path.getsize(path) / 1e6, 1), "events": {k: round(v, 2) for k, v in ev.items()}}
        for k in ("grasp", "lift"):
            if k in ev:
                out[f"waist_deg_at_{k}"] = round(float(np.degrees(waist[np.argmin(abs(jt - ev[k]))])), 1)
                out[f"green_at_{k}"] = round(green_fraction(img[int(np.argmin(abs(it - ev[k])))]), 4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args(argv)
    rows = [episode(p) for p in sorted(glob.glob(os.path.join(a.batch_dir, "ep_s*", "ep_s*.hdf5")))]
    for r in rows:
        print(json.dumps(r))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
