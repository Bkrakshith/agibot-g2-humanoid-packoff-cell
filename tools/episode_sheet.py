#!/usr/bin/env python3
"""Contact sheet of head-camera frames from one g2cell episode HDF5 (host, .venv-learn: h5py + Pillow).

  episode_sheet.py EP.hdf5 OUT.png --every 1.0            frames every N seconds
  episode_sheet.py EP.hdf5 OUT.png --events [--arm left]  frames at the PickBag events, found from the
                                                          commanded gripper (0 = closed, -0.785 = open):
      start | grasp (gripper closed) | lift (+1.5 s) | transfer (midway grasp..place) | place (gripper
      opens again) | end
Each tile is labelled with the event and the time since the start of the recording.
"""
from __future__ import annotations

import argparse

import numpy as np

GRIPPERS = {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}


def gripper_events(t_s: np.ndarray, q: np.ndarray, closed_above: float = -0.3, lift_after: float = 1.5):
    """Times (s) of start / grasp / lift / transfer / place / end from a gripper joint trace.
    grasp = first open -> closed transition (q rises above closed_above); place = first sample after that with q back
    below closed_above (opening). Missing events are left out."""
    ev = {"start": float(t_s[0])}
    closed = q > closed_above
    # the first open -> closed transition (the gripper may start closed at home)
    rises = np.nonzero(closed[1:] & ~closed[:-1])[0]
    i = int(rises[0]) + 1 if rises.size else None
    if i is not None:
        ev["grasp"] = float(t_s[i])
        ev["lift"] = float(min(t_s[-1], t_s[i] + lift_after))
        rest = np.nonzero(~closed[i:])[0]
        if rest.size:
            j = i + int(rest[0])
            ev["transfer"] = float((t_s[i] + t_s[j]) / 2)
            ev["place"] = float(t_s[j])
    ev["end"] = float(t_s[-1])
    return ev


def main(argv=None) -> int:
    import h5py  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5")
    ap.add_argument("out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--every", type=float)
    g.add_argument("--events", action="store_true")
    ap.add_argument("--arm", default="left", choices=sorted(GRIPPERS))
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args(argv)
    with h5py.File(a.hdf5, "r") as f:
        img = f["images/head_front_rgb/data"]
        it = f["images/head_front_rgb/t_ns"][:]
        t0 = int(f.attrs["start_time_ns"])
        if a.events:
            # the COMMANDED gripper: fingers pressed on the rigid bag only close once the lift starts
            grp = "joint_command" if "joint_command" in f else "joint_states"
            key = "actions" if grp == "joint_command" else "positions"
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f[grp].attrs["names"]]
            q = f[f"{grp}/{key}"][:, names.index(GRIPPERS[a.arm])]
            jt = (f[f"{grp}/t_ns"][:] - t0) / 1e9
            picks = list(gripper_events(jt, q).items())
        else:
            end = (it[-1] - t0) / 1e9
            picks = [(f"t", s) for s in np.arange(0.0, end + 1e-9, a.every)]
        tiles = []
        for label, s in picks:
            k = int(np.argmin(np.abs((it - t0) / 1e9 - s)))
            fr = Image.fromarray(img[k])
            if a.scale != 1.0:
                fr = fr.resize((int(fr.width * a.scale), int(fr.height * a.scale)))
            ImageDraw.Draw(fr).text((6, 6), f"{label} {s:.1f}s", fill=(255, 255, 0))
            tiles.append(fr)
    w, h = tiles[0].size
    cols = min(a.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(a.out)
    print(a.out, dict(picks) if a.events else len(picks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
