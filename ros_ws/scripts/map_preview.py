#!/usr/bin/env python3
"""PNG preview of a saved map_server map with the ground-truth layout outlined on top.

Outlines (cell frame = world frame of scene_g2cell_mobile; the SLAM map frame starts equal to odom,
which Genie Sim publishes as the ground-truth world pose): room inner wall faces, conveyor, pick
table, sort rack and the background props, from assets/*/README.md. Lets you judge map alignment.

  python3 map_preview.py <map.yaml> <out.png> [--scale 4]
"""
import argparse
import os

import cv2
import numpy as np
import yaml

# (name, xmin, xmax, ymin, ymax) in metres; BGR colour per group
ROOM = (-4.4, 5.4, -4.9, 4.9)
FIXTURES = [
    ("conveyor", 0.6 - 0.295, 0.6 + 0.295, -1.8, 1.8),
    ("pick table", -0.05 - 0.30, -0.05 + 0.30, 0.85 - 0.50, 0.85 + 0.50),
    ("sort rack", -0.78 - 0.325, -0.78 + 0.325, -0.675, 0.675),
    ("crate stack", -3.6 - 0.2, -3.6 + 0.2, 3.9 - 0.3, 3.9 + 0.3),
    ("pallet", -3.3 - 0.5, -3.3 + 0.5, -3.8 - 0.6, -3.8 + 0.6),
    ("waste bin", 4.6 - 0.29, 4.6 + 0.29, 4.2 - 0.365, 4.2 + 0.365),
    ("handwash", -4.08 - 0.2, -4.08 + 0.2, 2.2 - 0.25, 2.2 + 0.25),
    ("light curtain", 2.3 - 0.1, 2.3 + 0.1, -3.1 - 0.1, -3.1 + 0.1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_yaml")
    ap.add_argument("out")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--no-outlines", action="store_true")
    a = ap.parse_args()
    meta = yaml.safe_load(open(a.map_yaml))
    img = cv2.imread(os.path.join(os.path.dirname(a.map_yaml), meta["image"]), cv2.IMREAD_GRAYSCALE)
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    h, w = img.shape
    s = a.scale
    big = cv2.cvtColor(cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    def px(x, y):  # map yaml origin = lower-left pixel of the image
        return int(round((x - ox) / res * s)), int(round((h - (y - oy) / res) * s))
    if not a.no_outlines:
        x0, x1, y0, y1 = ROOM
        cv2.rectangle(big, px(x0, y1), px(x1, y0), (60, 160, 255), 1)
        for name, xa, xb, ya, yb in FIXTURES:
            cv2.rectangle(big, px(xa, yb), px(xb, ya), (40, 200, 40), 1)
            cv2.putText(big, name, (px(xa, yb)[0], px(xa, yb)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 140, 30), 1, cv2.LINE_AA)
        o = px(0.0, 0.0)
        cv2.drawMarker(big, o, (255, 0, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(big, "cell origin", (o[0] + 6, o[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    occ = int((img < 50).sum())
    free = int((img > 250).sum())
    cv2.putText(big, f"{os.path.basename(a.map_yaml)}: {w}x{h} @ {res} m, occupied {occ} / free {free} cells",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    if not a.no_outlines:
        cv2.putText(big, "orange: room inner walls, green: fixtures (ground-truth layout)", (6, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, big)
    print(a.out, big.shape, "occupied", occ, "free", free)


if __name__ == "__main__":
    main()
