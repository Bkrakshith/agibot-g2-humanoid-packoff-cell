#!/usr/bin/env python3
"""img_diff.py A.png B.png [...]: mean/p99 abs pixel difference of each image vs A, plus brightness and
sharpness (variance of the Laplacian) of every image, as JSON. Used to compare render profiles."""
import json, sys
import cv2
import numpy as np

def stats(p):
    im = cv2.imread(p)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float64), {"mean_luma": round(float(g.mean()), 1),
                                   "black_frac": round(float((g < 8).mean()), 4),
                                   "sharpness_lapvar": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)}

ref, s0 = stats(sys.argv[1])
out = {sys.argv[1]: s0}
for p in sys.argv[2:]:
    im, s = stats(p)
    d = np.abs(im - ref) if im.shape == ref.shape else None
    if d is not None:
        s.update(mean_abs_diff_vs_ref=round(float(d.mean()), 2), p99_abs_diff_vs_ref=round(float(np.percentile(d, 99)), 1))
    out[p] = s
print(json.dumps(out, indent=1))
