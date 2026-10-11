"""video_check.py MP4 [thresh]: per-second mean brightness; flags frames darker than thresh (occlusion)."""
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); th = float(sys.argv[2]) if len(sys.argv) > 2 else 30
fps = cap.get(cv2.CAP_PROP_FPS); means = []
while True:
    ok, f = cap.read()
    if not ok: break
    means.append(f.mean())
m = np.array(means); dark = np.where(m < th)[0]
print(f"{sys.argv[1]}: {len(m)} frames {len(m)/fps:.1f}s mean={m.mean():.1f} min={m.min():.1f} dark(<{th})={len(dark)} at s={sorted(set((dark/fps).round(0)))}")
