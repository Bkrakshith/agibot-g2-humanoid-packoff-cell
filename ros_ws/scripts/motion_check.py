"""motion_check.py MP4: frame-to-frame mean abs diff; reports frozen runs and jumps (render stalls)."""
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); prev = None; d = []
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is not None: d.append(float(np.abs(g - prev).mean()))
    prev = g
d = np.array(d); moving = d > 0.05
print(f"{sys.argv[1]}: frames {len(d)+1}, diff mean {d.mean():.3f} p50 {np.median(d):.3f} p95 {np.percentile(d,95):.3f} max {d.max():.3f} at {d.argmax()/fps:.1f}s")
print("per-second sum of diffs:", " ".join(f"{d[int(i*fps):int((i+1)*fps)].sum():.1f}" for i in range(int(len(d)/fps)+1)))
