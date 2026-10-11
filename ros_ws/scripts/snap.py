"""snap.py PREFIX TOPIC... : save one frame per topic as PNG (reliable QoS; grab.py's best-effort
subscription intermittently receives no 3.7 MB free-cam frames)."""
import sys, time, rclpy, cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from g2cell_demo.record_cams import to_rgb
rclpy.init(); n = rclpy.create_node("g2cell_snap"); got = {}
q = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
for t in sys.argv[2:]:
    n.create_subscription(Image, t, lambda m, t=t: got.__setitem__(t, m), q)
t0 = time.time()
while time.time() - t0 < 1.0 or (len(got) < len(sys.argv[2:]) and time.time() - t0 < 20):
    rclpy.spin_once(n, timeout_sec=0.05)
for t, m in got.items():
    out = f"/g2cell_output/{sys.argv[1]}_{t.strip('/').split('/')[1]}.png"
    cv2.imwrite(out, to_rgb(m)[:, :, ::-1]); print(out)
print("missing:", [t for t in sys.argv[2:] if t not in got])
