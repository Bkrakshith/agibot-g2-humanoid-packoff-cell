#!/usr/bin/env python3
"""Render /map (+ ground-truth robot pose, path and optional Nav2 /plan) to PNG frames at a fixed rate.

Used to compose the SLAM/Nav video next to the free camera. Writes <out>/f_00000.png ... (small,
delete after encoding). Starts on "start" on /g2cell_demo/state (if --wait), stops on "done" + 1.5 s
or after --max s.

  python3 map_frames.py --out /g2cell_output/slam/frames_map --fps 6 --wait --max 150
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

SIZE = 540


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--max", type=float, default=150.0)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--title", default="slam_toolbox /map + ground-truth pose")
    # fixed world window = the room (inner faces X [-4.4, 5.4], Y [-4.9, 4.9]) + margin
    ap.add_argument("--window", type=float, nargs=4, default=[-4.6, 5.6, -5.1, 5.1])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    n = rclpy.create_node("g2cell_map_frames")
    st = {"map": None, "odom": None, "plan": None, "t0": None if a.wait else time.monotonic(), "stop": None}
    qmap = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    n.create_subscription(OccupancyGrid, "/map", lambda m: st.__setitem__("map", m), qmap)
    n.create_subscription(Odometry, "/odom", lambda m: st.__setitem__("odom", m), 10)
    n.create_subscription(Path, "/plan", lambda m: st.__setitem__("plan", m), 10)

    def on_state(m):
        if m.data == "start" and st["t0"] is None:
            st["t0"] = time.monotonic()
        elif m.data == "done" and st["t0"] is not None:
            st["stop"] = time.monotonic() + 1.5
    n.create_subscription(String, "/g2cell_demo/state", on_state, 10)
    x0, x1, y0, y1 = a.window
    scale = SIZE / max(x1 - x0, y1 - y0)

    def px(x, y):  # world -> image (x right, y up)
        return int((x - x0) * scale), int(SIZE - (y - y0) * scale)
    trail, k, next_t = [], 0, None
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.02)
        if st["t0"] is None or st["map"] is None or st["odom"] is None:
            continue
        now = time.monotonic()
        if next_t is None:
            next_t = now
        if now < next_t:
            continue
        next_t += 1.0 / a.fps
        img = np.full((SIZE, SIZE, 3), 60, np.uint8)
        m = st["map"]
        g = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
        rgb = np.full(g.shape + (3,), 128, np.uint8)
        rgb[g == 0] = 235
        rgb[g > 50] = 20
        rgb = rgb[::-1]  # row 0 = max y
        r = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        tlx, tly = px(ox, oy + g.shape[0] * r)
        w, h = int(g.shape[1] * r * scale), int(g.shape[0] * r * scale)
        if w > 0 and h > 0:
            res = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            xa, ya = max(tlx, 0), max(tly, 0)
            xb, yb = min(tlx + w, SIZE), min(tly + h, SIZE)
            if xb > xa and yb > ya:
                img[ya:yb, xa:xb] = res[ya - tly:yb - tly, xa - tlx:xb - tlx]
        if st["plan"] is not None and st["plan"].poses:
            pts = np.array([px(p.pose.position.x, p.pose.position.y) for p in st["plan"].poses], np.int32)
            cv2.polylines(img, [pts], False, (60, 200, 60), 2)
        o = st["odom"].pose.pose
        q = o.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        trail.append(px(o.position.x, o.position.y))
        if len(trail) > 1:
            cv2.polylines(img, [np.array(trail, np.int32)], False, (255, 160, 0), 2)
        c, s = math.cos(yaw), math.sin(yaw)
        fp = [(0.4, 0.34), (0.4, -0.34), (-0.4, -0.34), (-0.4, 0.34)]
        poly = np.array([px(o.position.x + c * fx - s * fy, o.position.y + s * fx + c * fy) for fx, fy in fp], np.int32)
        cv2.polylines(img, [poly], True, (0, 120, 255), 2)
        cv2.arrowedLine(img, px(o.position.x, o.position.y), px(o.position.x + 0.6 * c, o.position.y + 0.6 * s),
                        (0, 120, 255), 2, tipLength=0.3)
        cv2.putText(img, a.title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"t = {now - st['t0']:5.1f} s", (8, SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out, f"f_{k:05d}.png"), img)
        k += 1
        if now - st["t0"] > a.max or (st["stop"] is not None and now > st["stop"]):
            break
    print(f"wrote {k} frames to {a.out}")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
