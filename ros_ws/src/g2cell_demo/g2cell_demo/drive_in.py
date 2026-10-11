"""Drive the mobile G2 from its start pose into the U-shaped cell (Layout v2) and stop at the working pose.

Closed loop on /odom (ground truth in Genie Sim), commands on /cmd_twist (geometry_msgs/Twist, body frame)
-> servo_node (GeneralServo) -> /cmd_4ws. Straight line start -> goal with a trapezoidal, accel-limited
speed profile (<= v_max), heading + cross-track steering, then an in-place yaw correction
(g2cell_demo.drive_control). Logs every control step and the chassis clearance to the U fixtures
(pick table inner edge Y +0.43, rack/tote inner edge Y -0.43; assets/cell/README.md "Layout v2").

    ros2 run g2cell_demo drive_in --ros-args -p goal:="[0.0, 0.0, 0.0]" -p v_max:=0.25

Use the mobile scene (scripts/mobile_up.sh). use_sim_time must stay false: the engine stamps /odom with
wall time. The node is also used as a library by packing_demo (DriveIn.run()).
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from g2cell_demo.drive_control import DriveController, DriveParams, footprint, pose_error, rect_distance

# U fixtures as 2D keep-outs at chassis height (cell frame, assets/cell/build.py Layout v2):
# (name, x0, x1, y0, y1). The conveyor is NOT a chassis keep-out: its lowest frame part is at
# z 0.70 m (chassis top 0.39 m) and its legs are at |Y| >= 0.58 m, so the chassis nose runs under it.
KEEP_OUT = [
    ("pick_table", -0.715, 0.285, 0.43, 1.03),
    ("sort_rack+totes", -0.275, 0.275, -1.145, -0.43),
    ("conveyor_legs_near", 0.305, 0.385, 0.58, 0.62),
    ("conveyor_legs_near_r", 0.305, 0.385, -0.62, -0.58),
]


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DriveIn:
    """Mixin-free helper: give it a Node; it subscribes /odom and publishes /cmd_twist."""

    def __init__(self, node: Node, params: DriveParams | None = None, rate: float = 50.0):
        self.node = node
        self.params = params or DriveParams()
        self.rate = rate
        self._lock = threading.Lock()
        self._pose = None
        self.pub = node.create_publisher(Twist, "/cmd_twist", 10)
        node.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.log = []

    def _on_odom(self, m: Odometry):
        p = m.pose.pose
        with self._lock:
            self._pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def pose(self):
        with self._lock:
            return self._pose

    def wait_pose(self, timeout=30.0):
        t0 = time.monotonic()
        while self.pose() is None and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        return self.pose()

    def send(self, v=0.0, w=0.0):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)

    def stop(self, n=10):
        for _ in range(n):
            self.send(0.0, 0.0)
            time.sleep(1.0 / self.rate)

    @staticmethod
    def clearance(pose):
        fp = footprint(pose)
        return {name: rect_distance(fp, (x0, x1, y0, y1)) for name, x0, x1, y0, y1 in KEEP_OUT}

    def run(self, goal, timeout=120.0, settle=1.5):
        """Drive to goal (x, y, yaw). Returns a result dict (also keeps self.log rows)."""
        start = self.wait_pose()
        if start is None:
            return {"ok": False, "reason": "no /odom"}
        ctl = DriveController(start=start[:2], goal=tuple(goal), p=self.params)
        dt = 1.0 / self.rate
        t0 = time.monotonic()
        nxt = t0
        min_clear = {name: float("inf") for name, *_ in KEEP_OUT}
        dist, prev = 0.0, start
        self.node.get_logger().info(f"drive-in: start {tuple(round(c, 3) for c in start)} -> goal {tuple(goal)}")
        while ctl.phase != "done" and time.monotonic() - t0 < timeout:
            pose = self.pose()
            v, w = ctl.step(pose, dt)
            self.send(v, w)
            cl = self.clearance(pose)
            for k, d in cl.items():
                min_clear[k] = min(min_clear[k], d)
            dist += math.hypot(pose[0] - prev[0], pose[1] - prev[1])
            prev = pose
            self.log.append((round(time.monotonic() - t0, 3), ctl.phase, round(v, 4), round(w, 4),
                             round(pose[0], 4), round(pose[1], 4), round(math.degrees(pose[2]), 3),
                             round(min(cl.values()), 4)))
            nxt += dt
            time.sleep(max(0.0, nxt - time.monotonic()))
        self.stop()
        t_drive = time.monotonic() - t0
        time.sleep(settle)
        final = self.pose()
        dp, dyaw = pose_error(final, goal)
        res = {"ok": ctl.phase == "done" and dp <= 0.03 and dyaw <= 2.0, "phase": ctl.phase,
               "start": start, "goal": list(goal), "final": final, "pos_err_m": dp, "yaw_err_deg": dyaw,
               "time_s": t_drive, "path_m": dist,
               "max_v": max((r[2] for r in self.log), default=0.0),
               "min_clearance_m": min_clear}
        self.node.get_logger().info(
            f"drive-in {'OK' if res['ok'] else 'FAILED'}: final {tuple(round(c, 4) for c in final)}, "
            f"error {dp * 100:.1f} cm / {dyaw:.2f} deg, {t_drive:.1f} s, path {dist:.2f} m, "
            f"min clearance {min(min_clear.values()):.3f} m")
        return res

    def write_log(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "phase", "v_cmd", "w_cmd", "x", "y", "yaw_deg", "min_clearance"])
            w.writerows(self.log)


def main():
    rclpy.init()
    node = rclpy.create_node("g2cell_drive_in")
    goal = list(node.declare_parameter("goal", [0.0, 0.0, 0.0]).value)
    goal[2] = math.radians(goal[2]) if abs(goal[2]) > 2 * math.pi else goal[2]
    prm = DriveParams(v_max=min(0.3, node.declare_parameter("v_max", 0.25).value))
    out = node.declare_parameter("out", "/g2cell_output/drive_in").value
    drv = DriveIn(node, prm)
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()
    try:
        res = drv.run(goal)
        drv.write_log(out + ".csv")
        json.dump(res, open(out + ".json", "w"), indent=1)
        print(json.dumps(res, indent=1))
    except KeyboardInterrupt:
        drv.stop()
    finally:
        drv.stop(3)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
