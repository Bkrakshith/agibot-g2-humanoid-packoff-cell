"""Scripted mapping drive: follow a list of (x, y) waypoints with rotate-then-go control.

Feedback is /odom, which in Genie Sim is the GROUND-TRUTH base pose (engine snapshot of base_link in
the world; odom == world, no drift). The SLAM map is built only from the lidar scans + TF; this node
just steers the robot along a collision-free loop.

Publishes geometry_msgs/Twist on /cmd_vel (-> cmd_vel_bridge -> /cmd_twist), and "start"/"done"
on /g2cell_demo/state (std_msgs/String) so g2cell_demo record_cams can start/stop with it.

  ros2 run g2cell_slam drive_path --ros-args -p "waypoints:=[3.4,3.2, -2.5,3.2, -2.5,-2.45, 3.4,-2.45, 3.4,0.0]" \
      -p final_yaw_deg:=180.0 -p max_linear:=0.35
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

DEFAULT_LOOP = [3.4, 3.2, -2.5, 3.2, -2.5, -2.45, 3.4, -2.45, 3.4, 0.0]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class DrivePath(Node):
    def __init__(self):
        super().__init__("g2cell_drive_path")
        p = self.declare_parameter
        wp = list(p("waypoints", DEFAULT_LOOP).value)
        self.wps = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)]
        self.vmax = min(0.4, float(p("max_linear", 0.35).value))  # task limit: <= 0.4 m/s
        self.wmax = float(p("max_angular", 0.5).value)
        self.tol = float(p("tolerance", 0.08).value)
        self.final_yaw = p("final_yaw_deg", 180.0).value
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/g2cell_demo/state", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.pose = None
        self.i = 0
        self.phase = "rotate"
        self.dist = 0.0
        self.prev_xy = None
        self.t0 = None
        self.create_timer(0.05, self.tick)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                     math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        if self.prev_xy is not None:
            self.dist += math.hypot(self.pose[0] - self.prev_xy[0], self.pose[1] - self.prev_xy[1])
        self.prev_xy = self.pose[:2]

    def send(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = v, w
        self.pub.publish(t)

    def tick(self):
        if self.pose is None:
            return
        if self.t0 is None:
            self.t0 = time.monotonic()
            self.state_pub.publish(String(data="start"))
            self.get_logger().info(f"start at {tuple(round(c, 2) for c in self.pose)}; {len(self.wps)} waypoints")
        x, y, yaw = self.pose
        if self.i >= len(self.wps):
            if self.final_yaw is None or self.phase == "done":
                return
            err = wrap(math.radians(float(self.final_yaw)) - yaw)
            if abs(err) < math.radians(2.0):
                self.send(0.0, 0.0)
                self.phase = "done"
                self.state_pub.publish(String(data="done"))
                self.get_logger().info(f"done in {time.monotonic() - self.t0:.1f} s, path {self.dist:.2f} m, "
                                       f"final pose {x:.3f} {y:.3f} {math.degrees(yaw):.1f} deg")
                return
            self.send(0.0, max(-self.wmax, min(self.wmax, 1.5 * err)))
            return
        gx, gy = self.wps[self.i]
        d = math.hypot(gx - x, gy - y)
        bearing = math.atan2(gy - y, gx - x)
        err = wrap(bearing - yaw)
        if d < self.tol:
            self.get_logger().info(f"waypoint {self.i} ({gx}, {gy}) reached: pos {x:.2f} {y:.2f}, t={time.monotonic() - self.t0:.1f} s")
            self.i += 1
            self.phase = "rotate"
            self.send(0.0, 0.0)
            return
        if self.phase == "rotate":
            if abs(err) < math.radians(3.0):
                self.phase = "go"
            else:
                self.send(0.0, max(-self.wmax, min(self.wmax, 1.5 * err)))
                return
        if abs(err) > math.radians(25.0):  # lost heading: stop and rotate again
            self.phase = "rotate"
            self.send(0.0, 0.0)
            return
        v = min(self.vmax, 0.6 * d + 0.05) * max(0.0, math.cos(err))
        self.send(v, max(-self.wmax, min(self.wmax, 2.0 * err)))


def main():
    rclpy.init()
    node = DrivePath()
    try:
        while rclpy.ok() and node.phase != "done":
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):
            node.send(0.0, 0.0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        node.send(0.0, 0.0)
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
