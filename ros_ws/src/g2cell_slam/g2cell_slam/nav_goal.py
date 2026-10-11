"""Send one Nav2 NavigateToPose goal and score it against ground truth.

Optionally publishes /initialpose from the ground-truth /odom first (AMCL needs a seed; in Genie Sim
/odom is the engine's ground-truth base pose and odom == world). The map was built by slam_toolbox
starting with map == odom, so map ~= world and the goal can be compared with /odom directly.

Prints and (with -p out:=file.json) writes: success, nav time, final ground-truth pose, position and
yaw error vs the goal, and the localisation error (map->base_footprint TF vs ground truth).

  ros2 run g2cell_slam nav_goal --ros-args -p x:=1.6 -p y:=0.0 -p yaw_deg:=180.0 -p set_initial_pose:=true
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class NavGoal(Node):
    def __init__(self):
        super().__init__("g2cell_nav_goal")
        p = self.declare_parameter
        self.gx, self.gy = float(p("x", 1.6).value), float(p("y", 0.0).value)
        self.gyaw = math.radians(float(p("yaw_deg", 180.0).value))
        self.set_init = bool(p("set_initial_pose", True).value)
        self.timeout = float(p("timeout", 180.0).value)
        self.out = p("out", "").value
        self.odom = None
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 20)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.state_pub = self.create_publisher(String, "/g2cell_demo/state", 10)  # for record_cams
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def spin_until(self, cond, timeout):
        t0 = time.monotonic()
        while rclpy.ok() and not cond() and time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return cond()

    def gt(self):
        o = self.odom.pose.pose
        return o.position.x, o.position.y, yaw_of(o.orientation)

    def est(self):
        try:
            t = self.tf.lookup_transform("map", "base_footprint", Time(), Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"no map->base_footprint: {e}")
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def run(self):
        if not self.spin_until(lambda: self.odom is not None, 10.0):
            raise SystemExit("no /odom")
        if self.set_init:
            x, y, yaw = self.gt()
            m = PoseWithCovarianceStamped()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x, m.pose.pose.position.y = x, y
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.pose.covariance[0] = m.pose.covariance[7] = 0.05 ** 2
            m.pose.covariance[35] = math.radians(3) ** 2
            for _ in range(3):
                self.init_pub.publish(m)
                self.spin_until(lambda: False, 0.5)
            self.get_logger().info(f"initial pose (from ground truth) {x:.2f} {y:.2f} {math.degrees(yaw):.1f}")
            self.spin_until(lambda: False, 2.0)
        if not self.ac.wait_for_server(timeout_sec=30.0):
            raise SystemExit("navigate_to_pose action server not available")
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = self.gx, self.gy
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(self.gyaw / 2), math.cos(self.gyaw / 2)
        goal.pose = ps
        start_gt = self.gt()
        self.get_logger().info(f"goal ({self.gx}, {self.gy}, {math.degrees(self.gyaw):.0f} deg) from "
                               f"{tuple(round(c, 2) for c in start_gt)}")
        t0 = time.monotonic()
        self.state_pub.publish(String(data="start"))
        fut = self.ac.send_goal_async(goal)
        self.spin_until(fut.done, 10.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            raise SystemExit("goal rejected")
        res = gh.get_result_async()
        self.spin_until(res.done, self.timeout)
        dt = time.monotonic() - t0
        self.state_pub.publish(String(data="done"))
        status = res.result().status if res.done() else -1  # 4 = SUCCEEDED
        self.spin_until(lambda: False, 1.0)  # let the base settle / odom update
        x, y, yaw = self.gt()
        est = self.est()
        rep = {
            "goal": [self.gx, self.gy, math.degrees(self.gyaw)],
            "start_ground_truth": [round(start_gt[0], 3), round(start_gt[1], 3), round(math.degrees(start_gt[2]), 1)],
            "status": status, "succeeded": status == 4, "nav_time_s": round(dt, 1),
            "final_ground_truth": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
            "pos_error_m": round(math.hypot(x - self.gx, y - self.gy), 3),
            "yaw_error_deg": round(math.degrees(abs(wrap(yaw - self.gyaw))), 1),
        }
        if est:
            rep["final_estimate_map"] = [round(est[0], 3), round(est[1], 3), round(math.degrees(est[2]), 1)]
            rep["localisation_error_m"] = round(math.hypot(est[0] - x, est[1] - y), 3)
            rep["localisation_yaw_error_deg"] = round(math.degrees(abs(wrap(est[2] - yaw))), 1)
        print(json.dumps(rep, indent=1))
        if self.out:
            with open(self.out, "w") as f:
                json.dump(rep, f, indent=1)
        return rep


def main():
    rclpy.init()
    node = NavGoal()
    try:
        node.run()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
