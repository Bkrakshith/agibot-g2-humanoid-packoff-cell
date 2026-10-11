#!/usr/bin/env python3
"""Aim the OVRTX FreeCam from the scene YAML's viewer_camera {pos, lookat}.

Why: on the Isaac PhysX launcher the engine ignores viewer_camera_json; the FreeCam keeps
assemble_scene's identity rotation at (3, 0, 2), i.e. looks straight down. The render node does
listen on /genie_sim_engine/viewer/camera_pose (PoseStamped, world frame) and writes the pose
verbatim to the USD camera (looks along local -Z, +Y up). This node computes that pose.

Usage: python3 free_cam_pose.py --scene <scene.yaml> [--pos x y z --lookat x y z] [--once]
Without --once it republishes at 1 Hz (harmless; keeps the pose after a render restart).
"""
import argparse, math, time
import rclpy, yaml
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    f = [t - p for t, p in zip(target, pos)]
    n = math.sqrt(sum(c * c for c in f)); f = [c / n for c in f]
    z = [-c for c in f]                                    # camera +Z points backwards
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    n = math.sqrt(sum(c * c for c in x)); x = [c / n for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]  # columns = camera axes
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s)
    i = max(range(3), key=lambda k: m[k][k]); j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k])
    q = [0.0, 0.0, 0.0]; q[i] = 0.25 * s
    q[j] = (m[j][i] + m[i][j]) / s; q[k] = (m[k][i] + m[i][k]) / s
    return (q[0], q[1], q[2], (m[k][j] - m[j][k]) / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="")
    ap.add_argument("--pos", type=float, nargs=3); ap.add_argument("--lookat", type=float, nargs=3)
    ap.add_argument("--topic", default="/genie_sim_engine/viewer/camera_pose")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    vc = (yaml.safe_load(open(a.scene)) or {}).get("viewer_camera", {}) if a.scene else {}
    pos = a.pos or vc.get("pos"); look = a.lookat or vc.get("lookat")
    q = look_at_quat(pos, look)
    rclpy.init(); node = rclpy.create_node("g2cell_free_cam_pose")
    # the engine subscribes transient-local (latched) + reliable
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(PoseStamped, a.topic, qos)
    msg = PoseStamped(); msg.header.frame_id = "world"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
    node.get_logger().info(f"free cam pos={pos} lookat={look} q_xyzw={tuple(round(c, 4) for c in q)}")
    n = 0
    while rclpy.ok():
        msg.header.stamp = node.get_clock().now().to_msg(); pub.publish(msg); n += 1
        rclpy.spin_once(node, timeout_sec=0.0)
        if a.once and n >= 5: break
        time.sleep(0.2 if n < 5 else 1.0)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
