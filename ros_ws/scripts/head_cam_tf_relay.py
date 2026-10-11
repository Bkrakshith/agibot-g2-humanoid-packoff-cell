#!/usr/bin/env python3
"""Drive the OVRTX head-camera parent Xform with head_link3's WORLD pose.

Why: assemble_scene.py synthesizes the URDF-route head camera at
/RenderOVRTX/Cameras/head_link3/Head_Front_Camera and the render node only moves
/RenderOVRTX/Cameras/head_link3 when a /tf_render child_frame_id equals "head_link3".
The engine, however, publishes full nested prim paths
("/genie/Geometry/base_link/.../head_link3") with parent-relative poses, so the key never
matches and the camera parent stays at identity = world origin (inside the chassis, near
the floor) -> black frames, 3-6 cm depth.

This node composes the nested /tf_render poses down to head_link3 and republishes the
world pose on /tf_render with child_frame_id "/RenderOVRTX/Cameras/head_link3" (an
absolute path, which the render node applies verbatim). The YAML extrinsic then stays the
official head_front_Camera mount relative to head_link3. Upstream code is untouched.

It also fixes a second render-only offset: the engine applies robot.init_base_pose to the
/<prefix> root Xform on the PHYSICS stage, but /tf_render only carries parent-relative link
poses and the OVRTX stage's /<prefix> stays at identity, so the rendered robot sits
init_base_pose away from the physics robot (0.2 m forward for scene_pnp_g2_op). With
--scene, the root pose is published for "/<prefix>" and prepended to the camera pose.

Lidars (mobile scene): the render node mirrors a lidar's parent body onto /RenderOVRTX/Lidars/<body>
only for child_frame_id == "<body>", which the engine never sends either. With --lidar-body base_link
the world pose of base_link is also published for "/RenderOVRTX/Lidars/base_link", so the OmniLidar
copies ray-cast from the moving base.

Usage: python3 head_cam_tf_relay.py --scene <scene.yaml> [--link head_link3] [--lidar-body base_link] [--once]
"""
import argparse
import math
import rclpy
import yaml
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def qmul(a, b):  # (w,x,y,z)
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)


def qrot(q, v):
    w, x, y, z = qmul(qmul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="head_link3")
    ap.add_argument("--scene", default="", help="scene YAML (reads robot.init_base_pose, robot_prefix)")
    ap.add_argument("--once", action="store_true", help="print world pose once and exit")
    ap.add_argument("--lidar-body", default="", help="also publish this link's world pose for /RenderOVRTX/Lidars/<link>")
    args = ap.parse_args()
    root_q, root_t, root_path = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
    if args.scene:
        robot = (yaml.safe_load(open(args.scene)) or {}).get("robot") or {}
        bp = robot.get("init_base_pose") or {}
        th = float(bp.get("theta", 0.0))
        root_q = (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))
        root_t = (float(bp.get("x", 0.0)), float(bp.get("y", 0.0)), float(bp.get("z", 0.0)))
        root_path = "/" + robot.get("robot_prefix", "genie")
    target = "/RenderOVRTX/Cameras/" + args.link
    rclpy.init()
    node = rclpy.create_node("g2cell_head_cam_tf_relay")
    pub = node.create_publisher(TFMessage, "/tf_render", qos_profile_sensor_data)
    state = {"done": False}

    def world_pose(poses, leaf):
        parts = leaf.split("/")
        q, t = root_q, root_t
        for i in range(2, len(parts) + 1):
            p = "/".join(parts[:i])
            if p not in poses:
                continue  # non-body prims (/genie, /genie/Geometry) are identity
            lq, lt = poses[p]
            rt = qrot(q, lt)
            t = (t[0] + rt[0], t[1] + rt[1], t[2] + rt[2])
            q = qmul(q, lq)
        return q, t

    def cb(msg):
        poses = {}
        leaf = None
        body_leaf = None
        for tf in msg.transforms:
            p = tf.child_frame_id
            if p.startswith("/RenderOVRTX") or p == root_path:
                return  # our own echo
            r, t = tf.transform.rotation, tf.transform.translation
            poses[p] = ((r.w, r.x, r.y, r.z), (t.x, t.y, t.z))
            if p.endswith("/" + args.link):
                leaf = p
            if args.lidar_body and p.endswith("/" + args.lidar_body):
                body_leaf = p
        if leaf is None:
            return
        q, t = world_pose(poses, leaf)
        if args.once:
            print(f"{args.link} world xyz={t} wxyz={q}")
            if body_leaf:
                print(f"{args.lidar_body} world xyz/wxyz={world_pose(poses, body_leaf)[::-1]}")
            state["done"] = True
            return
        def mk(child, tt, qq):
            out = TransformStamped()
            out.header = msg.transforms[0].header
            out.child_frame_id = child
            out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = tt
            (out.transform.rotation.w, out.transform.rotation.x,
             out.transform.rotation.y, out.transform.rotation.z) = qq
            return out
        tfs = [mk(target, t, q)]
        if body_leaf:
            bq, bt = world_pose(poses, body_leaf)
            tfs.append(mk("/RenderOVRTX/Lidars/" + args.lidar_body, bt, bq))
        if root_path:
            tfs.insert(0, mk(root_path, root_t, root_q))
        pub.publish(TFMessage(transforms=tfs))

    node.create_subscription(TFMessage, "/tf_render", cb, qos_profile_sensor_data)
    node.get_logger().info(f"relaying world pose of {args.link} -> {target}")
    while rclpy.ok() and not state["done"]:
        rclpy.spin_once(node, timeout_sec=0.1)


if __name__ == "__main__":
    main()
