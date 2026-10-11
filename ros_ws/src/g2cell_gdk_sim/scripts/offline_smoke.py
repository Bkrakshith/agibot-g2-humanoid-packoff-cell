#!/usr/bin/env python3
"""Offline smoke test of the ROS backend against a FAKE sim on an isolated ROS domain.

The fake sim publishes /joint_states (follows /joint_command instantly), /tf, the head camera
topics, and serves /compute_ik and /compute_fk with toy answers. It never talks to Genie Sim:
the script refuses to run on ROS_DOMAIN_ID 0 (the sim's domain) and needs localhost-only discovery.

    ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST python3 offline_smoke.py           # checks
    ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST python3 offline_smoke.py --serve   # fake sim only
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0") or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
    sys.exit("refusing: set ROS_DOMAIN_ID to an unused non-zero domain and ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped, TransformStamped  # noqa: E402
from moveit_msgs.srv import GetPositionFK, GetPositionIK  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

BODY = ([f"idx0{i}_body_joint{i}" for i in range(1, 6)] + [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
        + [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)] + [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)])
NAMES = BODY + ["idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1", "idx111_chassis_lwheel_front_joint1"]
HEAD = (0.05, 0.0, 1.45)
ENDS = {"arm_l_end_link": (0.6, 0.3, 1.0), "arm_r_end_link": (0.6, -0.3, 1.0)}


def stamp():
    return Time(nanoseconds=time.time_ns()).to_msg()


class FakeSim:
    def __init__(self):
        self.node = rclpy.create_node("fake_genie_sim")
        n = self.node
        self.pos = {name: 0.0 for name in NAMES}
        self.pos["idx13_head_joint3"] = 0.4
        self.pos["idx01_body_joint1"] = 0.005  # rests past the real limit, as the sim's waist does
        self.commands = 0
        self.js_pub = n.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        n.create_subscription(JointState, "/joint_command", self.on_command,
                              QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.tf_pub = n.create_publisher(TFMessage, "/tf", 10)
        self.rgb_pub = n.create_publisher(Image, "/genie_sim/head_front_camera_rgb/image_raw", qos_profile_sensor_data)
        self.depth_pub = n.create_publisher(Image, "/genie_sim/head_front_camera_depth/image_raw", qos_profile_sensor_data)
        n.create_service(GetPositionIK, "/compute_ik", self.on_ik)
        n.create_service(GetPositionFK, "/compute_fk", self.on_fk)
        n.create_timer(0.01, self.publish_joints)
        n.create_timer(0.02, self.publish_tf)
        n.create_timer(0.1, self.publish_images)

    def on_command(self, msg):
        self.commands += 1
        for name, value in zip(msg.name, msg.position):
            self.pos[name] = value

    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = stamp()
        msg.name = list(self.pos)
        msg.position = list(self.pos.values())
        msg.velocity = [0.0] * len(self.pos)
        msg.effort = [1.0] * len(self.pos)
        self.js_pub.publish(msg)

    def publish_tf(self):
        out = TFMessage()
        for child, xyz in [("head_link3", HEAD), *ENDS.items()]:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp(), "base_link", child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            t.transform.rotation.w = 1.0
            out.transforms.append(t)
        self.tf_pub.publish(out)

    def publish_images(self):
        rgb = Image(height=400, width=640, encoding="rgba8", step=640 * 4)
        rgb.header.stamp = stamp()
        px = np.zeros((400, 640, 4), np.uint8)
        px[..., 0] = 200  # red in RGBA
        rgb.data = px.tobytes()
        self.rgb_pub.publish(rgb)
        depth = Image(height=400, width=640, encoding="32FC1", step=640 * 4)
        depth.header.stamp = rgb.header.stamp
        depth.data = np.full((400, 640), 1.5, np.float32).tobytes()
        self.depth_pub.publish(depth)

    def on_ik(self, req, res):
        r = req.ik_request
        seed = dict(zip(r.robot_state.joint_state.name, r.robot_state.joint_state.position))
        side = "l" if r.group_name.endswith("_l") else "r"
        names = [n for n in BODY if f"arm_{side}_joint" in n]
        res.solution.joint_state.name = names
        res.solution.joint_state.position = [seed.get(n, 0.0) + 0.001 for n in names]
        res.error_code.val = 1
        return res

    def on_fk(self, req, res):
        for link in req.fk_link_names:
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            xyz = HEAD if link == "head_link3" else (0.6, 0.3, 0.9)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
            ps.pose.orientation.w = 1.0
            res.pose_stamped.append(ps)
        res.fk_link_names = list(req.fk_link_names)
        res.error_code.val = 1
        return res


results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'ok' if condition else 'FAIL'}] {name} {detail}", flush=True)


def run_checks(sim: FakeSim) -> None:
    import agibot_gdk
    from agibot_gdk import _geometry as g

    check("gdk_init", agibot_gdk.gdk_init() == agibot_gdk.GDKRes.kSuccess)
    robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
    C = agibot_gdk.CameraType
    camera = agibot_gdk.Camera([C.kHeadColor, C.kHeadDepth])
    time.sleep(3.0)
    js = robot.get_joint_states()
    check("get_joint_states 22 joints", js["nums"] == 22, f"nums={js['nums']}")
    check("nothing published before the first command", sim.commands == 0)
    check("whole body status", robot.get_whole_body_status()["left_arm_estop"] is False)
    status = robot.get_motion_control_status()
    check("motion control frames from TF", "arm_l_end_link" in status.frame_names, str(list(status.frame_names)))
    check("frame names", "head_link3" in tf.get_all_frame_names() and "gripper_l_center_link" in tf.get_all_frame_names())
    t, ts = tf.lookup_transform_latest("base_link", "head_link3", return_timestamp=True)
    check("lookup direction (pose of base in head)", abs(g.from_transform(t)[2, 3] + 1.45) < 1e-6 and ts > 0)
    started = time.monotonic()
    r = robot.move_head_joint([0.2, 0.0, 0.3], [0.5, 0.5, 0.5])
    took = time.monotonic() - started
    check("move_head_joint blocks and arrives", r == 0 and abs(sim.pos["idx11_head_joint1"] - 0.2) < 1e-3,
          f"result {r} in {took:.2f} s")
    check("streaming started", sim.commands > 10, f"{sim.commands} /joint_command messages")
    try:
        robot.move_head_joint([0.0, 0.0, 9.0], [0.5] * 3)
        check("out-of-limit target refused", False)
    except RuntimeError as error:
        check("out-of-limit target refused", "outside" in str(error), str(error))
    check("go back to a drifted waist pose", robot.move_waist_joint([0.005, 0.0, 0.0, 0.0, 0.0], [0.3] * 5) == 0)
    req = agibot_gdk.JointServoControlReq()
    req.control_period, req.joint_names, req.joint_positions = 0.035, ["idx27_arm_l_joint7"], [0.05]
    check("joint_servo_control", robot.joint_servo_control(req) == 0)
    time.sleep(0.1)
    check("servo target reached the sim", abs(sim.pos["idx27_arm_l_joint7"] - 0.05) < 1e-6)
    pose = agibot_gdk.EndEffectorPose()
    pose.life_time, pose.group = 0.02, int(agibot_gdk.EndEffectorControlGroup.kLeftArm)
    left = g.matrix(ENDS["arm_l_end_link"], (0, 0, 0, 1))
    left[2, 3] += 0.002
    g.fill_pose(pose.left_end_effector_pose, left)
    g.fill_pose(pose.right_end_effector_pose, g.matrix(ENDS["arm_r_end_link"], (0, 0, 0, 1)))
    check("end_effector_pose_control via /compute_ik", robot.end_effector_pose_control(pose) == 0)
    cmd = agibot_gdk.JointStates()
    cmd.group, cmd.target_type, cmd.nums = "dual_tool", "omnipicker", 2
    for p in (-0.785, 0.0):
        s = agibot_gdk.JointState()
        s.position = p
        cmd.states.append(s)
    check("move_ee_pos", robot.move_ee_pos(cmd) == 0)
    time.sleep(0.2)
    end = robot.get_end_state()
    check("gripper state", abs(end["left_end_state"]["end_states"][0]["position"] + 0.785) < 1e-3)
    image = camera.get_latest_image(C.kHeadColor, 500.0)
    ok = image is not None and image.color_format == agibot_gdk.ColorFormat.BGR
    pixels = np.frombuffer(image.data, np.uint8).reshape(image.height, image.width, 3) if ok else None
    check("head colour image BGR", ok and pixels[0, 0].tolist() == [0, 0, 200])
    depth = camera.get_latest_image(C.kHeadDepth, 500.0)
    mm = np.frombuffer(depth.data, np.uint16) if depth is not None else None
    check("head depth Z16 mm", mm is not None and int(mm[0]) == 1500)
    check("camera fps measured", 5.0 < camera.get_image_fps(C.kHeadColor) < 15.0, f"{camera.get_image_fps(C.kHeadColor):.1f}")

    # single-publisher guard: another node starts publishing /joint_command
    intruder = rclpy.create_node("intruder_teleop")
    intruder.create_publisher(JointState, "/joint_command", 5)
    time.sleep(3.0)
    try:
        robot.move_head_joint([0.0, 0.0, 0.3], [0.5] * 3)
        check("guard stops motion when a second publisher appears", False)
    except RuntimeError as error:
        check("guard stops motion when a second publisher appears", "kRuntimeError" in str(error), str(error)[:120])
    agibot_gdk.gdk_release()
    # a new session refuses to start streaming while the intruder exists
    agibot_gdk.gdk_init()
    time.sleep(2.0)
    try:
        agibot_gdk.Robot().move_head_joint([0.0, 0.0, 0.3], [0.5] * 3)
        check("guard refuses to start streaming", False)
    except RuntimeError as error:
        check("guard refuses to start streaming", "intruder_teleop" in str(error), str(error)[:120])
    intruder.destroy_node()
    agibot_gdk.gdk_release()


def main() -> int:
    rclpy.init()
    sim = FakeSim()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(sim.node)
    threading.Thread(target=executor.spin, daemon=True).start()
    if "--serve" in sys.argv:
        print("fake sim running, Ctrl+C to stop", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
    try:
        run_checks(sim)
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
