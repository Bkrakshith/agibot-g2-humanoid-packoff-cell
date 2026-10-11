#!/usr/bin/env python3
"""Offline smoke test of the use-case stack (motion_server, pick_bag_server, cell_manager) with a
FAKE robot and FAKE move_group -- never touches the sim.

Isolation, three layers deep:
  1. refuses to run unless ROS_DOMAIN_ID is set to something other than 0 (the sim's domain);
  2. ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST;
  3. every robot-facing name is remapped under /offline/ (/joint_command, /joint_states,
     /compute_ik, /compute_fk), so even a leak could not reach the sim's /joint_command.

    ROS_DOMAIN_ID=99 python3 /g2cell_ws/scripts/offline_smoke.py

The fake IK always solves (joints = simple function of the target), the fake FK keeps every
link 0.5 m+ from the head, and the fake robot tracks /offline/joint_command instantly. So this
checks plumbing (actions, feedback, outcomes, services, status, guard), not kinematics.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0"):
    sys.exit("refusing to run on ROS_DOMAIN_ID 0 (the sim's domain): set e.g. ROS_DOMAIN_ID=99")
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from moveit_msgs.srv import GetPositionFK, GetPositionIK  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from g2cell_interfaces.action import MoveArm, PickBag  # noqa: E402
from g2cell_interfaces.msg import CellStatus  # noqa: E402
from g2cell_interfaces.srv import EpisodeControl, SetLineState, SpawnBags  # noqa: E402

REMAP = ["-r", "/joint_command:=/offline/joint_command", "-r", "/joint_states:=/offline/joint_states",
         "-r", "/compute_ik:=/offline/compute_ik", "-r", "/compute_fk:=/offline/compute_fk"]
ARM = {"l": [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)], "r": [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]}
NAMES = ARM["l"] + ARM["r"] + ["idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1",
                               "idx11_head_joint1", "idx13_head_joint3"]


class FakeRobot(Node):
    def __init__(self):
        super().__init__("offline_fake_robot")
        self.q = {n: 0.0 for n in NAMES}
        self.n_cmd = 0
        self.lock = threading.Lock()
        self.js = self.create_publisher(JointState, "/offline/joint_states", qos_profile_sensor_data)
        self.create_subscription(JointState, "/offline/joint_command", self.on_cmd,
                                 QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.create_service(GetPositionIK, "/offline/compute_ik", self.ik)
        self.create_service(GetPositionFK, "/offline/compute_fk", self.fk)
        self.create_timer(0.01, self.tick)

    def on_cmd(self, m):
        with self.lock:
            self.n_cmd += 1
            self.q.update(zip(m.name, m.position))

    def tick(self):
        with self.lock:
            msg = JointState(name=list(self.q), position=list(self.q.values()))
        msg.header.stamp = self.get_clock().now().to_msg()
        self.js.publish(msg)

    def ik(self, req, resp):
        side = "l" if req.ik_request.group_name.endswith("_l") else "r"
        p = req.ik_request.pose_stamped.pose.position
        sol = [p.x, p.y, p.z - 1.0, 0.1, 0.0, 0.0, 0.0]
        resp.solution.joint_state = JointState(name=ARM[side], position=sol)
        resp.error_code.val = 1
        return resp

    def fk(self, req, resp):
        for link in req.fk_link_names:
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            if link == "head_link3":
                ps.pose.position.x, ps.pose.position.z = 0.05, 1.45
            else:
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = 0.6, 0.5, 0.9
            ps.pose.orientation.w = 1.0
            resp.pose_stamped.append(ps)
        resp.fk_link_names = list(req.fk_link_names)
        resp.error_code.val = 1
        return resp


def call(node, client, req, timeout=10.0):
    assert client.wait_for_service(timeout_sec=timeout), client.srv_name
    ev, box = threading.Event(), {}
    client.call_async(req).add_done_callback(lambda f: (box.update(r=f.result()), ev.set()))
    ev.wait(timeout)
    return box.get("r")


def action(client, goal, timeout=120.0):
    assert client.wait_for_server(timeout_sec=30.0), "action server missing"
    ev, box, fb = threading.Event(), {}, []

    def on_goal(f):
        gh = f.result()
        if not gh.accepted:
            ev.set()
            return
        gh.get_result_async().add_done_callback(lambda r: (box.update(r=r.result().result), ev.set()))

    client.send_goal_async(goal, feedback_callback=lambda m: fb.append(m.feedback)).add_done_callback(on_goal)
    ev.wait(timeout)
    return box.get("r"), fb


def ros_run(pkg, exe, *extra):
    return subprocess.Popen(["ros2", "run", pkg, exe, "--ros-args", *REMAP, *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    rclpy.init()
    robot = FakeRobot()
    test = rclpy.create_node("offline_smoke")
    status = []
    test.create_subscription(CellStatus, "/g2cell/cell_status", status.append, 10)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(robot)
    ex.add_node(test)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.0)

    procs = [ros_run("g2cell_tasks", "cell_manager"),
             # fake move_group: the local URDF IK would never be "confirmed" by the fake /compute_ik, so plan with
             # the (fake) MoveIt only; no plan cache (never write the real one from a fake robot)
             ros_run("g2cell_motion", "motion_server", "-p", "max_joint_vel:=3.0", "-p", "min_segment_time:=0.2",
                     "-p", "gripper_time:=0.2", "-p", "local_ik:=false", "-p", "plan_cache:=''"),
             ros_run("g2cell_tasks", "pick_bag_server")]
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond)))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)

    try:
        move = ActionClient(test, MoveArm, "/g2cell/move_arm")
        pick = ActionClient(test, PickBag, "/g2cell/pick_bag")
        r, fb = action(move, MoveArm.Goal(arm="left", named_pose="home"))
        check("MoveArm home", r is not None and r.error_code == 0, r and r.message)
        r, _ = action(move, MoveArm.Goal(arm="left", named_pose="nope"))
        check("MoveArm unknown named pose", r is not None and r.error_code == MoveArm.Result.UNKNOWN_NAMED_POSE)
        r, _ = action(move, MoveArm.Goal(arm="left", auto_orientation=True,
                                         pose=PoseStamped().pose.__class__(), via=[]))
        check("MoveArm auto orientation", r is not None and r.error_code == 0 and abs(r.orientation_used.w) < 1.0,
              r and f"q=({r.orientation_used.x:.2f},{r.orientation_used.y:.2f},{r.orientation_used.z:.2f},"
                    f"{r.orientation_used.w:.2f})")
        n0 = robot.n_cmd
        r, fb = action(pick, PickBag.Goal(bag_id="any"))
        stages = [f.stage_name for f in fb]
        check("PickBag any -> unverified (no scorer, never fake success)",
              r is not None and r.result.outcome == r.result.OUTCOME_UNVERIFIED and not r.result.verified,
              r and f"{r.result.bag_id} {r.result.arm} {r.result.outcome_name}: {r.result.message}")
        check("PickBag feedback stages", stages == ["approach", "grasp", "lift", "transfer", "place", "retract"],
              str(stages))
        check("commands streamed on /offline/joint_command", robot.n_cmd - n0 > 100, f"{robot.n_cmd - n0} msgs")
        r, _ = action(pick, PickBag.Goal(bag_id="Bag_03"))
        check("PickBag Bag_03 (Y 0.98) unreachable", r is not None and r.result.outcome == r.result.OUTCOME_UNREACHABLE,
              r and r.result.message)
        r, _ = action(pick, PickBag.Goal(bag_id="Bag_01"))
        check("PickBag same bag again -> invalid", r is not None and r.result.outcome == r.result.OUTCOME_INVALID,
              r and r.result.message)
        r, _ = action(pick, PickBag.Goal(bag_id="Bag_02", target="rack_lane", lane="L9_9"))
        check("PickBag unknown lane -> invalid", r is not None and r.result.outcome == r.result.OUTCOME_INVALID,
              r and r.result.message)

        e = call(test, test.create_client(EpisodeControl, "/g2cell/episode_control"),
                 EpisodeControl.Request(command=0, episode_id="offline_ep"))
        check("EpisodeControl start", e is not None and e.success and e.episode_id == "offline_ep")
        ls = call(test, test.create_client(SetLineState, "/g2cell/set_line_state"), SetLineState.Request(command=0))
        check("SetLineState -> NOT_IMPLEMENTED", ls is not None and not ls.success
              and ls.status == ls.STATUS_NOT_IMPLEMENTED, ls and ls.message)
        sb = call(test, test.create_client(SpawnBags, "/g2cell/spawn_bags"), SpawnBags.Request(count=5, seed=1))
        check("SpawnBags -> NOT_IMPLEMENTED", sb is not None and not sb.success
              and sb.status == sb.STATUS_NOT_IMPLEMENTED, sb and sb.message)
        time.sleep(1.5)
        s = status[-1] if status else None
        check("CellStatus published", s is not None and s.episode_id == "offline_ep" and s.episode_active,
              s and f"picks_done={s.picks_done} fill={s.crate_fill_count}/{s.crate_fill_target} "
                    f"bags_on_table={s.bags_on_table}")

        dup = ros_run("g2cell_motion", "motion_server", "-r", "__node:=second_motion_server")
        try:
            out, _ = dup.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            dup.kill()
            out = "still running"
        check("single-publisher guard refuses a 2nd motion_server", dup.returncode == 2, out.strip()[-160:])
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        ex.shutdown()
        rclpy.try_shutdown()
    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; failed: {failed}" if failed else ""))
    if failed:
        for p in procs:
            print("----", p.args[2:4], "\n", (p.stdout.read() or "")[-3000:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
