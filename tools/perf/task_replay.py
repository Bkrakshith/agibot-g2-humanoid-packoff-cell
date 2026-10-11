"""Replay the PickBag motion goals (g2cell_tasks waypoints, left arm, 3 table bags -> pack carton) through
motion_server's planner (_plan) against a standalone move_group, without the sim. Nothing is published.

    docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && \
        source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/task_replay.py --mode after'

--mode baseline: the ORIGINAL IkHelper.solve (random jitter, keep searching after a failed seeded attempt)
and head clearance through /compute_fk per path sample. --mode after: local IK + confirmation, local FK.
--episodes N with --cache FILE: repeated episodes reuse cached plans.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g2cell_interfaces.action import MoveArm
from g2cell_motion import motion_server as ms
from g2cell_motion.ik_client import IkHelper
from g2cell_motion.named_poses import load_named_poses
from g2cell_motion.plan_cache import PlanCache, layout_key
from g2cell_motion.robot import GROUPS, HOME_DEG, IK_LINKS, arm_joint_names, home_joints
from g2cell_motion.waist_policy import TwistLimits, WaistConfig
from g2cell_tasks.waypoints import CellTargets, pick_waypoints, place_waypoints

URDF = "/geniesim_assets/scenes/scene_g2cell/robot.urdf"
COLLIDERS = "/geniesim_assets/scenes/g2cell/cell_colliders.json"


def old_solve(self, arm, pos, quat, seed=None, attempts=8, timeout=0.1, jitter=0.4):
    """IkHelper.solve before the optimisation (verbatim logic): unseeded global random jitter."""
    js = self.joint_state()
    base = dict(zip(js.name, js.position))
    if seed:
        base.update(seed)
    names = self.arm_joint_names(arm)
    best, best_d = None, None
    for k in range(attempts):
        s = dict(base)
        if k:
            for n in names:
                s[n] = s[n] + random.uniform(-jitter, jitter)
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name = GROUPS[arm], IK_LINKS[arm]
        r.avoid_collisions = False
        r.timeout.nanosec = int(timeout * 1e9)
        r.robot_state.joint_state = JointState(name=list(s.keys()), position=[float(v) for v in s.values()])
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(c) for c in pos)
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w) = \
            (float(c) for c in quat)
        r.pose_stamped = ps
        res = self._call(req)
        self.n_ik += 1
        if res is None or res.error_code.val != 1:
            continue
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        sol = {n: sol[n] for n in names if n in sol}
        d = sum((sol[n] - base[n]) ** 2 for n in names)
        if best is None or d < best_d:
            best, best_d = sol, d
        if k == 0:
            break
    return best


def build(mode, cache, a_fallback=2):
    attrs = {k: v for k, v in vars(ms.MotionServer).items()
             if k in ("_clearance", "_plan", "_plan_uncached", "_goal_waist_mode", "_goal_dict")}
    if mode == "baseline":
        attrs["solve"] = old_solve
    Replay = type("Replay", (Node, IkHelper), attrs)
    node = Replay("g2cell_task_replay")
    node.init_ik()
    node.init_fk()
    node.n_ik = 0
    node.min_clear, node.ik_attempts = 0.20, 6
    node.waist_mode = ms.WAIST_FIRST
    node.waist_cfg, node.twist = WaistConfig(), TwistLimits()
    node.max_solves, node.max_seconds = 600, 60.0
    node.waist_fallback_step, node.waist_fallback_orients = 15.0, 0   # motion_server defaults
    from ament_index_python.packages import get_package_share_directory
    import os
    with open(os.path.join(get_package_share_directory("g2cell_motion"), "config", "named_poses.yaml")) as f:
        node.named = load_named_poses(yaml.safe_load(f))
    if mode == "after":
        node.init_local_ik(URDF, confirm=True, fallback_attempts=a_fallback)
    node.plan_cache = PlanCache(cache, layout_key(colliders=COLLIDERS), version=ms.PLAN_VERSION)
    import xml.etree.ElementTree as ET
    names = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
    st = {n: 0.0 for n in names}
    st["idx03_body_joint3"] = math.radians(20.0)
    for arm in ("left", "right"):
        st.update(home_joints(arm_joint_names(names, arm), HOME_DEG))
    node._js = JointState(name=list(st), position=list(st.values()))
    return node


def pose(xyz, quat=None):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(c) for c in xyz)
    if quat is not None:
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(c) for c in quat)
    return p


def episode(node, cfg_targets, bags, log):
    arm = "left"
    names = node.arm_joint_names(arm)
    wj = node.waist_cfg.joint
    meas = node.measured()
    current = {n: meas[n] for n in names}
    current[wj] = meas.get(wj, 0.0)
    rows = []

    def move(what, xyz=None, quat=None, via=(), named=""):
        nonlocal current
        g = MoveArm.Goal(arm=arm, frame=MoveArm.Goal.FRAME_CELL_TOOL)
        if named:
            g.named_pose = named
        else:
            g.pose, g.auto_orientation = pose(xyz, quat), quat is None
            g.via = [pose(v) for v in via]
        n0, t0 = node.n_ik + getattr(node, "ik_stats", {}).get("moveit_calls", 0), time.perf_counter()
        chain, clear, q, err = node._plan(g, arm, names, current)
        dt = time.perf_counter() - t0
        n1 = node.n_ik + getattr(node, "ik_stats", {}).get("moveit_calls", 0)
        rows.append({"goal": what, "ok": err is None, "s": round(dt, 3), "ik_calls": n1 - n0,
                     "clear": round(clear, 4), "err": None if err is None else err[1],
                     "end": None if err else {k: round(v, 5) for k, v in sorted(chain[-1].items())}})
        log(rows[-1])
        if err is None:
            current = {**current, **chain[-1]}
        return q if err is None else None

    for b in bags:
        pk = pick_waypoints(b["xy"], b["top_z"])
        pl = place_waypoints(b["xy"], pk["lift"][2], cfg_targets.point("crate", ""))
        q = move(f"{b['id']} approach", pk["approach"])
        if q is None:
            move(f"{b['id']} home", named="home")
            continue
        move(f"{b['id']} grasp", pk["grasp"], q)
        move(f"{b['id']} lift", pk["lift"], q)
        q2 = move(f"{b['id']} transfer", pl["above"], q, via=[pl["via"]])
        if q2 is None:
            q2 = move(f"{b['id']} transfer (reorient)", pl["above"], None, via=[pl["via"]])
        move(f"{b['id']} release", pl["release"], q2)
        move(f"{b['id']} retract", pl["retract"], q2)
        move(f"{b['id']} home", named="home")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="after", choices=("baseline", "after"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--cache", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fallback", type=int, default=2)
    a = ap.parse_args()
    rclpy.init()
    node = build(a.mode, a.cache or None, a.fallback)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not (node.ik.wait_for_service(timeout_sec=30) and node.fk.wait_for_service(timeout_sec=10)):
        sys.exit("no move_group")
    tasks = yaml.safe_load(open("/g2cell_ws/src/g2cell_tasks/config/tasks.yaml"))
    targets = CellTargets.from_dict(tasks["targets"])
    out = []
    for e in range(a.episodes):
        if not a.cache:
            node.plan_cache.clear()
        t0 = time.perf_counter()
        rows = episode(node, targets, tasks["bags"], lambda r: print(json.dumps({"ep": e, **{k: v for k, v in r.items()
                                                                                         if k != "end"}}), flush=True))
        out.append({"episode": e, "total_s": round(time.perf_counter() - t0, 2), "goals": rows,
                    "ok": sum(r["ok"] for r in rows), "n": len(rows)})
        print(json.dumps({"episode": e, "total_s": out[-1]["total_s"], "ok": out[-1]["ok"], "n": len(rows)}), flush=True)
    tag = a.tag or a.mode
    json.dump(out, open(f"/g2cell_output/perf/task_replay_{tag}.json", "w"), indent=1)
    ex.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
