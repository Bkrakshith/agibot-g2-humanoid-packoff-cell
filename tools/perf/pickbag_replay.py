"""Faithful offline replay of the PickBag action: the REAL g2cell_tasks PickPlaceMachine + MotionRunner
(pick_bag_server) driving motion_server's own planner (_plan) against a standalone move_group, without the
sim. A fake MotionClient plans each MoveArm goal with motion_server's code and "executes" it by jumping the
joint state to the planned end (nothing is published). Start state = scene_g2cell.yaml init_joint_pos
(arms home, body joint3 20 deg, head yaw 40 / pitch 28 deg).

    cp tools/perf/pickbag_replay.py output/perf/
    docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && \
        source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/pickbag_replay.py --runs 5 --tag after'

Each run plans every bag in tasks.yaml (bag_id given explicitly) with a cleared plan cache, from the start
state, and writes /g2cell_output/perf/pickbag_replay_<tag>.json (per goal: what, waist mode, ok, error, time,
end waist / arm joints) plus the final arm pose after the task (to check recovery retracts home).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g2cell_interfaces.action import MoveArm
from g2cell_motion import motion_server as ms
from g2cell_motion.ik_client import IkHelper
from g2cell_motion.named_poses import load_named_poses
from g2cell_motion.plan_cache import PlanCache, layout_key
from g2cell_motion.robot import HOME_DEG, arm_joint_names, home_joints
from g2cell_motion.waist_policy import TwistLimits, WaistConfig
from g2cell_tasks import pick_bag_server as pbs
from g2cell_tasks.config import load_task_config
from g2cell_tasks.reachability import select_arm, waist_for
from g2cell_tasks.state_machine import Hooks, PickPlaceMachine
from g2cell_tasks.waypoints import pick_waypoints, place_waypoints

URDF = "/geniesim_assets/scenes/scene_g2cell/robot.urdf"
COLLIDERS = "/geniesim_assets/scenes/g2cell/cell_colliders.json"
CLEAR_POSE = "clear"   # pick_bag_server approach_clear_pose default
INIT_DEG = {"idx03_body_joint3": 20.0, "idx11_head_joint1": 40.0, "idx13_head_joint3": 28.0}


def build_node(ik_fallback: int = 0, fixtures: bool = True):
    attrs = {k: v for k, v in vars(ms.MotionServer).items()
             if k in ("_clearance", "_plan", "_plan_uncached", "_goal_waist_mode", "_goal_dict")}
    Replay = type("PickBagReplay", (Node, IkHelper), attrs)
    node = Replay("g2cell_pickbag_replay")
    node.init_ik()
    node.init_fk()
    node.n_ik = 0
    node.min_clear, node.ik_attempts = 0.20, 6
    node.waist_mode = ms.WAIST_FIRST
    node.waist_cfg, node.twist = WaistConfig(), TwistLimits()
    node.max_solves, node.max_seconds = 600, 60.0
    node.waist_fallback_step, node.waist_fallback_orients = 15.0, 0   # motion_server defaults
    with open(os.path.join(get_package_share_directory("g2cell_motion"), "config", "named_poses.yaml")) as f:
        node.named = load_named_poses(yaml.safe_load(f))
    node.init_local_ik(URDF, confirm=True, fallback_attempts=ik_fallback)
    node.plan_cache = PlanCache(None, layout_key(colliders=COLLIDERS), version=ms.PLAN_VERSION)
    node.fixture_margin = 0.04          # motion_server defaults (fixture_check true)
    node.fixtures = None
    if fixtures:
        from g2cell_motion.clearance import load_fixture_boxes  # noqa: PLC0415
        from g2cell_motion.geometry import BASE_Z  # noqa: PLC0415
        node.fixtures = [{**b, "c": [b["c"][0], b["c"][1], b["c"][2] - BASE_Z]} for b in load_fixture_boxes(COLLIDERS)]
    return node


def init_state():
    names = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
    st = {n: 0.0 for n in names}
    for k, d in INIT_DEG.items():
        if k in st:
            st[k] = math.radians(d)
    for arm in ("left", "right"):
        st.update(home_joints(arm_joint_names(names, arm), HOME_DEG))
    return st


class ReplayClient:
    """MotionClient stand-in: plans with motion_server._plan, executes by jumping to the plan's end."""

    def __init__(self, node, log):
        self.node, self.log, self.rows = node, log, []
        self.state = init_state()
        self._sync()

    def _sync(self):
        self.node._js = JointState(name=list(self.state), position=[float(v) for v in self.state.values()])

    def wait(self, timeout=5.0):
        return True

    def _plan(self, g: MoveArm.Goal, what: str):
        arm = g.arm
        names = arm_joint_names(list(self.state), arm)
        wj = self.node.waist_cfg.joint
        current = {n: self.state[n] for n in names}
        current[wj] = self.state.get(wj, 0.0)
        t0 = time.perf_counter()
        try:
            chain, clear, quat, err = self.node._plan(g, arm, names, current)
        except Exception as e:  # noqa: BLE001 - motion_server reports these as INVALID_GOAL
            chain, clear, quat, err = [], -1.0, None, (MoveArm.Result.INVALID_GOAL, f"error: {e}")
        dt = time.perf_counter() - t0
        r = MoveArm.Result()
        if err is None:
            self.state.update(chain[-1])
            self._sync()
            r.error_code, r.message = MoveArm.Result.OK, "ok"
            if quat is not None:
                (r.orientation_used.x, r.orientation_used.y, r.orientation_used.z,
                 r.orientation_used.w) = quat
        else:
            r.error_code, r.message = int(err[0]), err[1]
        r.waist_yaw = float(self.state.get(wj, 0.0))
        r.head_clearance = float(clear)
        row = {"what": what, "arm": arm, "waist_mode": g.waist_mode or "(default)", "ok": err is None,
               "err": None if err is None else err[1], "s": round(dt, 3),
               "waist_deg": round(math.degrees(self.state.get(wj, 0.0)), 1),
               "arm_deg": [round(math.degrees(self.state[n]), 1) for n in names]}
        self.rows.append(row)
        self.log(row)
        return r

    def move_tool(self, arm, xyz, quat=None, via=(), duration=0.0, gripper=MoveArm.Goal.GRIPPER_KEEP,
                  cancelled=lambda: False, waist_mode="", waist_yaw=0.0, lookahead=()):
        g = MoveArm.Goal(arm=arm, frame=MoveArm.Goal.FRAME_CELL_TOOL, pose=pbs_pose(xyz, quat),
                         auto_orientation=quat is None, via=[pbs_pose(v, quat) for v in via],
                         waist_mode=waist_mode, waist_yaw=float(waist_yaw))
        if lookahead:
            g.lookahead = [pbs_pose(v, quat) for v in lookahead]
        what = f"tool {tuple(round(c, 3) for c in xyz)}" + (f" via {[tuple(round(c, 3) for c in v) for v in via]}"
                                                         if via else "") + (" quat" if quat is not None else " auto") \
            + (f" ahead {len(lookahead)}" if lookahead else "")
        return self._plan(g, what)

    def move_named(self, arm, name, duration=0.0, cancelled=lambda: False, waist_mode=""):
        return self._plan(MoveArm.Goal(arm=arm, named_pose=name, waist_mode=waist_mode), f"named {name}")

    def gripper(self, arm, close, cancelled=lambda: False):
        r = MoveArm.Result()
        r.error_code, r.message = MoveArm.Result.OK, "ok"
        self.rows.append({"what": "gripper " + ("close" if close else "open"), "arm": arm, "ok": True})
        return r


def pbs_pose(xyz, quat=None):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(c) for c in xyz)
    if quat is not None:
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(c) for c in quat)
    return p


def run_bag(node, cfg, bag, log):
    """The pick_bag_server._run path for one bag (no action plumbing)."""
    t0 = time.perf_counter()
    target, lane = cfg.sort.resolve("", "", bag.label)
    choice = select_arm(cfg.reach, "table", bag.xy, target, "", lane=lane)
    out = {"bag": bag.id, "target": target, "lane": lane}
    if choice.arm is None:
        out.update(outcome="unreachable", message=choice.reason, s=round(time.perf_counter() - t0, 3), goals=[])
        return out
    mc = ReplayClient(node, log)
    point = cfg.targets.point(target, lane)
    plan_fn = getattr(pbs, "plan_task", None)
    if plan_fn is not None:            # after the fix: the server builds ctx through plan_task()
        ctx = plan_fn(cfg, bag, choice.arm, target, lane)
    else:
        pick = pick_waypoints(bag.xy, bag.top_z)
        ctx = {"arm": choice.arm, "bag": bag, "target": target, "lane": lane, "pick": pick,
               "place": place_waypoints(bag.xy, pick["lift"][2], point), "quat": None,
               "waist_pick": waist_for("table", cfg.reach.waist_first),
               "waist_place": waist_for(target, cfg.reach.waist_first)}
    stages = []
    runner = pbs.MotionRunner(mc, cancelled=lambda: False, sleep=lambda s: None, clear_pose=CLEAR_POSE)
    tr = PickPlaceMachine(runner, Hooks(), cfg.max_grasp_attempts,
                          lambda st, a: stages.append(st.value)).run(ctx)
    arm_names = arm_joint_names(list(mc.state), choice.arm)
    home = [round(d, 1) for d in HOME_DEG]
    final = [round(math.degrees(mc.state[n]), 1) for n in arm_names]
    out.update(arm=choice.arm, outcome=tr.outcome, failed_stage=tr.failed_stage, message=tr.message,
               stages=stages, s=round(time.perf_counter() - t0, 3), goals=mc.rows,
               final_arm_deg=final, final_waist_deg=round(math.degrees(mc.state.get(node.waist_cfg.joint, 0.0)), 1),
               ends_home=all(abs(a - b) < 0.5 for a, b in zip(final, home)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--bags", default="", help="comma list (default: all in tasks.yaml)")
    ap.add_argument("--tasks", default="/g2cell_ws/src/g2cell_tasks/config/tasks.yaml")
    ap.add_argument("--ik-fallback", type=int, default=0, help="motion_server ik_fallback_attempts")
    ap.add_argument("--no-fixtures", action="store_true", help="motion_server fixture_check:=false")
    ap.add_argument("--clear-pose", default=None, help="pick_bag_server approach_clear_pose ('' = off)")
    ap.add_argument("--fallback-orients", type=int, default=0, help="experiment: override plan_chain_waist default")
    ap.add_argument("--fallback-step", type=float, default=0.0, help="experiment: override plan_chain_waist default")
    a = ap.parse_args()
    if a.clear_pose is not None:
        global CLEAR_POSE
        CLEAR_POSE = a.clear_pose
    if a.fallback_orients or a.fallback_step:
        import functools
        kw = {}
        if a.fallback_orients:
            kw["fallback_orients"] = a.fallback_orients
        if a.fallback_step:
            kw["fallback_step_deg"] = a.fallback_step
        ms.plan_chain_waist = functools.partial(ms.plan_chain_waist, **kw)
    rclpy.init()
    node = build_node(a.ik_fallback, fixtures=not a.no_fixtures)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not (node.ik.wait_for_service(timeout_sec=30) and node.fk.wait_for_service(timeout_sec=10)):
        sys.exit("no move_group")
    runs = []
    for k in range(a.runs):
        cfg = load_task_config(yaml.safe_load(open(a.tasks)))
        bags = [b for b in cfg.bags.on_table() if not a.bags or b.id in a.bags.split(",")]
        res = []
        for b in bags:
            node.plan_cache.clear()
            r = run_bag(node, cfg, b, lambda row: print(json.dumps(row), flush=True))
            print(json.dumps({"run": k, **{x: r[x] for x in r if x != "goals"}}), flush=True)
            res.append(r)
        runs.append(res)
    summary = [{"bag": r["bag"], "outcomes": [run[i]["outcome"] for run in runs],
                "s": [run[i]["s"] for run in runs], "ends_home": [run[i].get("ends_home") for run in runs],
                "identical_goals": all(json.dumps([{k: v for k, v in g.items() if k != "s"} for g in run[i]["goals"]])
                                       == json.dumps([{k: v for k, v in g.items() if k != "s"}
                                                      for g in runs[0][i]["goals"]]) for run in runs)}
               for i, r in enumerate(runs[0])]
    print(json.dumps({"summary": summary}, indent=1), flush=True)
    json.dump({"runs": runs, "summary": summary}, open(f"/g2cell_output/perf/pickbag_replay_{a.tag}.json", "w"),
              indent=1)
    ex.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
