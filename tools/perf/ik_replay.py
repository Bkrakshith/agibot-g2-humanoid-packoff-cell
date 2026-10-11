"""Replay packing_demo's planning (Layout v2 targets) against a standalone move_group, without the sim.

Runs INSIDE the g2cell container on a private ROS domain (no /joint_states, no /odom, nothing published):

    docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && \
        source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/ik_replay.py --runs 1 --tag baseline'

It binds PackingDemo's planning methods (plan_left / plan_right + whatever they call) onto a bare node that
only has /compute_ik + /compute_fk clients, injects the joint state the demo sees at planning time (drive
tuck arms, head 0/30 deg, waist 0, body_joint3 20 deg; base pose from take3), and instruments:
  - every /compute_ik call (count, wall time, success),
  - every fixture/head path check (count, wall time),
  - total planning time per arm, the demo's 3x retry loop included.
Works with the original (randomised) code and the optimised one; the latter's counters are read if present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g2cell_demo import packing_demo as pdm
from g2cell_demo.env_check import EnvChecker, joint_origin
from g2cell_motion.ik_client import IkHelper

URDF = "/geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf"
COLLIDERS = "/geniesim_assets/scenes/g2cell/cell_colliders.json"
BASE = (-0.009389638900756836, 0.00018552300753071904, 0.0)   # take3 base_for_ik


def movable_joints(urdf):
    import xml.etree.ElementTree as ET
    r = ET.parse(urdf).getroot()
    return [j.get("name") for j in r.findall("joint") if j.get("type") not in ("fixed", "floating", "planar")]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.ik_n = self.ik_ok = 0
        self.ik_t = 0.0
        self.ik_times = []
        self.chk_n = 0
        self.chk_t = 0.0

    def as_dict(self):
        ts = sorted(self.ik_times)
        pct = (lambda q: ts[min(len(ts) - 1, int(q * len(ts)))] if ts else 0.0)
        return {"ik_calls": self.ik_n, "ik_ok": self.ik_ok, "ik_time_s": round(self.ik_t, 2),
                "ik_ms_p50": round(1e3 * pct(0.5), 1), "ik_ms_p90": round(1e3 * pct(0.9), 1),
                "ik_ms_max": round(1e3 * (ts[-1] if ts else 0), 1),
                "path_checks": self.chk_n, "path_check_time_s": round(self.chk_t, 2)}


SKIP = {"__init__", "run", "move_joints", "exec_left", "exec_right", "_hold_loop", "gripper", "pause",
        "_publish", "_set"}


def build_node(stats, mode="baseline", cache=None, confirm=True):
    class Replay(Node, IkHelper):
        pass
    # borrow the demo's planning code (plain functions in py3): everything PackingDemo defines itself
    for name, v in vars(pdm.PackingDemo).items():
        if name not in SKIP and not name.startswith("__"):
            setattr(Replay, name, v)
    node = Replay("g2cell_ik_replay")
    node.init_ik()
    node.init_fk()
    names = movable_joints(URDF)
    js = JointState(name=names, position=[0.0] * len(names))
    st = dict(zip(names, js.position))
    for a, pre in (("l", "idx2"), ("r", "idx6")):
        for k, v in pdm.TUCK_DEG.items():
            st[f"{pre}{k}_arm_{a}_joint{k}"] = math.radians(v)
    st["idx03_body_joint3"] = math.radians(20.0)
    st.update({k: math.radians(v) for k, v in pdm.HEAD.items()})
    st[pdm.WAIST] = 0.0
    node._js = JointState(name=list(st), position=list(st.values()))
    node.out_dir = "/tmp"
    node.tote = "L1_1"
    node.env_margin = 0.015
    node.flip = 1.5
    node.why = {}
    node.base = BASE
    node.env = EnvChecker(URDF, COLLIDERS)
    node.fixed = dict(st)
    node.waist_axis = tuple(joint_origin(node.env, pdm.WAIST, node.fixed)[:2])
    node.rest = {a: node.joints_deg(a, pdm.TUCK_DEG) for a in ("left", "right")}
    if mode == "after":
        from g2cell_motion.plan_cache import PlanCache, layout_key
        node.init_local_ik(URDF, confirm=confirm, fallback_attempts=0)
        node.plan_retries, node.fallback_attempts = 1, 3
        node.layout = layout_key(colliders=COLLIDERS)
        node.plan_cache = PlanCache(cache, node.layout, version=pdm.PLAN_VERSION)
        node.plan_cache_reset()

    # instrument /compute_ik
    orig_call = node._call

    def timed_call(req, timeout=2.0):
        t0 = time.perf_counter()
        res = orig_call(req, timeout)
        dt = time.perf_counter() - t0
        with stats.lock:
            stats.ik_n += 1
            stats.ik_t += dt
            stats.ik_times.append(dt)
            stats.ik_ok += int(res is not None and res.error_code.val == 1)
        return res
    node._call = timed_call
    # instrument the fixture/head path checks
    orig_pc = pdm.path_check

    def timed_pc(*a, **k):
        t0 = time.perf_counter()
        r = orig_pc(*a, **k)
        with stats.lock:
            stats.chk_n += 1
            stats.chk_t += time.perf_counter() - t0
        return r
    pdm.path_check = timed_pc
    return node


def run_arm(node, arm, stats, retries=3, mode="baseline"):
    homes = node.joints_deg(arm, pdm.HOME_DEG)
    s0 = dict(stats.as_dict())
    ik0 = dict(getattr(node, "ik_stats", {}) or {})
    t0 = time.perf_counter()
    res = path = None
    tries, how = 0, None
    if mode == "after":
        res, path, how = node.plan_arm(arm, homes)
        tries = 1
    else:
        for _ in range(retries):     # the demo's "IK seeds are randomised" retry loop
            tries += 1
            res, path = (node.plan_left if arm == "left" else node.plan_right)(homes)
            if path:
                break
    dt = time.perf_counter() - t0
    ik1 = dict(getattr(node, "ik_stats", {}) or {})
    local = {k: round(ik1[k] - ik0.get(k, 0), 3) for k in ik1} if ik1 else None
    s1 = stats.as_dict()
    d = {k: (round(s1[k] - s0[k], 2) if k in ("ik_calls", "ik_ok", "ik_time_s", "path_checks", "path_check_time_s")
             else s1[k]) for k in s1}
    sig = None
    if path:
        sig = [[round(v, 4) for _, v in sorted(p.items())] for p in path]
    return {"arm": arm, "ok": bool(path), "wall_s": round(dt, 2), "tries": tries, "how": how, "result": res, **d,
            "path_sig": sig, "local": local}


def fk_check(node, n=20):
    """Local URDF FK vs /compute_fk for random arm/waist states (the local model must be MoveIt's)."""
    import numpy as np
    from g2cell_motion.robot import CHECK_LINKS, HEAD_LINK
    rng = np.random.default_rng(0)
    links = CHECK_LINKS["left"] + CHECK_LINKS["right"] + [HEAD_LINK]
    worst = 0.0
    for _ in range(n):
        j = {pdm.WAIST: float(rng.uniform(-1.7, 1.7))}
        for arm in ("left", "right"):
            for k, nme in enumerate(node.arm_joint_names(arm)):
                j[nme] = float(rng.uniform(-1.0, 1.0)) + math.radians((0, -45, 0, -75, 0, -55, 0)[k])
        mv = node.fk_poses(j, links)
        loc = node.fk_local(j, links)
        for l in links:
            p = mv[l].position
            worst = max(worst, float(np.linalg.norm(loc[l][:3, 3] - np.array([p.x, p.y, p.z]))))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--out", default="/g2cell_output/perf")
    ap.add_argument("--mode", default="baseline", choices=("baseline", "after"))
    ap.add_argument("--cache", default="", help="plan cache file (after mode); empty = no cache (cold runs)")
    ap.add_argument("--no-confirm", action="store_true", help="skip the /compute_ik confirmation")
    ap.add_argument("--base-offset", type=float, nargs=3, default=None, help="shift the base (m, m, deg)")
    a = ap.parse_args()
    rclpy.init()
    stats = Stats()
    node = build_node(stats, a.mode, a.cache or None, not a.no_confirm)
    if a.base_offset:
        node.base = (BASE[0] + a.base_offset[0], BASE[1] + a.base_offset[1], BASE[2] + math.radians(a.base_offset[2]))
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not (node.ik.wait_for_service(timeout_sec=30) and node.fk.wait_for_service(timeout_sec=10)):
        print("no move_group", file=sys.stderr)
        sys.exit(1)
    runs = []
    if a.mode == "after":
        print(json.dumps({"fk_local_vs_moveit_max_m": fk_check(node)}), flush=True)
    for r in range(a.runs):
        if hasattr(node, "plan_cache_reset"):
            node.plan_cache_reset()
        if a.mode == "after" and a.cache:
            from g2cell_motion.plan_cache import PlanCache
            node.plan_cache = PlanCache(a.cache, node.layout, version=pdm.PLAN_VERSION)   # as a fresh process would
        elif a.mode == "after":
            node.plan_cache.clear()                     # cold run: no cached plans
        row = {"run": r}
        for arm in a.arms.split(","):
            row[arm] = run_arm(node, arm, stats, mode=a.mode)
            print(json.dumps({"run": r, **{k: v for k, v in row[arm].items() if k not in ("path_sig", "result")}}),
                  flush=True)
        row["total_s"] = round(sum(row[x]["wall_s"] for x in a.arms.split(",")), 2)
        runs.append(row)
    os.makedirs(a.out, exist_ok=True)
    out = f"{a.out}/replay_{a.tag}.json"
    json.dump(runs, open(out, "w"), indent=1, default=str)
    print(f"wrote {out}; totals {[r['total_s'] for r in runs]}")
    ex.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
