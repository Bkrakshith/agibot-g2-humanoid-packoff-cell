"""Bag_02 transfer from its (elbow-up, waist 54 deg) lift posture: which via / waist sequence passes the head
check. Prints reason + best clearance per variant."""
import json, math, sys, threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0, "/g2cell_output/perf")
from pickbag_replay import ReplayClient, build_node
from g2cell_motion.planner import plan_chain_waist
from g2cell_motion.robot import HOME_DEG, arm_joint_names, home_joints

rclpy.init(); node = build_node(); ex = MultiThreadedExecutor(); ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start(); node.ik.wait_for_service(timeout_sec=30)
mc = ReplayClient(node, lambda r: None)
names = arm_joint_names(list(mc.state), "left"); wj = node.waist_cfg.joint
lift = dict(zip(names, [math.radians(d) for d in json.loads(sys.argv[1])]))
w0 = math.radians(float(sys.argv[2]))
cur = {**lift, wj: w0}
mc.state.update(cur); mc._sync()
above, rel, ret = (0.6, 0.0, 1.24), (0.6, 0.0, 1.10), (0.6, 0.0, 1.27)
lx, ly = -0.25, 0.62
variants = {"default via": [(0.175, 0.31, 1.19)], "no via": [], "via z1.11": [(0.175, 0.31, 1.11)],
            "via z1.30": [(0.175, 0.31, 1.30)], "via near pick": [(-0.05, 0.50, 1.19)],
            "2 vias": [(0.0, 0.45, 1.19), (0.35, 0.2, 1.22)], "via front": [(0.35, 0.35, 1.19)]}
clr = node._clearance("left")
for name, vias in variants.items():
    for wy in (0.0, math.radians(15)):
        r = plan_chain_waist(vias + [above], lambda pos, q, s: node.solve("left", pos, q, seed=s, attempts=1), clr,
                             seed=cur, joint_names=names, home=home_joints(names, HOME_DEG), waist_now=w0,
                             mode="absolute", waist_yaw=wy, cfg=node.waist_cfg, limits=node.twist, start=cur,
                             lookahead=[rel, ret], fallback_orients=24, min_clearance=0.20)
        print(json.dumps({"variant": name, "waist_end": round(math.degrees(wy)), "ok": r.ok, "reason": r.reason,
                          "label": r.label, "clear": round(r.clearance, 3), "yaws": [round(math.degrees(y)) for y in r.waist_yaws],
                          "twist": r.twist_rejected}), flush=True)
ex.shutdown(); rclpy.try_shutdown()
