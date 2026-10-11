"""Pick-column IK (approach -> grasp -> lift, one orientation, waist held) per waist yaw x orientation for a
bag, from home, with the same seeds / twist filter as motion_server. ROS_DOMAIN_ID 42 move_group needed.
    python3 /g2cell_output/perf/column_diag.py Bag_02"""
import json, math, sys, threading
import rclpy, yaml
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0, "/g2cell_output/perf")
from pickbag_replay import ReplayClient, build_node
from g2cell_motion.planner import plan_chain_waist
from g2cell_motion.robot import HOME_DEG, arm_joint_names, home_joints
from g2cell_tasks.waypoints import pick_waypoints

bag_id = sys.argv[1]
rclpy.init(); node = build_node(); ex = MultiThreadedExecutor(); ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start(); node.ik.wait_for_service(timeout_sec=30)
mc = ReplayClient(node, lambda r: None)
names = arm_joint_names(list(mc.state), "left"); wj = node.waist_cfg.joint
bag = [b for b in yaml.safe_load(open("/g2cell_ws/src/g2cell_tasks/config/tasks.yaml"))["bags"] if b["id"] == bag_id][0]
pk = pick_waypoints(bag["xy"], bag["top_z"])
cur = {**{n: mc.state[n] for n in names}, wj: 0.0}
for wdeg in range(0, 101, 5):
    w = math.radians(wdeg)
    r = plan_chain_waist([pk["approach"]], lambda pos, q, s: node.solve("left", pos, q, seed=s, attempts=1),
                         node._clearance("left"), seed=cur, joint_names=names, home=home_joints(names, HOME_DEG),
                         waist_now=0.0, mode="absolute", waist_yaw=w, cfg=node.waist_cfg, limits=node.twist,
                         start=cur, lookahead=[pk["grasp"], pk["lift"]])
    print(json.dumps({"bag": bag_id, "waist": wdeg, "ok": r.ok, "reason": r.reason, "label": r.label,
                      "clear": round(r.clearance, 3),
                      "arm": [round(math.degrees(r.solutions[0][n])) for n in names] if r.ok else None}), flush=True)
ex.shutdown(); rclpy.try_shutdown()
