"""Which PickBag transfer waypoint fails, and why: IK (local + /compute_ik confirm) per point x waist yaw x
orientation from the lift posture, against a standalone move_group (ROS_DOMAIN_ID 42). Prints a table.

    docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && \
        source /g2cell_ws/install/setup.bash && python3 /g2cell_output/perf/transfer_diag.py'
"""
from __future__ import annotations

import json
import math
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, "/g2cell_output/perf")
from pickbag_replay import ReplayClient, build_node  # noqa: E402

from g2cell_motion.geometry import end_link_for_tool  # noqa: E402
from g2cell_motion.planner import orient_candidates  # noqa: E402
from g2cell_motion.robot import HOME_DEG, arm_joint_names, home_joints  # noqa: E402
from g2cell_motion.waist_policy import TwistLimits, best_solution  # noqa: E402


def main():
    rclpy.init()
    node = build_node()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.ik.wait_for_service(timeout_sec=30)
    mc = ReplayClient(node, lambda r: None)
    names = arm_joint_names(list(mc.state), "left")
    home = home_joints(names, HOME_DEG)
    wj = node.waist_cfg.joint
    pts = {"lift Bag_01": (0.05, 0.62, 1.11), "via Bag_01": (0.325, 0.31, 1.19),
           "above rim1.07": (0.60, 0.0, 1.24), "release rim1.07": (0.60, 0.0, 1.10),
           "retract rim1.07": (0.60, 0.0, 1.27), "above rim1.044": (0.60, 0.0, 1.214),
           "above_crate named": (0.60, 0.0, 1.144), "via Bag_02": (0.175, 0.31, 1.19),
           "via low": (0.325, 0.31, 1.11), "above low 1.17": (0.60, 0.0, 1.17)}
    rows = []
    for label, p in pts.items():
        for wdeg in (0, 15, 30, 45, 60, 78.5):
            ok_or, tw = [], 0
            for o, q in orient_candidates(yaw_offset=math.radians(wdeg)):
                tgt = end_link_for_tool(p, q)
                cands = []
                for sd in (home,):
                    sol = node.solve("left", tgt, q, seed={**mc.state, **sd, wj: math.radians(wdeg)}, attempts=1)
                    cands.append(None if sol is None else {n: sol[n] for n in names})
                best, rej = best_solution(cands, home, home, names, TwistLimits())
                tw += len(rej)
                if best is not None:
                    ok_or.append(o)
            rows.append({"point": label, "xyz": p, "waist": wdeg, "n_ok": len(ok_or), "first": ok_or[:4], "twist": tw})
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open("/g2cell_output/perf/transfer_diag.json", "w"), indent=1)
    ex.shutdown()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
