"""Offline PickBag reach sweep over the datagen spawn region: for each grid point, the REAL PickBag stages and
motion_server planner (pickbag_replay.run_bag, fixture check and clear pose on) for one bag at that point, to
the pack carton ("good") and/or the rack tote L1_1 ("reject"). Tells which part of the table the planner
reliably completes, so datagen's randomisation region can be bounded by measurement (datagen 2026-10-11:
3 of the first 5 pilot seeds failed planning at the far corner / rack transfer).

    cp tools/perf/pickbag_replay.py tools/perf/reach_sweep.py output/perf/
    docker exec -u isaac-sim -e ROS_DOMAIN_ID=42 g2cell bash -lc 'source /workspace/devel/setup.bash && \
        source /g2cell_ws/install/setup.bash && cd /g2cell_output/perf && python3 reach_sweep.py \
        --xs=-0.30:0.15:0.075 --ys 0.52:0.80:0.07 --labels good,reject --tag pilot'

Writes /g2cell_output/perf/reach_sweep_<tag>.json: per point and label the outcome, failed stage, message, s.
Needs a move_group (the sim's, or a standalone one) on the same ROS domain.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading

import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor

import pickbag_replay as pr
from g2cell_tasks.config import load_task_config


def frange(spec: str):
    a, b, step = (float(v) for v in spec.split(":"))
    out, x = [], a
    while x <= b + 1e-9:
        out.append(round(x, 4))
        x += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/g2cell_ws/src/g2cell_tasks/config/tasks.yaml")
    ap.add_argument("--xs", default="-0.30:0.15:0.075")
    ap.add_argument("--ys", default="0.52:0.80:0.07")
    ap.add_argument("--labels", default="good,reject")
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()
    base = yaml.safe_load(open(a.tasks))
    rclpy.init()
    node = pr.build_node(0, fixtures=True)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not (node.ik.wait_for_service(timeout_sec=30) and node.fk.wait_for_service(timeout_sec=10)):
        sys.exit("no move_group")
    rows = []
    for label in a.labels.split(","):
        for y in frange(a.ys):
            for x in frange(a.xs):
                doc = copy.deepcopy(base)
                doc["bags"] = [{"id": "Bag_01", "variant": "standard", "label": label, "xy": [x, y], "top_z": 0.94}]
                cfg = load_task_config(doc)
                node.plan_cache.clear()
                r = pr.run_bag(node, cfg, cfg.bags.get("Bag_01"), lambda row: None)
                row = {"label": label, "x": x, "y": y, "outcome": r["outcome"], "failed_stage": r.get("failed_stage", ""),
                       "message": r.get("message", "")[:160], "s": r["s"], "ends_home": r.get("ends_home")}
                rows.append(row)
                print(json.dumps(row), flush=True)
    json.dump(rows, open(f"/g2cell_output/perf/reach_sweep_{a.tag}.json", "w"), indent=1)
    ex.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
