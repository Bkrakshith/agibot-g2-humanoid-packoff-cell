"""Health check of the running cell stack: lists the required topics/services/actions and what is
missing (read-only graph introspection; publishes nothing).

    ros2 run g2cell_bringup check_stack            # exit 0 = all present, 1 = something missing
    ros2 run g2cell_bringup check_stack --wait 5   # give discovery longer
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

# name, what provides it, optional exact publisher count
REQUIRED = {
    "topics": [
        {"name": "/joint_states", "by": "sim (stage1_up.sh window 'sim')"},
        {"name": "/joint_command", "by": "motion_server (exactly one publisher: quit teleop/demo)", "publishers": 1},
        {"name": "/tf", "by": "sim"},
        {"name": "/genie_sim/head_front_camera_rgb/image_raw", "by": "sim render + head_cam_tf_relay"},
        {"name": "/g2cell/cell_status", "by": "g2cell_tasks cell_manager"},
    ],
    "services": [
        {"name": "/compute_ik", "by": "MoveIt move_group (wbc.launch.py use_ros2_control:=false)"},
        {"name": "/compute_fk", "by": "MoveIt move_group"},
        {"name": "/g2cell/set_line_state", "by": "g2cell_tasks cell_manager"},
        {"name": "/g2cell/spawn_bags", "by": "g2cell_tasks cell_manager"},
        {"name": "/g2cell/episode_control", "by": "g2cell_tasks cell_manager"},
    ],
    "actions": [
        {"name": "/g2cell/move_arm", "by": "g2cell_motion motion_server"},
        {"name": "/g2cell/pick_bag", "by": "g2cell_tasks pick_bag_server"},
    ],
}


@dataclass
class Graph:
    topics: dict = field(default_factory=dict)      # name -> number of publishers
    services: set = field(default_factory=set)
    actions: set = field(default_factory=set)


@dataclass
class Row:
    kind: str
    name: str
    ok: bool
    by: str
    detail: str = ""


def evaluate(required: dict, g: Graph):
    """-> (rows, missing rows)."""
    rows = []
    for t in required["topics"]:
        n = g.topics.get(t["name"])
        want = t.get("publishers")
        if n is None or n == 0:
            rows.append(Row("topic", t["name"], False, t["by"], "no publisher"))
        elif want is not None and n != want:
            rows.append(Row("topic", t["name"], False, t["by"], f"{n} publishers, need exactly {want}"))
        else:
            rows.append(Row("topic", t["name"], True, t["by"], f"{n} publisher(s)"))
    for kind, have in (("services", g.services), ("actions", g.actions)):
        for s in required[kind]:
            ok = s["name"] in have
            rows.append(Row(kind[:-1], s["name"], ok, s["by"], "" if ok else "not advertised"))
    return rows, [r for r in rows if not r.ok]


def format_report(rows, missing) -> str:
    lines = [f"{'OK' if r.ok else 'MISSING':8s} {r.kind:8s} {r.name:45s} {r.detail:32s} <- {r.by}" for r in rows]
    lines.append(f"{len(rows) - len(missing)}/{len(rows)} present" + ("" if not missing else
                 "; missing: " + ", ".join(m.name for m in missing)))
    return "\n".join(lines)


def read_graph(node) -> Graph:
    topics = {}
    for name, _ in node.get_topic_names_and_types():
        topics[name] = node.count_publishers(name)
    services = {n for n, _ in node.get_service_names_and_types()}
    actions = {n[: -len("/_action/send_goal")] for n in services if n.endswith("/_action/send_goal")}
    return Graph(topics, services, actions)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="check_stack", description=__doc__.splitlines()[0])
    ap.add_argument("--wait", type=float, default=2.0, help="seconds of discovery before reading the graph")
    a, ros_args = ap.parse_known_args(argv)
    import rclpy  # noqa: PLC0415 - keep the pure part importable without ROS

    rclpy.init(args=ros_args)
    node = rclpy.create_node("g2cell_check_stack")
    try:
        end = time.monotonic() + a.wait
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        rows, missing = evaluate(REQUIRED, read_graph(node))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    print(format_report(rows, missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
