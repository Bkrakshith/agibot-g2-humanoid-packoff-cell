"""IK reachability of the cell's key poses (MoveIt /compute_ik, move_group must be running).

Targets are fingertip points in the cell frame, gripper pointing down (several yaws and a
few tilts tried). Prints a table and writes JSON (default /g2cell_output/reach_check.json).
    ros2 run g2cell_demo reach_check --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json
import math
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from g2cell_demo.geometry import down_quat, end_link_for_tool
from g2cell_demo.ik_client import IkHelper

# pick table spawn zone (assets/pick_table/README.md): 0.50 (X) x 0.90 (Y) m centred on the
# table, bottom 0.89 m. "Above" = fingertips 0.10 m over the zone bottom.
SPAWN_HALF = (0.25, 0.45)
SPAWN_Z = 0.89
CLEAR = 0.10
WAIST_JOINT = "idx05_body_joint5"   # torso yaw about Z
WAIST_YAWS = [0, 30, 60, 90]        # degrees, toward +Y for the left arm
PACK = (0.60, 0.0, 0.82 + 0.224 + CLEAR)   # above the pack-position crate rim
YAWS = [0, 45, 90, -45, -90, 135, -135, 180]
TILTS = [0, 20, -20, 40]


class ReachCheck(Node, IkHelper):
    def __init__(self):
        super().__init__("g2cell_reach_check")
        self.out = self.declare_parameter("out", "/g2cell_output/reach_check.json").value
        cx, cy = self.declare_parameter("table_centre", [-0.05, 0.85]).value
        hx, hy = SPAWN_HALF
        self.corners = {}
        for tag, sx, sy in (("near_inner", 1, -1), ("far_inner", -1, -1),
                            ("near_outer", 1, 1), ("far_outer", -1, 1)):
            x, y = round(cx + sx * hx, 3), round(cy + sy * hy, 3)
            self.corners[f"spawn_{tag} ({x:+.2f},{y:+.2f})"] = (x, y)
        self.corners[f"spawn_centre ({cx:+.2f},{cy:+.2f})"] = (cx, cy)
        self.init_ik()

    def try_pose(self, arm, tool):
        """Fixed torso first; then turn the waist toward the target and retry. The target is
        expressed in base_link; a waist yaw w is emulated by seeding body_joint5 = w (the arm
        groups exclude the waist, so IK keeps it fixed at the seed)."""
        sign = 1 if arm == "left" else -1
        for waist in WAIST_YAWS:
            seed = {WAIST_JOINT: math.radians(sign * waist)}
            for tilt in TILTS:
                for yaw in YAWS:
                    q = down_quat(math.radians(yaw), math.radians(tilt))
                    sol = self.solve(arm, end_link_for_tool(tool, q), q, seed=seed, attempts=4)
                    if sol:
                        return {"ok": True, "waist_deg": sign * waist, "yaw_deg": yaw, "tilt_deg": tilt}
        return {"ok": False}

    def run(self):
        if not self.ik.wait_for_service(timeout_sec=60):
            self.get_logger().error("/compute_ik not available")
            return
        while self.joint_state() is None:
            threading.Event().wait(0.1)
        targets = {name: (x, y, SPAWN_Z + CLEAR) for name, (x, y) in self.corners.items()}
        targets["pack_crate (+0.60,0)"] = PACK
        rows = {}
        for name, tool in targets.items():
            rows[name] = {"tool_xyz": tool}
            for arm in ("left", "right"):
                rows[name][arm] = self.try_pose(arm, tool)
            l, r = rows[name]["left"], rows[name]["right"]
            fmt = lambda d: (f"yes (waist {d['waist_deg']}, yaw {d['yaw_deg']}, tilt {d['tilt_deg']})"
                             if d["ok"] else "NO")
            print(f"{name:34s} tool={tuple(round(c, 3) for c in tool)}  left: {fmt(l):36s} right: {fmt(r)}",
                  flush=True)
        json.dump(rows, open(self.out, "w"), indent=1)
        print(f"wrote {self.out}")


def main():
    rclpy.init()
    node = ReachCheck()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
