"""Scripted pick-and-place-like motion of the G2 in the pack-off cell (for video renders).

Left arm: reach over a bag on the pick table, close the gripper, lift, swing above the crate at
the pack position, open, retract, home. Then the right arm does the same with its own target.
Each Cartesian waypoint (fingertip point in the cell frame, gripper pointing down) is turned into
arm joints with MoveIt /compute_ik (seeded with the previous waypoint's solution), and the arm
moves between them by minimum-jerk joint-space interpolation, streamed at 50 Hz on
/joint_command (best-effort, same message layout as g2cell_teleop). Must be the ONLY
/joint_command publisher (stop the keyboard teleop first).

    ros2 run g2cell_demo pick_place_demo --ros-args -p use_sim_time:=true
Real grasping is not attempted: the rigid placeholder bags only render at their spawn pose
(Genie Sim's render stage does not receive scene rigid-body poses).
"""
from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from g2cell_motion.clearance import path_clearance
from g2cell_motion.geometry import duration_for
from g2cell_motion.ik_client import IkHelper
from g2cell_motion.planner import orient_candidates, plan_chain
from g2cell_motion.robot import CHECK_LINKS, GRIP_CLOSED, GRIP_OPEN, GRIPPERS, HEAD_CLEARANCE  # noqa: F401
from g2cell_motion.streamer import CommandBuffer, run_segment

ARMS = ("left", "right")
# Home = the arm part of scene_g2cell.yaml init_joint_pos (deg): joint2 -45, joint4 -75, joint6 -55.
HOME_DEG = {2: -45.0, 4: -75.0, 6: -55.0}
BAG_TOP = 0.87 + 0.07          # table top + bag height
CRATE_RIM = 0.82 + 0.224       # pack crate rim
# candidate pick points per arm, first reachable wins (cell frame; layout in assets/cell/build.py)
CRATE_FLOOR_BAG_TOP = 0.82 + 0.014 + 0.07   # a bag lying on the crate's base slab
# (name, (x, y), surface top z, approach/lift height above it)
# The right arm cannot reach the pick table from the base pose (IK: no solution for Y >= 0.40 at
# table height), so it "arranges" the bag the left arm dropped: grips it inside the pack crate and
# re-places it on the crate's other side.
PICKS = {
    "left": [("Bag_02", (-0.16, 0.60), BAG_TOP, 0.13), ("Bag_01", (0.10, 0.56), BAG_TOP, 0.13)],
    "right": [("bag in pack crate", (0.58, -0.08), CRATE_FLOOR_BAG_TOP, 0.26),
              ("Bag_01 near edge", (0.10, 0.44), BAG_TOP, 0.13)],
}
PLACES = {"left": (0.60, 0.0), "right": (0.62, 0.09)}
# Head clearance (HEAD_CLEARANCE = 0.20 m), the checked links and the orientation search order
# now live in g2cell_motion (robot.py, planner.py) and are shared with motion_server.


class PickPlaceDemo(Node, IkHelper):
    def __init__(self):
        super().__init__("g2cell_pick_place_demo")
        p = self.declare_parameter
        self.rate = p("rate_hz", 50.0).value
        self.max_vel = p("max_joint_vel", 1.1).value
        self.repeat = p("repeat", 1).value
        self.init_ik()
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, "/joint_command", qos)
        # "start" when motion begins, "done" at the end (record_cams can start/stop on it)
        self.state_pub = self.create_publisher(String, "/g2cell_demo/state", 10)
        self.init_fk()
        self.buf = CommandBuffer()
        self.create_timer(1.0 / self.rate, self._publish)

    # ---- output ------------------------------------------------------------------------
    def _publish(self):
        if self.buf.empty():
            return
        items = self.buf.snapshot()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [k for k, _ in items]
        msg.position = [v for _, v in items]
        self.pub.publish(msg)

    def _set(self, values: dict):
        self.buf.set(values)

    def move_joints(self, target: dict, min_t: float = 0.6):
        names = list(target)
        t = duration_for(self.buf.get(names), [target[n] for n in names], self.max_vel, min_t)
        run_segment(self.buf, target, t, self.rate)

    def gripper(self, arm: str, value: float, t: float = 0.8):
        self.move_joints({GRIPPERS[arm]: value}, min_t=t)

    def pause(self, t: float):
        time.sleep(t)

    # ---- planning (g2cell_motion.planner / clearance) -------------------------------------
    def path_clearance(self, arm: str, chain: list, samples: int = 8) -> float:
        """Min distance from the head (link3 origin + camera) to the arm's links along the
        min-jerk joint path through `chain` (list of joint dicts)."""
        heads = self.head_points()
        if heads is None:
            return 0.0
        return path_clearance(chain, lambda j: self.fk_points(j, CHECK_LINKS[arm]), heads, samples)

    def plan(self, arm: str, tools: list, seed: dict, prefer=None, start=None, end=None):
        """IK for a chain of fingertip points with one shared orientation. Returns list of joint
        dicts or None. Tries `prefer` (yaw, tilt) first, then the shared orientation order, and
        rejects chains whose joint path start -> sols -> end passes within HEAD_CLEARANCE of the head."""
        res = plan_chain(tools, lambda pos, q, s: self.solve(arm, pos, q, seed=s, attempts=6),
                         lambda path: self.path_clearance(arm, path), seed=seed,
                         joint_names=self.arm_joint_names(arm), orients=orient_candidates(prefer=prefer),
                         start=start, end=end, min_clearance=HEAD_CLEARANCE,
                         log=lambda m: self.get_logger().info(f"[{arm}] {m}"))
        self.last_orient = res.label if res.ok else None
        if res.ok:
            self.get_logger().info(f"[{arm}] plan ok: yaw/tilt {res.label}, head clearance {res.clearance:.2f} m")
            return res.solutions
        return None

    def plan_arm(self, arm: str, home: dict):
        px, py = PLACES[arm]
        for name, (x, y), top, clear in PICKS[arm]:
            picks = [(x, y, top + clear), (x, y, top - 0.01), (x, y, top + clear + 0.04)]
            pick_sols = self.plan(arm, picks, home, start=home)
            if not pick_sols:
                continue
            # Via point halfway, no higher than the lift: with arm colliders disabled a free
            # joint-space swing from table to crate can sweep the forearm through the head.
            lz = picks[-1][2]
            place = [((x + px) / 2, (y + py) / 2, max(lz, CRATE_RIM + 0.12)),
                     (px, py, CRATE_RIM + 0.17), (px, py, CRATE_RIM + 0.03), (px, py, CRATE_RIM + 0.20)]
            # same orientation as the pick if possible, seeded from the lift pose (same IK branch)
            place_sols = self.plan(arm, place, {**home, **pick_sols[-1]}, prefer=self.last_orient,
                                   start=pick_sols[-1], end=home)
            if pick_sols and place_sols:
                break
            self.get_logger().warn(f"[{arm}] {name} at ({x}, {y}) -> pack crate unreachable, trying next")
        else:
            self.get_logger().error(f"[{arm}] no reachable pick target")
            return None
        self.get_logger().info(f"[{arm}] will pick {name}")
        return pick_sols + place_sols

    def execute(self, arm: str, home: dict, sols):
        pre, grasp, lift, via, above, drop, retract = sols
        threading.Thread(target=self.gripper, args=(arm, GRIP_OPEN, 1.0), daemon=True).start()
        self.move_joints(pre, min_t=2.0)
        self.move_joints(grasp, min_t=1.4)
        self.gripper(arm, GRIP_CLOSED)
        self.pause(0.3)
        self.move_joints(lift, min_t=1.0)
        self.move_joints(via, min_t=1.2)
        self.move_joints(above, min_t=1.2)
        self.move_joints(drop, min_t=1.2)
        self.gripper(arm, GRIP_OPEN)
        self.pause(0.3)
        self.move_joints(retract, min_t=1.0)
        threading.Thread(target=self.gripper, args=(arm, GRIP_CLOSED, 1.0), daemon=True).start()
        self.move_joints(home, min_t=2.0)
        return True

    def run(self) -> bool:
        if not (self.ik.wait_for_service(timeout_sec=60.0) and self.fk.wait_for_service(timeout_sec=10.0)):
            self.get_logger().error("/compute_ik not available: is move_group running?")
            return False
        while rclpy.ok() and self.joint_state() is None:
            time.sleep(0.1)
        meas = self.measured()
        homes = {}
        for arm in ARMS:
            names = self.arm_joint_names(arm)
            self._set({n: meas[n] for n in names})
            self._set({GRIPPERS[arm]: meas.get(GRIPPERS[arm], GRIP_CLOSED)})
            # idx2k_arm_l_jointk / idx6k_arm_r_jointk -> joint number = last char
            homes[arm] = {n: math.radians(HOME_DEG.get(int(n[-1]), 0.0)) for n in names}
        self.get_logger().info("moving both arms to home (scene init pose)")
        self.move_joints({**homes["left"], **homes["right"]}, min_t=3.0)
        self.move_joints({GRIPPERS["left"]: GRIP_CLOSED, GRIPPERS["right"]: GRIP_CLOSED}, min_t=0.5)
        self.get_logger().info("planning (IK for all waypoints) ...")
        plans = {}
        for arm in ARMS:  # IK (KDL + our seed jitter) is randomised: retry a few times
            for attempt in range(self.declare_parameter(f"plan_retries_{arm}", 4).value):
                plans[arm] = self.plan_arm(arm, homes[arm])
                if plans[arm] is not None:
                    break
        start = self.declare_parameter("start_delay", 1.0).value
        self.get_logger().info(f"plans: { {a: p is not None for a, p in plans.items()} }; moving in {start} s")
        self.state_pub.publish(String(data="start"))
        self.pause(start)
        ok = all(p is not None for p in plans.values())
        for _ in range(self.repeat):
            for arm in ARMS:
                if plans[arm] is not None:
                    self.execute(arm, homes[arm], plans[arm])
        self.pause(1.0)
        self.state_pub.publish(String(data="done"))
        self.get_logger().info("demo done")
        return ok


def main():
    rclpy.init()
    node = PickPlaceDemo()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
