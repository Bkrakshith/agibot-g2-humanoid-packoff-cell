"""Blocking MoveArm action client for task code (g2cell_tasks). The node must be spun by a
MultiThreadedExecutor in another thread while these calls block.

    mc = MotionClient(node)
    r = mc.move_tool("left", (0.10, 0.56, 1.07))          # orientation searched
    mc.gripper("left", close=True)
    mc.move_named("left", "home")
"""
from __future__ import annotations

import threading
import time

from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from g2cell_interfaces.action import MoveArm


def _pose(xyz, quat=None) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in xyz)
    if quat is not None:
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
    return p


class MotionClient:
    def __init__(self, node, action_name: str = "/g2cell/move_arm", timeout: float = 60.0):
        self.node = node
        self.timeout = timeout
        self.client = ActionClient(node, MoveArm, action_name, callback_group=ReentrantCallbackGroup())
        self._active = None

    def wait(self, timeout: float = 10.0) -> bool:
        return self.client.wait_for_server(timeout_sec=timeout)

    def send(self, goal: MoveArm.Goal, feedback=None, cancelled=lambda: False):
        """Send and block. Returns MoveArm.Result, or None if rejected / timed out. When
        cancelled() turns true the goal is canceled and its (CANCELED) result returned."""
        ev = threading.Event()
        box = {}

        def on_result(fut):
            box["result"] = fut.result().result
            ev.set()

        def on_goal(fut):
            gh = fut.result()
            if not gh.accepted:
                ev.set()
                return
            self._active = gh
            gh.get_result_async().add_done_callback(on_result)

        self.client.send_goal_async(goal, feedback_callback=feedback).add_done_callback(on_goal)
        deadline, sent_cancel = time.monotonic() + self.timeout, False
        while not ev.wait(0.1):
            if not sent_cancel and cancelled():
                self.cancel()
                sent_cancel = True
            if time.monotonic() > deadline:
                self.cancel()
                return None
        self._active = None
        return box.get("result")

    def cancel(self):
        if self._active is not None:
            self._active.cancel_goal_async()

    def move_tool(self, arm, xyz, quat=None, via=(), duration=0.0, gripper=MoveArm.Goal.GRIPPER_KEEP,
                  cancelled=lambda: False, waist_mode: str = "", waist_yaw: float = 0.0, lookahead=()):
        """waist_mode: "" (server default) | "waist_first" | "fixed" | "absolute" (waist_yaw, rad).
        lookahead: points the next moves go to (planned with the same orientation, not executed)."""
        g = MoveArm.Goal(arm=arm, frame=MoveArm.Goal.FRAME_CELL_TOOL, pose=_pose(xyz, quat),
                         auto_orientation=quat is None,
                         via=[_pose(v, quat) for v in via], lookahead=[_pose(v, quat) for v in lookahead],
                         duration=float(duration), gripper=gripper,
                         waist_mode=waist_mode, waist_yaw=float(waist_yaw))
        return self.send(g, cancelled=cancelled)

    def move_named(self, arm, name, duration=0.0, cancelled=lambda: False, waist_mode: str = ""):
        return self.send(MoveArm.Goal(arm=arm, named_pose=name, duration=float(duration), waist_mode=waist_mode),
                         cancelled=cancelled)

    def gripper(self, arm, close: bool, cancelled=lambda: False):
        cmd = MoveArm.Goal.GRIPPER_CLOSE if close else MoveArm.Goal.GRIPPER_OPEN
        return self.send(MoveArm.Goal(arm=arm, gripper_only=True, gripper=cmd), cancelled=cancelled)


def ok(result) -> bool:
    return result is not None and result.error_code == MoveArm.Result.OK


def waist_of(result):
    """(final waist yaw rad, base_turn_needed, base_turn rad) of a MoveArm result, or None."""
    if result is None:
        return None
    return (float(getattr(result, "waist_yaw", 0.0)), bool(getattr(result, "base_turn_needed", False)),
            float(getattr(result, "base_turn", 0.0)))


def quat_of(result):
    """orientation_used of a MoveArm result as (x, y, z, w), or None if unset."""
    if result is None:
        return None
    o = result.orientation_used
    q = (o.x, o.y, o.z, o.w)
    return None if all(abs(c) < 1e-12 for c in q) else q
