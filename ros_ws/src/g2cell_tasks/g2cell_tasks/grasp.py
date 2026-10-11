"""Grasp detection from the gripper leader joint (omnipicker: 0 = closed, -0.785 = open).

After a close command the fingers stop short of 0 when something is between them. Unvalidated
on the rigid placeholder bags in Genie Sim, so pick_bag_server only uses it with
grasp_check:=gripper.
"""
from __future__ import annotations

GRIP_OPEN, GRIP_CLOSED = -0.785, 0.0


def grasp_from_gripper(q, closed: float = GRIP_CLOSED, open_: float = GRIP_OPEN, margin: float = 0.05):
    """True: fingers stopped between open and closed (object held); False: closed on nothing or
    still open; None: no measurement."""
    if q is None:
        return None
    return open_ + margin < q < closed - margin
