"""Per-pick outcome classification (spec "per-pick outcomes to log").

Precedence: drop > wrong_target > overflow > regrasp > success. Without a placement
observation the result is "unverified" -- never a claimed success.
"""
from __future__ import annotations

from dataclasses import dataclass

# must match g2cell_interfaces/msg/PickPlaceResult OUTCOME_* constants
OUTCOME_CODES = {"success": 0, "wrong_target": 1, "drop": 2, "regrasp": 3, "overflow": 4,
                 "unverified": 10, "plan_failed": 11, "grasp_failed": 12, "unreachable": 13,
                 "aborted": 14, "invalid": 15}
OFF_TARGET = ("floor", "conveyor")


@dataclass(frozen=True)
class Placement:
    """Where the bag came to rest, as observed by a scorer (Stage 5)."""
    target: str                 # crate | rack_lane | floor | conveyor | table
    lane: str = ""
    proud_of_rim: bool = False


def classify(intended_target: str, intended_lane: str, placement: Placement | None,
             grasp_attempts: int, dropped: bool = False):
    """-> (outcome name, verified)."""
    if dropped:
        return "drop", True
    if placement is None:
        return "unverified", False
    if placement.target in OFF_TARGET:
        return "drop", True
    if placement.target != intended_target or (intended_target == "rack_lane" and placement.lane != intended_lane):
        return "wrong_target", True
    if placement.proud_of_rim:
        return "overflow", True
    if grasp_attempts > 1:
        return "regrasp", True
    return "success", True
