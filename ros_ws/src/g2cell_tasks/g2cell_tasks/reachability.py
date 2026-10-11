"""Arm selection from the reachability table, and the waist plan per target.

Stage 2 (MoveIt /compute_ik, assets/cell/README.md) and the Layout v2 offline waist sweep
(tools/reach_envelope.py --waist-yaw, output/reach/reach_summary_waist.json):
- Pick table (left, +Y): only the LEFT arm, up to about Y <= 0.85 (cell frame) at table height. Turning
  the waist does not extend this (the left shoulder swings back as the torso turns left: 32% -> 35%
  of the spawn zone); the right arm cannot pick from the table.
- Pack-position carton (0.60, 0): both arms, waist ~0.
- Sort rack (right, -Y): per lane, `rack_lane_arms` (preference order). Offline kinematics only ->
  flagged unverified until checked with MoveIt.
Waist (g2cell_motion waist_policy, MoveArm.waist_mode): table picks and rack drops use waist_first
(turn toward the target, arms kept near forward); the carton uses waist 0 (absolute).
"""
from __future__ import annotations

from dataclasses import dataclass, field

ARMS = ("left", "right")


@dataclass(frozen=True)
class ReachTable:
    left_table_max_y: float = 0.85
    right_reaches_table: bool = False
    crate_arms: tuple = ARMS
    rack_arms: tuple = ARMS
    rack_verified: bool = False
    waist_first: bool = False
    rack_lane_arms: dict = field(default_factory=dict)   # lane -> arms (preference order) with the waist

    @classmethod
    def from_dict(cls, d: dict) -> "ReachTable":
        d = dict(d or {})
        for k in ("crate_arms", "rack_arms"):
            if k in d:
                d[k] = tuple(d[k])
        if "rack_lane_arms" in d:
            d["rack_lane_arms"] = {k: tuple(v or ()) for k, v in (d["rack_lane_arms"] or {}).items()}
        return cls(**d)


@dataclass(frozen=True)
class ArmChoice:
    arm: str | None
    reason: str = ""
    verified: bool = True     # False: the target's reach was never measured


def waist_for(target: str, waist_first: bool = True) -> tuple[str, float]:
    """(MoveArm waist_mode, waist_yaw) for moves toward a target: 'table' | 'rack_lane' | 'crate'."""
    if not waist_first:
        return ("fixed", 0.0)
    if target == "crate":
        return ("absolute", 0.0)
    return ("waist_first", 0.0)


def _source_arms(t: ReachTable, source: str, xy):
    if source == "table":
        arms = []
        if xy[1] <= t.left_table_max_y:
            arms.append("left")
        if t.right_reaches_table:
            arms.append("right")
        why = f"table pick at Y {xy[1]:.2f}: left arm reaches Y <= {t.left_table_max_y:.2f}, " \
              f"right arm {'reaches' if t.right_reaches_table else 'cannot reach'} the table"
        if t.waist_first:
            why += " (waist turned toward the table)"
        return arms, why
    if source == "crate":
        arms = ["right", "left"] if xy[1] < 0 else ["left", "right"]
        return [a for a in arms if a in t.crate_arms], "crate pick"
    return [], f"unknown pick source {source!r}"


def select_arm(t: ReachTable, source: str, xy, target: str, requested: str = "", lane: str = "") -> ArmChoice:
    """source: 'table' | 'crate'; xy: bag position (cell frame); target: 'crate' | 'rack_lane'."""
    arms, why = _source_arms(t, source, xy)
    if target == "crate":
        allowed, verified = t.crate_arms, True
        arms = [a for a in arms if a in allowed]
    elif target == "rack_lane":
        verified = t.rack_verified
        if lane and lane in t.rack_lane_arms:
            allowed = t.rack_lane_arms[lane]
            arms = [a for a in allowed if a in arms]          # the lane's preference order
            why += f"; lane {lane} reached by {list(allowed) or 'no arm'}" + \
                   (" with the waist turned toward the rack" if t.waist_first else "")
        else:
            allowed = t.rack_arms
            arms = [a for a in arms if a in allowed]
    else:
        return ArmChoice(None, f"unknown target {target!r}")
    if requested:
        if requested not in arms:
            return ArmChoice(None, f"{requested} arm not usable here ({why})")
        return ArmChoice(requested, why, verified)
    if not arms:
        return ArmChoice(None, f"no arm reaches it ({why})")
    return ArmChoice(arms[0], why, verified)
