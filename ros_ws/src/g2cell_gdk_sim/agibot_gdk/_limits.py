"""G2 joint names and limits (rad).

REAL_SOFT_LIMITS: the robot's soft limits, gdk/config/t2_soft_limit.json as listed in the GDK v2.3.4
reference (agibot-gdk-skil); the same values as gdk_training/joints.py (GDK 2.6.3).
SIM_URDF_LIMITS: Genie Sim's G2_crs_omnipicker URDF (g2_viz_ws/g2_crs_omnipicker.urdf).

A target outside the real limits is rejected, as the robot rejects it. A target inside the real
limits but outside the narrower sim limits (head, waist joint 3) is clamped into the sim limits.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

WAIST = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
HEAD = [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
LEFT_ARM = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
ALL = WAIST + HEAD + LEFT_ARM + RIGHT_ARM
ARMS = {"left": LEFT_ARM, "right": RIGHT_ARM}
# OmniPicker leader joints: 0 rad = closed, -0.785 rad = open (sim and GDK use the same values)
GRIPPERS = {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}
GRIPPER_OPEN, GRIPPER_CLOSED = -0.7853981633974483, 0.0

_ARM_REAL = [(-3.072, 3.072), (-2.060, 2.060), (-3.072, 3.072), (-2.496, 1.012), (-3.072, 3.072),
             (-1.012, 1.012), (-1.536, 1.536)]
REAL_SOFT_LIMITS: dict[str, tuple[float, float]] = {
    "idx01_body_joint1": (-1.082, 0.000),
    "idx02_body_joint2": (0.000, 2.653),
    "idx03_body_joint3": (-1.920, 1.571),
    "idx04_body_joint4": (-0.436, 0.436),
    "idx05_body_joint5": (-3.046, 3.046),
    "idx11_head_joint1": (-1.571, 1.571),
    "idx12_head_joint2": (-0.349, 0.349),
    "idx13_head_joint3": (-0.535, 0.535),
    **dict(zip(LEFT_ARM, _ARM_REAL)),
    **dict(zip(RIGHT_ARM, _ARM_REAL)),
}

_ARM_SIM = [(-3.0718, 3.0718), (-2.0595, 2.0595), (-3.0718, 3.0718), (-2.4959, 1.0123), (-3.0718, 3.0718),
            (-1.0123, 1.0123), (-1.5359, 1.5359)]
SIM_URDF_LIMITS: dict[str, tuple[float, float]] = {
    "idx01_body_joint1": (-1.0822, 0.0001),
    "idx02_body_joint2": (-0.0001, 2.653),
    "idx03_body_joint3": (-1.907, 1.039),
    "idx04_body_joint4": (-0.4364, 0.4364),
    "idx05_body_joint5": (-3.0456, 3.0456),
    "idx11_head_joint1": (-0.781, 0.781),
    "idx12_head_joint2": (-0.151, 0.151),
    "idx13_head_joint3": (-0.0181, 0.5231),
    **dict(zip(LEFT_ARM, _ARM_SIM)),
    **dict(zip(RIGHT_ARM, _ARM_SIM)),
}
# URDF <limit velocity>, rad/s: the cap for the speeds of a blocking move
SIM_VELOCITY_LIMITS = {**dict.fromkeys(WAIST, 1.21), **dict.fromkeys(HEAD, 2.5),
                       **{n: v for arm in (LEFT_ARM, RIGHT_ARM) for n, v in zip(arm, (3.9, 3.9, 3.6, 3.6, 4.7, 4.7, 4.7))}}

TOLERANCE = 0.01  # rad: a target this close outside the real limit is clamped, not rejected (sim drift, saved poses)


class LimitError(ValueError):
    """The robot would reject the request (GDKRes.kInvalidInput)."""


def effective_limits(name: str) -> tuple[float, float]:
    (rl, rh), (sl, sh) = REAL_SOFT_LIMITS[name], SIM_URDF_LIMITS[name]
    return max(rl, sl), min(rh, sh)


def _excess(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


def check_and_clamp(names, positions, current: dict | None = None) -> tuple[list[float], list[str]]:
    """Validates targets against the real limits and clamps them into the sim limits.

    Returns the clamped targets and one note per clamped joint. Raises LimitError for an
    unknown joint or a target outside the real soft limits. `current` (measured positions)
    allows a target as far outside as the joint already is: the sim's physics lets a joint rest
    a few mrad past its limit, and commanding a saved measured pose must still work."""
    current = current or {}
    out, notes = [], []
    for name, value in zip(names, positions):
        if name not in REAL_SOFT_LIMITS:
            raise LimitError(f"unknown joint {name}")
        value = float(value)
        low, high = REAL_SOFT_LIMITS[name]
        allowed = TOLERANCE + (_excess(float(current[name]), low, high) if name in current else 0.0)
        if _excess(value, low, high) > allowed:
            raise LimitError(f"{name} {value:.3f} outside {low} .. {high}")
        lo, hi = effective_limits(name)
        clamped = min(max(value, lo), hi)
        if abs(clamped - value) > TOLERANCE:
            notes.append(f"{name} {value:.3f} clamped to {clamped:.3f} (real soft limits ∩ sim URDF limits: {lo} .. {hi})")
        out.append(clamped)
    return out, notes


def clamp_gripper(value: float) -> float:
    return min(max(float(value), GRIPPER_OPEN), GRIPPER_CLOSED)


def parse_urdf_limits(text: str) -> dict[str, tuple[float, float]]:
    out = {}
    for joint in ET.fromstring(text).iter("joint"):
        limit = joint.find("limit")
        if joint.get("type") in ("revolute", "prismatic") and limit is not None:
            out[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return out
