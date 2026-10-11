"""Teleop waist helpers (ROS-free): auto-face via g2cell_motion.waist_policy.

Keys ',' / '.' step the waist yaw (teleop_core); with auto-face on ('t'), every arm move first turns
the waist so the active arm's end-effector target lies within +-auto_face_cone_deg of the torso's
forward direction (smallest turn, clamped to waist_min_deg..waist_max_deg).
"""
from __future__ import annotations

import math

from g2cell_motion.waist_policy import WaistConfig, WaistDecision, choose_waist_yaw

from g2cell_teleop.teleop_core import TeleopState


def waist_config_from(p: dict) -> WaistConfig:
    lo, hi = math.radians(p["waist_min_deg"]), math.radians(p["waist_max_deg"])
    return WaistConfig(cone_deg=p["auto_face_cone_deg"], max_turn_deg=None, lower=lo, upper=hi,
                       pivot=tuple(p["auto_face_pivot"]), joint=p["waist_joint"])


def auto_face(state: TeleopState, cfg: WaistConfig) -> WaistDecision:
    """Waist decision for the active arm's target from the current waist yaw (state is not changed)."""
    return choose_waist_yaw(state.targets[state.active].position, state.waist, cfg)
