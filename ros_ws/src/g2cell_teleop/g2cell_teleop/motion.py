"""Continuous twist -> end-effector target integration (ROS-free, unit-tested).

A device gives a desired 6D twist in the robot base frame each tick. TwistFilter applies a
deadband, caps the speed (0.1 m/s, 0.5 rad/s by default) and limits how fast the twist may
change (acceleration). apply_twist integrates the result into the active arm's target and
clamps the position to that arm's workspace box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from g2cell_teleop.teleop_core import Pose, Quat, TeleopState, Vec3, _quat_mul, set_position


@dataclass(frozen=True)
class Twist:
    lin: Vec3  # m/s, base frame
    ang: Vec3  # rad/s about the base axes


ZERO = Twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class MotionLimits:
    max_lin: float = 0.1  # m/s
    max_ang: float = 0.5  # rad/s
    max_lin_acc: float = 0.5  # m/s^2
    max_ang_acc: float = 2.5  # rad/s^2
    lin_deadband: float = 0.0  # m/s, on the vector norm
    ang_deadband: float = 0.0  # rad/s


def _norm(v) -> float:
    return math.sqrt(sum(c * c for c in v))


def deadband(v: Vec3, band: float) -> Vec3:
    return (0.0, 0.0, 0.0) if _norm(v) <= band else tuple(v)


def limit_norm(v: Vec3, limit: float) -> Vec3:
    n = _norm(v)
    return tuple(v) if n <= limit else tuple(c * limit / n for c in v)


def _rate_limit(prev: Vec3, want: Vec3, max_delta: float) -> Vec3:
    delta = limit_norm(tuple(w - p for w, p in zip(want, prev)), max_delta)
    return tuple(p + d for p, d in zip(prev, delta))


class TwistFilter:
    """Deadband -> speed cap -> acceleration limit. Keeps the last output twist."""

    def __init__(self, limits: MotionLimits):
        self.limits = limits
        self.last = ZERO

    def reset(self) -> None:
        self.last = ZERO

    def update(self, raw: Twist, dt: float) -> Twist:
        lim = self.limits
        lin = limit_norm(deadband(raw.lin, lim.lin_deadband), lim.max_lin)
        ang = limit_norm(deadband(raw.ang, lim.ang_deadband), lim.max_ang)
        dt = max(dt, 0.0)
        self.last = Twist(_rate_limit(self.last.lin, lin, lim.max_lin_acc * dt),
                          _rate_limit(self.last.ang, ang, lim.max_ang_acc * dt))
        return self.last


def _quat_from_rotvec(r: Vec3) -> Quat:
    angle = _norm(r)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle / 2) / angle
    return (r[0] * s, r[1] * s, r[2] * s, math.cos(angle / 2))


def integrate(pose: Pose, twist: Twist, dt: float) -> Pose:
    pos = tuple(p + v * dt for p, v in zip(pose.position, twist.lin))
    q = _quat_mul(_quat_from_rotvec(tuple(w * dt for w in twist.ang)), pose.orientation)
    n = _norm(q)
    return Pose(pos, tuple(c / n for c in q))


def apply_twist(state: TeleopState, twist: Twist, dt: float) -> str | None:
    """Move the active arm's target by an already-filtered twist. 'arm' if it moved."""
    if state.paused or dt <= 0 or twist == ZERO:
        return None
    arm = state.active
    moved = integrate(state.targets[arm], twist, dt)
    state.targets[arm] = Pose(state.targets[arm].position, moved.orientation)
    set_position(state, arm, moved.position)
    return "arm"
