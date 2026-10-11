"""Pure-python pose/trajectory helpers (no ROS imports, unit-tested).

Moved here from g2cell_demo.geometry (which now re-exports this module).
"""
from __future__ import annotations

import math

# Fingertip centre relative to arm_*_end_link, along the end link's +Z (FK: gripper inner/outer
# link4 sit ~0.11-0.12 m out; +0.01 for the pads).
TOOL_OFFSET = 0.125
# base_link sits init_base_pose.z above the cell floor (scene_g2cell.yaml)
BASE_Z = 0.04


def qmul(a, b):
    """Hamilton product, (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, _ = qmul(qmul(q, (v[0], v[1], v[2], 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return (x, y, z)


def axis_angle(axis, angle):
    s = math.sin(angle / 2)
    n = math.sqrt(sum(c * c for c in axis))
    return (axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, math.cos(angle / 2))


def down_quat(yaw: float = 0.0, tilt: float = 0.0):
    """End-link orientation with its +Z (the gripper) pointing down, rotated `yaw` about world Z.
    `tilt` pitches the gripper about the rotated world Y (positive tips the fingers toward +X)."""
    q = axis_angle((1, 0, 0), math.pi)                 # +Z -> -Z
    if tilt:
        q = qmul(axis_angle((0, 1, 0), -tilt), q)
    return qmul(axis_angle((0, 0, 1), yaw), q)


def quat_is_unset(q) -> bool:
    """An all-zero quaternion (the ROS default) means "no orientation given"."""
    return all(abs(c) < 1e-12 for c in q)


def end_link_for_tool(tool_world, q, tool_offset: float = TOOL_OFFSET, base_z: float = BASE_Z):
    """End-link position in base_link given the desired fingertip point in the cell (world) frame."""
    d = qrot(q, (0.0, 0.0, tool_offset))
    return (tool_world[0] - d[0], tool_world[1] - d[1], tool_world[2] - d[2] - base_z)


def min_jerk(s: float) -> float:
    """Quintic time scaling 10s^3 - 15s^4 + 6s^5: zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


quintic = min_jerk


def interpolate(a, b, n_steps: int):
    """Smooth (minimum-jerk) joint-space samples from a to b, excluding a, including b."""
    out = []
    for i in range(1, n_steps + 1):
        s = min_jerk(i / n_steps)
        out.append([x + (y - x) * s for x, y in zip(a, b)])
    return out


def duration_for(a, b, max_vel: float = 0.6, min_t: float = 0.6) -> float:
    """Segment time so the min-jerk peak joint speed (1.875 * dq / T) stays under max_vel rad/s."""
    dq = max((abs(y - x) for x, y in zip(a, b)), default=0.0)
    return max(min_t, 1.875 * dq / max_vel)


def segment_durations(chain, total: float, max_vel: float, min_t: float) -> list[float]:
    """Durations for consecutive segments of `chain` (lists of joint values).

    total <= 0: each segment gets its speed-limited duration. total > 0: the speed-limited
    durations are scaled up proportionally to sum to `total`; a total shorter than the speed
    limit allows is ignored (never faster than max_vel)."""
    auto = [duration_for(a, b, max_vel, min_t) for a, b in zip(chain, chain[1:])]
    s = sum(auto)
    if total <= 0 or total <= s or s <= 0:
        return auto
    return [d * total / s for d in auto]
