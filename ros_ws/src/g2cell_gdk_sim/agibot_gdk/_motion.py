"""Quintic joint interpolation and the thread-safe joint command buffer (no ROS)."""
from __future__ import annotations

import threading

from agibot_gdk._limits import SIM_VELOCITY_LIMITS

MIN_MOVE_TIME = 0.2  # s
PEAK = 1.875         # peak speed of the quintic profile = 1.875 * distance / duration


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def interpolate(a, b, steps: int) -> list[list[float]]:
    """Quintic samples from a to b, excluding a and including b."""
    out = []
    for i in range(1, steps + 1):
        s = min_jerk(i / steps)
        out.append([x + (y - x) * s for x, y in zip(a, b)])
    return out


def normalize_velocities(velocities, count: int, default: float) -> list[float]:
    """GDK requests carry one speed per joint, one speed for all, or none. Zero or less: default."""
    v = [float(x) for x in velocities]
    if not v:
        v = [default]
    if len(v) == 1:
        v = v * count
    if len(v) != count:
        raise ValueError(f"{len(v)} velocities for {count} joints")
    return [x if x > 0 else default for x in v]


def cap_velocities(names, velocities) -> list[float]:
    return [min(v, SIM_VELOCITY_LIMITS.get(n, v)) for n, v in zip(names, velocities)]


def move_duration(start, goal, velocities) -> float:
    """Duration of a synchronised quintic move in which no joint exceeds its speed."""
    t = max((PEAK * abs(b - a) / v for a, b, v in zip(start, goal, velocities)), default=0.0)
    return max(MIN_MOVE_TIME, t)


class CommandBuffer:
    """{joint: commanded position}, streamed at 50 Hz by the backend.

    Every command claims its joints; a later claim on any of them preempts an earlier blocking
    move, which notices through owns() and stops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._targets: dict[str, float] = {}
        self._owner: dict[str, int] = {}
        self._next = 0

    def claim(self, names) -> int:
        with self._lock:
            self._next += 1
            for n in names:
                self._owner[n] = self._next
            return self._next

    def owns(self, names, token: int) -> bool:
        with self._lock:
            return all(self._owner.get(n) == token for n in names)

    def set(self, targets: dict) -> None:
        with self._lock:
            self._targets.update({k: float(v) for k, v in targets.items()})

    def get(self, name: str) -> float | None:
        with self._lock:
            return self._targets.get(name)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._targets)
