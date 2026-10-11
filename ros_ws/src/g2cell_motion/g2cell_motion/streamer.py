"""Joint command buffer + quintic segment streaming (no ROS imports; clock is injected).

The motion server's 50 Hz timer publishes CommandBuffer.snapshot() on /joint_command; motion
code only changes the buffer, sample by sample, with run_segment.
"""
from __future__ import annotations

import threading
import time

from g2cell_motion.geometry import interpolate


class CommandBuffer:
    """Thread-safe {joint: commanded position}."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cmd: dict[str, float] = {}

    def set(self, values: dict) -> None:
        with self._lock:
            self._cmd.update(values)

    def get(self, names) -> list[float]:
        with self._lock:
            return [self._cmd[n] for n in names]

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._cmd

    def snapshot(self) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._cmd.items())

    def empty(self) -> bool:
        with self._lock:
            return not self._cmd


def run_segment(buf: CommandBuffer, target: dict, duration: float, rate: float = 50.0,
                now=time.monotonic, sleep=time.sleep, cancelled=lambda: False, on_step=None) -> bool:
    """Move the buffered joints to `target` along a quintic profile lasting `duration` s,
    one sample per 1/rate. Joints not yet in the buffer start at their target. Returns False if
    `cancelled()` became true (the buffer then holds the last sample, i.e. the arm stops there)."""
    names = list(target)
    a = [buf.get([n])[0] if buf.has(n) else target[n] for n in names]
    b = [target[n] for n in names]
    steps = max(1, int(round(duration * rate)))
    t0 = now()
    for i, q in enumerate(interpolate(a, b, steps), start=1):
        if cancelled():
            return False
        buf.set(dict(zip(names, q)))
        if on_step:
            on_step(i, steps)
        delay = t0 + i / rate - now()
        if delay > 0:
            sleep(delay)
    return True
