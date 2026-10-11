"""Input devices behind one interface (ROS-free, unit-tested).

Every device's poll() returns a DeviceInput for one control tick:
  twist          desired 6D velocity in the robot base frame (m/s, rad/s), continuous devices
  gripper_delta  continuous gripper velocity in rad/s (positive = joint value up)
  events         discrete events: keyboard keys (handled by teleop_core.handle_key, the
                 existing key map in discrete steps) or button events
                 ("toggle_gripper", "switch_arm")

The ROS node feeds SpaceNavTopicDevice from /spacenav/twist and /spacenav/joy
(ros-jazzy-spacenav); PySpaceMouseDevice reads the USB device directly through pyspacemouse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from g2cell_teleop.motion import ZERO, Twist, TwistFilter, apply_twist
from g2cell_teleop.teleop_core import TeleopState, handle_key, switch_arm, toggle_gripper

BUTTON_EVENTS: dict[str, Callable[[TeleopState], str | None]] = {
    "toggle_gripper": toggle_gripper,
    "switch_arm": switch_arm,
}


@dataclass(frozen=True)
class DeviceInput:
    twist: Twist = ZERO
    gripper_delta: float = 0.0
    events: tuple[str, ...] = ()


def apply_input(state: TeleopState, inp: DeviceInput, flt: TwistFilter, dt: float) -> list[str]:
    """Apply one tick of device input to the state. Returns the actions, in order."""
    actions = []
    for ev in inp.events:
        act = BUTTON_EVENTS[ev](state) if ev in BUTTON_EVENTS else handle_key(state, ev)
        if act is not None:
            actions.append(act)
    if state.paused:
        flt.reset()  # resume from standstill, not from the pre-pause velocity
        return actions
    act = apply_twist(state, flt.update(inp.twist, dt), dt)
    if act is not None:
        actions.append(act)
    if inp.gripper_delta and dt > 0:
        lo, hi = sorted(state.gripper_limits)
        g = state.gripper[state.active] + inp.gripper_delta * dt
        state.gripper[state.active] = min(hi, max(lo, g))
        actions.append("gripper")
    return actions


class KeyboardDevice:
    """Discrete steps from the existing key map. read_key() returns a key or None (no wait)."""

    def __init__(self, read_key: Callable[[], str | None]):
        self.read_key = read_key

    def poll(self) -> DeviceInput:
        keys = []
        while (k := self.read_key()) is not None:
            keys.append(k)
        return DeviceInput(events=tuple(keys))


# ---- SpaceMouse --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceMouseConfig:
    lin_scale: float = 0.1  # m/s at full deflection
    ang_scale: float = 0.5  # rad/s at full deflection
    deadband: float = 0.1  # fraction of full deflection, per axis
    # output axis i (vx, vy, vz, wx, wy, wz in the base frame) = raw[axis_map[i]] * axis_sign[i]
    # raw order: x, y, z, roll, pitch, yaw as the source reports them
    axis_map: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    axis_sign: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    gripper_button: int = 0
    switch_button: int = 1
    stale_after: float = 0.5  # s without a twist message -> zero (topic source only)

    def __post_init__(self):
        if sorted(self.axis_map) != [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"axis_map must be a permutation of 0..5, got {self.axis_map}")
        if len(self.axis_sign) != 6 or any(s not in (1, -1) for s in self.axis_sign):
            raise ValueError(f"axis_sign must be six values of 1 or -1, got {self.axis_sign}")


def _shape(v: float, band: float) -> float:
    v = max(-1.0, min(1.0, v))
    if abs(v) <= band:
        return 0.0
    return (abs(v) - band) / (1.0 - band) * (1 if v > 0 else -1)


def map_axes(raw, cfg: SpaceMouseConfig) -> Twist:
    """Six normalised axes (about -1..1) -> base-frame twist, with deadband and scale."""
    out = [_shape(raw[src] * sign, cfg.deadband) for src, sign in zip(cfg.axis_map, cfg.axis_sign)]
    return Twist(tuple(c * cfg.lin_scale for c in out[:3]), tuple(c * cfg.ang_scale for c in out[3:]))


def rising_edges(prev, cur) -> list[int]:
    return [i for i, b in enumerate(cur) if b and not (i < len(prev) and prev[i])]


class _SpaceMouseBase:
    def __init__(self, cfg: SpaceMouseConfig):
        self.cfg = cfg
        self._prev_buttons: list[int] = []

    def _input(self, raw, buttons) -> DeviceInput:
        events = []
        for i in rising_edges(self._prev_buttons, buttons):
            if i == self.cfg.gripper_button:
                events.append("toggle_gripper")
            elif i == self.cfg.switch_button:
                events.append("switch_arm")
        self._prev_buttons = list(buttons)
        twist = ZERO if raw is None else map_axes(raw, self.cfg)
        return DeviceInput(twist=twist, events=tuple(events))


class SpaceNavTopicDevice(_SpaceMouseBase):
    """Fed by the node's /spacenav/twist and /spacenav/joy callbacks (thread-safe enough:
    each callback swaps a whole tuple/list)."""

    def __init__(self, cfg: SpaceMouseConfig, clock: Callable[[], float] = time.monotonic):
        super().__init__(cfg)
        self.clock = clock
        self._raw = None
        self._stamp = -float("inf")
        self._buttons: list[int] = []

    def on_twist(self, lin, ang) -> None:
        self._raw = tuple(lin) + tuple(ang)
        self._stamp = self.clock()

    def on_joy(self, buttons) -> None:
        self._buttons = list(buttons)

    def poll(self) -> DeviceInput:
        fresh = self.clock() - self._stamp <= self.cfg.stale_after
        return self._input(self._raw if fresh else None, self._buttons)


class PySpaceMouseDevice(_SpaceMouseBase):
    """Direct USB/HID read through pyspacemouse (no spacenavd needed)."""

    def __init__(self, cfg: SpaceMouseConfig, module=None):
        super().__init__(cfg)
        if module is None:
            import pyspacemouse as module  # noqa: PLC0415 - optional dependency
        self.mod = module
        if not self.mod.open():
            raise RuntimeError("pyspacemouse could not open a SpaceMouse (check USB / hidraw permissions)")

    def poll(self) -> DeviceInput:
        s = self.mod.read()
        raw = (s.x, s.y, s.z, s.roll, s.pitch, s.yaw)
        return self._input(raw, list(s.buttons))
