import types

import pytest

from g2cell_teleop.devices import (
    DeviceInput,
    KeyboardDevice,
    PySpaceMouseDevice,
    SpaceMouseConfig,
    SpaceNavTopicDevice,
    apply_input,
    map_axes,
    rising_edges,
)
from g2cell_teleop.motion import ZERO, MotionLimits, Twist, TwistFilter
from g2cell_teleop.teleop_core import Pose, TeleopState, handle_key

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def make_state():
    return TeleopState(
        targets={"left": Pose((0.4, 0.2, 1.0), IDENTITY), "right": Pose((0.4, -0.2, 1.0), IDENTITY)},
        gripper={"left": 0.0, "right": 0.0},
        gripper_limits=(-0.785, 0.0),
    )


def fast_filter():
    return TwistFilter(MotionLimits(max_lin_acc=1e9, max_ang_acc=1e9))


# ---- keyboard --------------------------------------------------------------------------

def test_keyboard_yields_pending_keys_as_discrete_events_and_no_twist():
    keys = iter(["w", "r", None])
    dev = KeyboardDevice(lambda: next(keys))
    inp = dev.poll()
    assert inp.events == ("w", "r")
    assert inp.twist == ZERO and inp.gripper_delta == 0.0


def test_keyboard_input_applies_exactly_like_handle_key():
    a, b = make_state(), make_state()
    seq = list("wwqiu2dg]r")
    for k in seq:
        handle_key(a, k)
    keys = iter(seq + [None])
    actions = apply_input(b, KeyboardDevice(lambda: next(keys)).poll(), fast_filter(), 0.02)
    assert b.targets == a.targets and b.gripper == a.gripper and b.recording == a.recording
    assert "record_start" in actions


# ---- spacemouse mapping ----------------------------------------------------------------

def test_map_axes_scales_full_deflection_to_configured_speed():
    cfg = SpaceMouseConfig(lin_scale=0.1, ang_scale=0.5, deadband=0.0)
    t = map_axes((1.0, -1.0, 0.5, 0.0, 0.0, 1.0), cfg)
    assert t.lin == pytest.approx((0.1, -0.1, 0.05))
    assert t.ang == pytest.approx((0.0, 0.0, 0.5))


def test_map_axes_deadband_is_per_axis_and_rescaled():
    cfg = SpaceMouseConfig(lin_scale=1.0, ang_scale=1.0, deadband=0.2)
    t = map_axes((0.1, 0.6, 1.0, -0.15, 0.0, -1.0), cfg)
    assert t.lin == pytest.approx((0.0, 0.5, 1.0))
    assert t.ang == pytest.approx((0.0, 0.0, -1.0))


def test_map_axes_remaps_and_flips_axes():
    cfg = SpaceMouseConfig(lin_scale=1.0, ang_scale=1.0, deadband=0.0,
                           axis_map=(1, 0, 2, 3, 4, 5), axis_sign=(1, -1, 1, 1, 1, -1))
    t = map_axes((0.3, 0.7, 0.0, 0.0, 0.0, 0.2), cfg)
    assert t.lin == pytest.approx((0.7, -0.3, 0.0))
    assert t.ang == pytest.approx((0.0, 0.0, -0.2))


def test_map_axes_clips_overrange_input():
    cfg = SpaceMouseConfig(lin_scale=0.1, ang_scale=0.5, deadband=0.0)
    assert map_axes((3.0, 0, 0, 0, 0, 0), cfg).lin[0] == pytest.approx(0.1)


def test_config_rejects_bad_axis_map():
    with pytest.raises(ValueError):
        SpaceMouseConfig(axis_map=(0, 1, 2))
    with pytest.raises(ValueError):
        SpaceMouseConfig(axis_sign=(1, 1, 1, 1, 1, 2))


def test_rising_edges_only_on_press():
    assert rising_edges([0, 0], [1, 0]) == [0]
    assert rising_edges([1, 0], [1, 1]) == [1]
    assert rising_edges([1, 1], [0, 0]) == []
    assert rising_edges([], [1]) == [0]


# ---- spacenav topic device ---------------------------------------------------------------

def test_spacenav_topic_device_button0_toggles_gripper_button1_switches_arm():
    now = [0.0]
    dev = SpaceNavTopicDevice(SpaceMouseConfig(deadband=0.0), clock=lambda: now[0])
    dev.on_joy([1, 0])
    inp = dev.poll()
    assert inp.events == ("toggle_gripper",)
    assert dev.poll().events == ()  # still held: no repeat
    dev.on_joy([0, 1])
    assert dev.poll().events == ("switch_arm",)


def test_spacenav_topic_device_twist_and_staleness():
    now = [0.0]
    cfg = SpaceMouseConfig(lin_scale=0.1, ang_scale=0.5, deadband=0.0, stale_after=0.5)
    dev = SpaceNavTopicDevice(cfg, clock=lambda: now[0])
    dev.on_twist((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert dev.poll().twist.lin == pytest.approx((0.1, 0.0, 0.0))
    now[0] = 1.0  # no message for 1 s: the stream has stopped, so do not keep driving
    assert dev.poll().twist == ZERO


# ---- pyspacemouse fallback ---------------------------------------------------------------

class FakePySpaceMouse:
    def __init__(self):
        self.state = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=[0, 0])
        self.opened = False

    def open(self, **_):
        self.opened = True
        return True

    def read(self):
        return self.state


def test_pyspacemouse_device_reads_axes_in_order_and_buttons():
    mod = FakePySpaceMouse()
    cfg = SpaceMouseConfig(lin_scale=1.0, ang_scale=1.0, deadband=0.0)
    dev = PySpaceMouseDevice(cfg, module=mod)
    assert mod.opened
    mod.state.x, mod.state.yaw, mod.state.buttons = 0.5, -0.25, [1, 0]
    inp = dev.poll()
    assert inp.twist.lin == pytest.approx((0.5, 0.0, 0.0))
    assert inp.twist.ang == pytest.approx((0.0, 0.0, -0.25))
    assert inp.events == ("toggle_gripper",)


def test_pyspacemouse_device_raises_when_open_fails():
    mod = FakePySpaceMouse()
    mod.open = lambda **_: False
    with pytest.raises(RuntimeError):
        PySpaceMouseDevice(SpaceMouseConfig(), module=mod)


# ---- apply_input -----------------------------------------------------------------------

def test_apply_input_integrates_twist_through_filter():
    s = make_state()
    acts = apply_input(s, DeviceInput(twist=Twist((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))), fast_filter(), 0.1)
    assert acts == ["arm"]
    assert s.targets["left"].position == pytest.approx((0.41, 0.2, 1.0))  # capped to 0.1 m/s


def test_apply_input_button_events():
    s = make_state()
    acts = apply_input(s, DeviceInput(events=("toggle_gripper", "switch_arm")), fast_filter(), 0.02)
    assert acts == ["gripper", "select"]
    assert s.gripper["left"] == pytest.approx(-0.785) and s.active == "right"


def test_apply_input_continuous_gripper_delta_is_clamped():
    s = make_state()
    apply_input(s, DeviceInput(gripper_delta=-100.0), fast_filter(), 0.1)
    assert s.gripper["left"] == pytest.approx(-0.785)


def test_apply_input_holds_everything_while_paused_and_resets_filter():
    s = make_state()
    f = TwistFilter(MotionLimits(max_lin_acc=0.5))
    for _ in range(30):
        apply_input(s, DeviceInput(twist=Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0))), f, 0.02)
    s.paused = True
    before = (dict(s.targets), dict(s.gripper))
    acts = apply_input(s, DeviceInput(twist=Twist((0.1, 0, 0), (0, 0, 0)), gripper_delta=1.0,
                                      events=("toggle_gripper",)), f, 0.02)
    assert acts == [] and (dict(s.targets), dict(s.gripper)) == before
    assert f.last == ZERO
