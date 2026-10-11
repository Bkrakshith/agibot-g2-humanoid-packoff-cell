import math

import pytest

from g2cell_teleop.motion import (
    ZERO,
    MotionLimits,
    Twist,
    TwistFilter,
    apply_twist,
    deadband,
    integrate,
    limit_norm,
)
from g2cell_teleop.teleop_core import Pose, TeleopState

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def norm(v):
    return math.sqrt(sum(c * c for c in v))


def make_state():
    return TeleopState(
        targets={"left": Pose((0.4, 0.2, 1.0), IDENTITY), "right": Pose((0.4, -0.2, 1.0), IDENTITY)},
        gripper={"left": 0.0, "right": 0.0},
        gripper_limits=(-0.785, 0.0),
    )


def test_default_limits_match_spec():
    lim = MotionLimits()
    assert lim.max_lin == 0.1 and lim.max_ang == 0.5


def test_deadband_zeroes_small_vectors_and_keeps_large_ones():
    assert deadband((0.003, 0.0, 0.002), 0.01) == (0.0, 0.0, 0.0)
    assert deadband((0.05, 0.0, 0.0), 0.01) == (0.05, 0.0, 0.0)


def test_limit_norm_preserves_direction():
    v = limit_norm((1.0, 1.0, 0.0), 0.1)
    assert norm(v) == pytest.approx(0.1)
    assert v[0] == pytest.approx(v[1])
    assert limit_norm((0.01, 0.0, 0.0), 0.1) == (0.01, 0.0, 0.0)


def test_filter_caps_linear_and_angular_speed():
    f = TwistFilter(MotionLimits(max_lin_acc=1e9, max_ang_acc=1e9))
    out = f.update(Twist((3.0, 0.0, 4.0), (0.0, 0.0, 9.0)), 0.02)
    assert norm(out.lin) == pytest.approx(0.1)
    assert norm(out.ang) == pytest.approx(0.5)


def test_filter_rate_limits_acceleration_and_then_reaches_command():
    f = TwistFilter(MotionLimits(max_lin_acc=0.5))
    out = f.update(Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.02)
    assert out.lin[0] == pytest.approx(0.01)
    for _ in range(20):
        out = f.update(Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.02)
    assert out.lin[0] == pytest.approx(0.1)


def test_filter_applies_deadband_before_rate_limit():
    f = TwistFilter(MotionLimits(lin_deadband=0.01, ang_deadband=0.05, max_lin_acc=1e9, max_ang_acc=1e9))
    out = f.update(Twist((0.005, 0.0, 0.0), (0.0, 0.04, 0.0)), 0.02)
    assert out == ZERO


def test_filter_reset_forgets_velocity():
    f = TwistFilter(MotionLimits(max_lin_acc=0.5))
    for _ in range(50):
        f.update(Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.02)
    f.reset()
    out = f.update(Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.02)
    assert out.lin[0] == pytest.approx(0.01)


def test_integrate_translates_and_rotates_in_base_frame():
    p = integrate(Pose((0.0, 0.0, 0.0), IDENTITY), Twist((0.1, 0.0, -0.05), (0.0, 0.0, 0.5)), 0.1)
    assert p.position == pytest.approx((0.01, 0.0, -0.005))
    half = 0.05 / 2
    assert p.orientation == pytest.approx((0.0, 0.0, math.sin(half), math.cos(half)))


def test_integrate_rotation_about_tilted_axis_is_unit_quaternion():
    p = integrate(Pose((0.0, 0.0, 0.0), IDENTITY), Twist(ZERO.lin, (0.3, -0.2, 0.4)), 1.0)
    assert norm(p.orientation) == pytest.approx(1.0)
    angle = 2 * math.acos(p.orientation[3])
    assert angle == pytest.approx(norm((0.3, -0.2, 0.4)))


def test_apply_twist_moves_active_arm_only():
    s = make_state()
    assert apply_twist(s, Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.1) == "arm"
    assert s.targets["left"].position == pytest.approx((0.41, 0.2, 1.0))
    assert s.targets["right"].position == (0.4, -0.2, 1.0)


def test_apply_twist_clamps_to_workspace_box():
    s = make_state()
    s.workspace = {"left": ((0.0, 0.0, 0.0), (0.405, 1.0, 2.0)), "right": None}
    apply_twist(s, Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0)
    assert s.targets["left"].position[0] == pytest.approx(0.405)


def test_apply_twist_never_pulls_an_outside_start_pose_into_the_box():
    s = make_state()  # x=0.4 is already above the box max of 0.3
    s.workspace = {"left": ((0.0, 0.0, 0.0), (0.3, 1.0, 2.0)), "right": None}
    apply_twist(s, Twist((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)), 0.1)
    assert s.targets["left"].position == pytest.approx((0.4, 0.2, 1.01))
    apply_twist(s, Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.1)
    assert s.targets["left"].position[0] == pytest.approx(0.4)  # may not move further out
    apply_twist(s, Twist((-0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.1)
    assert s.targets["left"].position[0] == pytest.approx(0.39)  # may move back toward it


def test_apply_twist_does_nothing_when_paused_or_zero():
    s = make_state()
    s.paused = True
    assert apply_twist(s, Twist((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.1) is None
    s.paused = False
    assert apply_twist(s, ZERO, 0.1) is None
    assert s.targets["left"].position == (0.4, 0.2, 1.0)
