"""Pure motion helpers: quintic profile, move durations from GDK velocities, command buffer."""
import pytest

from agibot_gdk import _motion as m


def test_min_jerk_ends_and_midpoint():
    assert m.min_jerk(0.0) == 0.0 and m.min_jerk(1.0) == 1.0 and m.min_jerk(0.5) == pytest.approx(0.5)
    assert m.min_jerk(-1) == 0.0 and m.min_jerk(2) == 1.0


def test_interpolate_excludes_start_includes_goal():
    path = m.interpolate([0.0, 1.0], [1.0, 1.0], 4)
    assert len(path) == 4 and path[-1] == [1.0, 1.0] and 0 < path[0][0] < 0.25


def test_duration_keeps_every_joint_under_its_velocity():
    # quintic peak speed = 1.875 * distance / T
    t = m.move_duration([0.0, 0.0], [0.4, 0.1], [0.2, 0.2])
    assert t == pytest.approx(1.875 * 0.4 / 0.2)


def test_duration_uses_per_joint_velocity_and_the_slowest_joint_wins():
    t = m.move_duration([0.0, 0.0], [0.1, 0.4], [0.05, 1.0])
    assert t == pytest.approx(1.875 * 0.1 / 0.05)


def test_one_velocity_applies_to_all_and_missing_uses_default():
    assert m.normalize_velocities([], 3, default=0.3) == [0.3, 0.3, 0.3]
    assert m.normalize_velocities([0.2], 2, default=0.3) == [0.2, 0.2]
    assert m.normalize_velocities([0.1, 0.0], 2, default=0.3) == [0.1, 0.3]  # zero: default
    with pytest.raises(ValueError):
        m.normalize_velocities([0.1, 0.2], 3, default=0.3)


def test_velocity_is_capped_by_the_joint_limit():
    assert m.cap_velocities(["idx11_head_joint1", "idx21_arm_l_joint1"], [9.0, 0.2]) == [2.5, 0.2]


def test_short_moves_have_a_minimum_duration():
    assert m.move_duration([0.0], [0.0], [0.3]) == m.MIN_MOVE_TIME


def test_buffer_generations_detect_preemption():
    buf = m.CommandBuffer()
    token = buf.claim(["a", "b"])
    buf.set({"a": 1.0, "b": 2.0})
    assert buf.owns(["a", "b"], token)
    buf.claim(["b"])
    assert not buf.owns(["a", "b"], token)
    assert buf.snapshot() == {"a": 1.0, "b": 2.0}
    assert buf.get("a") == 1.0 and buf.get("zz") is None
