import math

from g2cell_motion.geometry import (down_quat, duration_for, end_link_for_tool, interpolate,
                                    min_jerk, qrot, quat_is_unset, segment_durations)


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_down_quat_points_z_down():
    for yaw in (0.0, 0.7, -1.2):
        assert close(qrot(down_quat(yaw), (0, 0, 1)), (0, 0, -1))


def test_down_quat_yaw_rotates_x():
    assert close(qrot(down_quat(math.pi / 2), (1, 0, 0)), (0, 1, 0))


def test_end_link_above_tool_when_down():
    p = end_link_for_tool((0.1, 0.7, 0.95), down_quat(0.0), tool_offset=0.125, base_z=0.04)
    assert close(p, (0.1, 0.7, 0.95 + 0.125 - 0.04))


def test_min_jerk_is_quintic_with_zero_boundary_velocity():
    assert min_jerk(0) == 0 and min_jerk(1) == 1
    h = 1e-6
    assert abs((min_jerk(h) - min_jerk(0)) / h) < 1e-4
    assert abs((min_jerk(1) - min_jerk(1 - h)) / h) < 1e-4
    assert math.isclose(min_jerk(0.5), 0.5)


def test_interpolate_ends_on_target():
    pts = interpolate([0, 0], [1, -2], 10)
    assert len(pts) == 10 and close(pts[-1], [1, -2])


def test_duration_respects_speed():
    assert math.isclose(duration_for([0], [1.2], max_vel=0.6), 1.875 * 1.2 / 0.6)
    assert duration_for([0], [0.01]) == 0.6


def test_quat_is_unset():
    assert quat_is_unset((0, 0, 0, 0))
    assert not quat_is_unset((0, 0, 0, 1))


def test_segment_durations_auto_when_total_zero():
    chain = [[0.0], [0.6], [0.6]]
    d = segment_durations(chain, total=0.0, max_vel=0.6, min_t=0.5)
    assert d == [duration_for([0.0], [0.6], 0.6, 0.5), 0.5]


def test_segment_durations_stretches_to_total_proportionally():
    chain = [[0.0], [0.6], [1.8]]
    auto = segment_durations(chain, 0.0, 0.6, 0.1)
    d = segment_durations(chain, total=2 * sum(auto), max_vel=0.6, min_t=0.1)
    assert all(math.isclose(x, 2 * a) for x, a in zip(d, auto))


def test_segment_durations_never_faster_than_speed_limit():
    chain = [[0.0], [1.2]]
    auto = segment_durations(chain, 0.0, 0.6, 0.1)
    assert segment_durations(chain, total=0.01, max_vel=0.6, min_t=0.1) == auto
