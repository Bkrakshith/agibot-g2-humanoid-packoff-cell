import math

from g2cell_demo.geometry import (down_quat, duration_for, end_link_for_tool, interpolate,
                                  min_jerk, qrot)


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_down_quat_points_z_down():
    for yaw in (0.0, 0.7, -1.2):
        assert close(qrot(down_quat(yaw), (0, 0, 1)), (0, 0, -1))


def test_down_quat_yaw_rotates_x():
    x = qrot(down_quat(math.pi / 2), (1, 0, 0))
    assert close(x, (0, 1, 0))


def test_tilt_tips_fingers_toward_plus_x():
    z = qrot(down_quat(0.0, tilt=0.3), (0, 0, 1))
    assert z[0] > 0.2 and z[2] < -0.9


def test_end_link_above_tool_when_down():
    p = end_link_for_tool((0.1, 0.7, 0.95), down_quat(0.0), tool_offset=0.125, base_z=0.04)
    assert close(p, (0.1, 0.7, 0.95 + 0.125 - 0.04))


def test_min_jerk_endpoints_and_monotonic():
    assert min_jerk(0) == 0 and min_jerk(1) == 1
    v = [min_jerk(i / 50) for i in range(51)]
    assert all(b >= a for a, b in zip(v, v[1:]))


def test_interpolate_ends_on_target():
    pts = interpolate([0, 0], [1, -2], 10)
    assert len(pts) == 10 and close(pts[-1], [1, -2])


def test_duration_respects_speed():
    assert math.isclose(duration_for([0], [1.2], max_vel=0.6), 1.875 * 1.2 / 0.6)
    assert duration_for([0], [0.01]) == 0.6
