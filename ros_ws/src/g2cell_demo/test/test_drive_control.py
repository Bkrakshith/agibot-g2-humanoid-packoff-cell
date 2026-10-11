import math

from g2cell_demo.drive_control import (DriveController, DriveParams, footprint, pose_error,
                                       rect_distance)


def simulate(start_pose, goal, dt=0.02, t_max=60.0, lag=0.0, yaw_bias=0.0):
    """Unicycle with optional first-order command lag and a steering bias (rad/s)."""
    c = DriveController(start=start_pose[:2], goal=goal)
    x, y, yaw = start_pose
    v_act = w_act = 0.0
    log, t = [], 0.0
    while t < t_max and c.phase != "done":
        v, w = c.step((x, y, yaw), dt)
        a = 1.0 if lag <= 0 else dt / (lag + dt)
        v_act += a * (v - v_act)
        w_act += a * (w - w_act)
        x += v_act * math.cos(yaw) * dt
        y += v_act * math.sin(yaw) * dt
        yaw += (w_act + (yaw_bias if abs(v_act) > 1e-3 else 0.0)) * dt
        log.append((t, v, w, x, y, yaw))
        t += dt
    return c, (x, y, yaw), log


def test_straight_drive_in_reaches_goal_within_tolerance():
    c, pose, log = simulate((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert c.phase == "done"
    dp, dyaw = pose_error(pose, (0.0, 0.0, 0.0))
    assert dp < 0.03 and dyaw < 2.0
    assert max(v for _, v, *_ in log) <= 0.3 + 1e-9


def test_speed_is_accel_limited():
    p = DriveParams()
    c, pose, log = simulate((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    dv = max(abs(b[1] - a[1]) for a, b in zip(log, log[1:]))
    # ramps are accel-limited; the only larger step is the final creep stop (<= 0.04 m/s)
    assert dv <= max(p.accel * 0.02 + 1e-6, 0.04)


def test_lateral_and_heading_offset_corrected_with_lag_and_bias():
    c, pose, log = simulate((-2.0, 0.05, math.radians(4)), (0.0, 0.0, 0.0), lag=0.15, yaw_bias=0.01)
    assert c.phase == "done"
    dp, dyaw = pose_error(pose, (0.0, 0.0, 0.0))
    assert dp < 0.03 and dyaw < 2.0
    # stays inside the U corridor: |y| of every chassis corner < 0.43 once the base enters (x > -1.5)
    for _, _, _, x, y, yaw in log:
        if x > -1.5:
            assert max(abs(py) for _, py in footprint((x, y, yaw))) < 0.43


def test_rect_distance():
    fp = footprint((0.0, 0.0, 0.0))
    assert abs(rect_distance(fp, (-0.35, 0.25, 0.43, 1.43)) - (0.43 - 0.318)) < 1e-9
    assert rect_distance(fp, (-0.1, 0.1, 0.2, 0.5)) == 0.0          # overlap
    rot = footprint((0.0, 0.0, math.radians(10)))
    assert rect_distance(rot, (-0.35, 0.25, 0.43, 1.43)) < 0.43 - 0.318
