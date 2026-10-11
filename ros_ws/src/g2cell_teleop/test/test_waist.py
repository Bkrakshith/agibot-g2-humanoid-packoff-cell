import math

import pytest

from g2cell_teleop.teleop_core import HELP, Pose, TeleopState, handle_key, status_line
from g2cell_teleop.waist import auto_face, waist_config_from

IDENTITY = (0.0, 0.0, 0.0, 1.0)
D = math.radians


def make_state(**kw):
    return TeleopState(
        targets={"left": Pose((0.4, 0.2, 1.0), IDENTITY), "right": Pose((0.4, -0.2, 1.0), IDENTITY)},
        gripper={"left": 0.0, "right": 0.0}, gripper_limits=(-0.785, 0.0), **kw)


def test_comma_and_period_yaw_the_waist_in_5_degree_steps():
    s = make_state()
    assert s.waist == 0.0 and s.waist_step == pytest.approx(D(5))
    assert handle_key(s, ",") == "waist"
    assert s.waist == pytest.approx(D(5))          # ',' = turn left (+yaw)
    handle_key(s, ".")
    handle_key(s, ".")
    assert s.waist == pytest.approx(D(-5))


def test_waist_is_clamped():
    s = make_state(waist_limits=(D(-10), D(10)))
    for _ in range(10):
        handle_key(s, ",")
    assert s.waist == pytest.approx(D(10))
    for _ in range(10):
        handle_key(s, ".")
    assert s.waist == pytest.approx(D(-10))


def test_waist_held_across_arm_moves_and_arm_switch():
    s = make_state()
    handle_key(s, ",")
    for k in "wasd2qe1":
        handle_key(s, k)
    assert s.waist == pytest.approx(D(5))


def test_waist_keys_ignored_while_paused():
    s = make_state()
    handle_key(s, "p")
    assert handle_key(s, ",") is None and s.waist == 0.0


def test_t_toggles_auto_face():
    s = make_state()
    assert handle_key(s, "t") == "auto_face_on" and s.auto_face
    assert handle_key(s, "t") == "auto_face_off" and not s.auto_face


def test_help_and_status_show_the_waist():
    assert "," in HELP and "." in HELP and "waist" in HELP and "t " in HELP
    s = make_state()
    handle_key(s, ",")
    handle_key(s, "t")
    line = status_line(s)
    assert "waist=+5" in line and "AUTO-FACE" in line


def test_auto_face_turns_toward_active_target():
    cfg = waist_config_from({"auto_face_cone_deg": 30.0, "waist_min_deg": -100.0, "waist_max_deg": 100.0,
                             "auto_face_pivot": [0.0, 0.0], "waist_joint": "idx05_body_joint5"})
    s = make_state()
    s.targets["left"] = Pose((0.0, 0.8, 1.0), IDENTITY)    # 90 deg left
    d = auto_face(s, cfg)
    assert d.yaw == pytest.approx(D(60)) and not d.needs_base
    s.targets["left"] = Pose((0.6, 0.0, 1.0), IDENTITY)    # straight ahead from 0: no turn
    assert auto_face(make_state(), cfg).yaw == 0.0


def test_auto_face_respects_teleop_limits():
    cfg = waist_config_from({"auto_face_cone_deg": 30.0, "waist_min_deg": -45.0, "waist_max_deg": 45.0,
                             "auto_face_pivot": [0.0, 0.0], "waist_joint": "idx05_body_joint5"})
    s = make_state()
    s.targets["left"] = Pose((0.0, 0.8, 1.0), IDENTITY)
    d = auto_face(s, cfg)
    assert d.yaw == pytest.approx(D(45)) and d.needs_base
