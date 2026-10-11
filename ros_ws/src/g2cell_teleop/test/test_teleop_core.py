import math

import pytest

from g2cell_teleop.teleop_core import Pose, TeleopState, handle_key, rotate_world

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def make_state():
    return TeleopState(
        targets={"left": Pose((0.4, 0.2, 1.0), IDENTITY), "right": Pose((0.4, -0.2, 1.0), IDENTITY)},
        gripper={"left": 0.0, "right": 0.0},
        gripper_limits=(-0.785, 0.0),
    )


def test_forward_key_moves_active_arm_along_base_x():
    s = make_state()
    action = handle_key(s, "w")
    assert action == "arm"
    assert s.targets["left"].position == pytest.approx((0.4 + s.lin_step, 0.2, 1.0))
    assert s.targets["right"].position == (0.4, -0.2, 1.0)


def test_up_and_left_keys_move_along_z_and_y():
    s = make_state()
    handle_key(s, "q")
    handle_key(s, "a")
    assert s.targets["left"].position == pytest.approx((0.4, 0.2 + s.lin_step, 1.0 + s.lin_step))


def test_switching_arm_routes_motion_to_right_arm():
    s = make_state()
    assert handle_key(s, "2") == "select"
    handle_key(s, "s")
    assert s.active == "right"
    assert s.targets["right"].position == pytest.approx((0.4 - s.lin_step, -0.2, 1.0))
    assert s.targets["left"].position == (0.4, 0.2, 1.0)


def test_rotate_world_yaw_90_from_identity():
    q = rotate_world(IDENTITY, (0.0, 0.0, math.pi / 2))
    assert q == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))


def test_yaw_key_rotates_by_angular_step_about_world_z():
    s = make_state()
    handle_key(s, "u")
    half = s.ang_step / 2
    assert s.targets["left"].orientation == pytest.approx((0.0, 0.0, math.sin(half), math.cos(half)))


def test_gripper_close_is_clamped_to_limits():
    # omnipicker leader joint: 0 rad = closed, -0.785 rad = open (seen in the head camera)
    s = make_state()
    for _ in range(100):
        assert handle_key(s, "h") == "gripper"
    assert s.gripper["left"] == pytest.approx(-0.785)
    for _ in range(100):
        handle_key(s, "g")
    assert s.gripper["left"] == pytest.approx(0.0)


def test_default_close_direction_is_toward_upper_limit():
    s = make_state()
    s.gripper["left"] = -0.5
    handle_key(s, "g")
    assert s.gripper["left"] == pytest.approx(-0.4)


def test_step_size_keys_scale_and_stay_bounded():
    s = make_state()
    base = s.lin_step
    handle_key(s, "]")
    assert s.lin_step == pytest.approx(base * 2)
    for _ in range(20):
        handle_key(s, "[")
    assert s.lin_step == pytest.approx(0.001)
    for _ in range(20):
        handle_key(s, "]")
    assert s.lin_step == pytest.approx(0.05)


def test_unknown_key_changes_nothing():
    s = make_state()
    before = (dict(s.targets), dict(s.gripper), s.active, s.lin_step)
    assert handle_key(s, "z") is None
    assert (dict(s.targets), dict(s.gripper), s.active, s.lin_step) == before


def test_escape_and_x_request_quit():
    s = make_state()
    assert handle_key(s, "x") == "quit"
    assert handle_key(s, "\x1b") == "quit"


# ---------- episode control keys ----------

from g2cell_teleop.teleop_core import (  # noqa: E402
    DEFAULT_HOME_DEG,
    HELP,
    arm_joint_names,
    home_joint_targets,
    step_toward,
    switch_arm,
    toggle_gripper,
)


def test_r_toggles_recording_and_reports_start_then_stop():
    s = make_state()
    assert handle_key(s, "r") == "record_start"
    assert s.recording
    assert handle_key(s, "r") == "record_stop"
    assert not s.recording


def test_success_and_failure_marks_set_outcome():
    s = make_state()
    assert s.outcome == "unknown"
    assert handle_key(s, "n") == "mark_success"
    assert s.outcome == "true"
    assert handle_key(s, "f") == "mark_failure"
    assert s.outcome == "false"


def test_starting_a_recording_clears_the_previous_outcome():
    s = make_state()
    handle_key(s, "n")
    handle_key(s, "r")
    assert s.outcome == "unknown"


def test_zero_requests_home():
    s = make_state()
    assert handle_key(s, "0") == "home"


def test_pause_holds_targets_and_gripper_until_resumed():
    s = make_state()
    assert handle_key(s, "p") == "pause"
    assert s.paused
    before = (dict(s.targets), dict(s.gripper))
    for k in "wsadqeikjluogh":
        assert handle_key(s, k) is None
    assert (dict(s.targets), dict(s.gripper)) == before
    assert handle_key(s, "p") == "resume"
    assert not s.paused
    assert handle_key(s, "w") == "arm"


def test_record_and_marks_still_work_while_paused():
    s = make_state()
    handle_key(s, "p")
    assert handle_key(s, "r") == "record_start"
    assert handle_key(s, "f") == "mark_failure"


def test_help_lists_new_keys():
    for token in ("r ", "n ", "f ", "0 ", "p "):
        assert token in HELP
    assert "record" in HELP and "home" in HELP and "pause" in HELP


def test_arm_joint_names_follow_g2_convention():
    assert arm_joint_names("left") == [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    assert arm_joint_names("right")[0] == "idx61_arm_r_joint1"
    assert arm_joint_names("right")[-1] == "idx67_arm_r_joint7"


def test_default_home_is_scene_init_pose_in_radians():
    assert DEFAULT_HOME_DEG == [0.0, -45.0, 0.0, -75.0, 0.0, -55.0, 0.0]
    home = home_joint_targets()
    assert home["left"]["idx22_arm_l_joint2"] == pytest.approx(math.radians(-45))
    assert home["left"]["idx24_arm_l_joint4"] == pytest.approx(math.radians(-75))
    assert home["right"]["idx66_arm_r_joint6"] == pytest.approx(math.radians(-55))
    assert home["right"]["idx61_arm_r_joint1"] == 0.0
    assert len(home["left"]) == len(home["right"]) == 7


def test_home_pose_is_configurable_per_arm():
    home = home_joint_targets(left_deg=[10, 0, 0, 0, 0, 0, 0])
    assert home["left"]["idx21_arm_l_joint1"] == pytest.approx(math.radians(10))
    assert home["right"]["idx62_arm_r_joint2"] == pytest.approx(math.radians(-45))


def test_home_pose_rejects_wrong_length():
    with pytest.raises(ValueError):
        home_joint_targets(right_deg=[0, 0, 0])


def test_step_toward_is_rate_limited_and_arrives():
    cur = {"a": 0.0, "b": 1.0}
    tgt = {"a": 1.0, "b": 0.95}
    nxt, done = step_toward(cur, tgt, max_step=0.1)
    assert nxt == pytest.approx({"a": 0.1, "b": 0.95})
    assert not done
    for _ in range(20):
        cur, done = step_toward(cur, tgt, 0.1)
    assert done and cur == pytest.approx(tgt)


def test_step_toward_keeps_joints_without_a_target():
    nxt, done = step_toward({"a": 0.0, "x": 5.0}, {"a": 0.0}, 0.1)
    assert nxt == {"a": 0.0, "x": 5.0} and done


def test_switch_arm_flips_active():
    s = make_state()
    assert switch_arm(s) == "select" and s.active == "right"
    switch_arm(s)
    assert s.active == "left"


def test_toggle_gripper_goes_to_the_other_end():
    s = make_state()  # left at 0.0 = closed for omnipicker convention (closed is upper limit)
    assert toggle_gripper(s) == "gripper"
    assert s.gripper["left"] == pytest.approx(-0.785)
    toggle_gripper(s)
    assert s.gripper["left"] == pytest.approx(0.0)


def test_toggle_gripper_is_held_while_paused():
    s = make_state()
    s.paused = True
    assert toggle_gripper(s) is None
    assert s.gripper["left"] == 0.0


def test_motion_is_clamped_to_workspace_box():
    s = make_state()
    s.workspace = {"left": ((0.0, 0.0, 0.0), (0.405, 1.0, 2.0)), "right": None}
    handle_key(s, "w")
    assert s.targets["left"].position[0] == pytest.approx(0.405)
