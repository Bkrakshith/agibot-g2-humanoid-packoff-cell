"""Joint limits: reject outside the REAL soft limits (as the robot does), clamp into the sim URDF limits."""
import pytest

from agibot_gdk import _limits as lim
from conftest import URDF


def test_joint_groups_have_22_body_joints_in_gdk_order():
    assert len(lim.ALL) == 22
    assert lim.ALL[:5] == [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
    assert lim.HEAD == ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
    assert lim.LEFT_ARM[0] == "idx21_arm_l_joint1" and lim.RIGHT_ARM[-1] == "idx67_arm_r_joint7"
    assert lim.GRIPPERS == {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}


def test_effective_limit_is_the_intersection():
    # head yaw: real +-1.571, sim URDF +-0.781
    assert lim.effective_limits("idx11_head_joint1") == pytest.approx((-0.781, 0.781))
    # head roll: real +-0.535, sim -0.0181 .. 0.5231
    assert lim.effective_limits("idx13_head_joint3") == pytest.approx((-0.0181, 0.5231))
    # waist joint3: real -1.920 .. 1.571, sim -1.907 .. 1.039
    assert lim.effective_limits("idx03_body_joint3") == pytest.approx((-1.907, 1.039))
    # arms: identical up to rounding
    lo, hi = lim.effective_limits("idx24_arm_l_joint4")
    assert lo == pytest.approx(-2.4959, abs=1e-3) and hi == pytest.approx(1.012, abs=1e-3)


def test_every_joint_has_real_and_sim_limits():
    for name in lim.ALL:
        r, s = lim.REAL_SOFT_LIMITS[name], lim.SIM_URDF_LIMITS[name]
        assert r[0] < r[1] and s[0] < s[1]


def test_target_outside_the_real_limits_is_rejected_like_the_robot():
    with pytest.raises(lim.LimitError, match="outside"):
        lim.check_and_clamp(["idx13_head_joint3"], [9.0])


def test_unknown_joint_is_rejected():
    with pytest.raises(lim.LimitError, match="unknown joint"):
        lim.check_and_clamp(["idx99_nope"], [0.0])


def test_target_inside_real_but_outside_sim_is_clamped_with_a_note():
    values, notes = lim.check_and_clamp(["idx11_head_joint1", "idx21_arm_l_joint1"], [1.2, 0.5])
    assert values == pytest.approx([0.781, 0.5])
    assert len(notes) == 1 and "idx11_head_joint1" in notes[0]


def test_tiny_overshoot_of_the_real_limit_is_tolerated():
    values, _ = lim.check_and_clamp(["idx27_arm_l_joint7"], [1.5365])
    assert values == pytest.approx([1.5359])


def test_gripper_value_is_clamped_to_open_closed_range():
    assert lim.clamp_gripper(-2.0) == pytest.approx(-0.785, abs=1e-3)
    assert lim.clamp_gripper(0.3) == 0.0
    assert lim.clamp_gripper(-0.4) == -0.4


def test_sim_table_matches_the_urdf():
    if not URDF.exists():
        pytest.skip("URDF not available")
    parsed = lim.parse_urdf_limits(URDF.read_text())
    for name in lim.ALL:
        assert parsed[name] == pytest.approx(lim.SIM_URDF_LIMITS[name], abs=1e-4), name
    assert parsed["idx31_gripper_l_inner_joint1"] == pytest.approx((-0.7854, 0.0), abs=1e-4)


def test_sim_is_never_wider_than_real_except_rounding():
    for name in lim.ALL:
        (rl, rh), (sl, sh) = lim.REAL_SOFT_LIMITS[name], lim.SIM_URDF_LIMITS[name]
        assert sl >= rl - 1e-3 and sh <= rh + 1e-3, name


def test_measured_sim_drift_past_a_real_limit_does_not_make_the_target_invalid():
    # the sim's waist joint 1 rests at +0.005 rad, above the real upper limit 0.0;
    # commanding that measured value back (exercise 4 "go <saved pose>") must work
    values, _ = lim.check_and_clamp(["idx01_body_joint1"], [0.005], current={"idx01_body_joint1": 0.005})
    assert values == pytest.approx([0.0])  # clamped into real ∩ sim
    # but moving further out than the joint already is stays an error
    with pytest.raises(lim.LimitError, match="outside"):
        lim.check_and_clamp(["idx01_body_joint1"], [0.05], current={"idx01_body_joint1": 0.005})
    # a saved pose a few mrad past the limit is accepted even after the joint was clamped back to it
    values, _ = lim.check_and_clamp(["idx01_body_joint1"], [0.005], current={"idx01_body_joint1": 0.0})
    assert values == pytest.approx([0.0])
    with pytest.raises(lim.LimitError, match="outside"):
        lim.check_and_clamp(["idx01_body_joint1"], [0.02])
