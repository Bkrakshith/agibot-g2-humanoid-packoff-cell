import math

import pytest

from g2cell_motion.robot import (GRIP_CLOSED, GRIP_OPEN, GRIPPERS, arm_joint_names, gripper_value,
                                 home_joints, other_arm)

NAMES = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)] + \
        [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)] + ["idx31_gripper_l_inner_joint1"]


def test_gripper_convention():
    assert GRIP_CLOSED == 0.0 and GRIP_OPEN == -0.785
    assert gripper_value("open") == GRIP_OPEN and gripper_value("close") == GRIP_CLOSED
    assert gripper_value(-2.0) == GRIP_OPEN and gripper_value(0.3) == GRIP_CLOSED
    assert gripper_value(-0.3) == -0.3
    with pytest.raises(ValueError):
        gripper_value("half")
    assert GRIPPERS["left"] == "idx31_gripper_l_inner_joint1"


def test_arm_joint_names_sorted_and_split():
    left = arm_joint_names(NAMES, "left")
    assert left == sorted(left) and len(left) == 7 and all("arm_l" in n for n in left)
    assert all("arm_r" in n for n in arm_joint_names(NAMES, "right"))
    with pytest.raises(ValueError):
        arm_joint_names(NAMES, "middle")


def test_home_joints_from_degrees():
    h = home_joints(arm_joint_names(NAMES, "left"), [0, -45, 0, -75, 0, -55, 0])
    assert math.isclose(h["idx22_arm_l_joint2"], math.radians(-45))
    assert math.isclose(h["idx26_arm_l_joint6"], math.radians(-55))
    with pytest.raises(ValueError):
        home_joints(arm_joint_names(NAMES, "left"), [0, 1])


def test_other_arm():
    assert other_arm("left") == "right" and other_arm("right") == "left"
