from g2cell_tasks.grasp import grasp_from_gripper


def test_unknown_without_measurement():
    assert grasp_from_gripper(None) is None


def test_fully_closed_means_empty():
    assert grasp_from_gripper(0.0) is False and grasp_from_gripper(-0.02) is False


def test_stopped_short_means_object():
    assert grasp_from_gripper(-0.3) is True


def test_still_open_is_not_a_grasp():
    assert grasp_from_gripper(-0.78) is False
