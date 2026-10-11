import pytest

from g2cell_tasks.outcome import OUTCOME_CODES, Placement, classify


def test_codes_match_interface_constants():
    assert OUTCOME_CODES == {"success": 0, "wrong_target": 1, "drop": 2, "regrasp": 3, "overflow": 4,
                             "unverified": 10, "plan_failed": 11, "grasp_failed": 12, "unreachable": 13,
                             "aborted": 14, "invalid": 15}


def test_no_observation_is_unverified_not_success():
    assert classify("crate", "", None, 1) == ("unverified", False)


def test_success_and_regrasp():
    assert classify("crate", "", Placement("crate"), 1) == ("success", True)
    assert classify("crate", "", Placement("crate"), 2) == ("regrasp", True)


def test_wrong_target_and_lane():
    assert classify("crate", "", Placement("rack_lane", "L1_1"), 1) == ("wrong_target", True)
    assert classify("rack_lane", "L1_1", Placement("rack_lane", "L1_2"), 1) == ("wrong_target", True)


@pytest.mark.parametrize("where", ["floor", "conveyor"])
def test_drop(where):
    assert classify("crate", "", Placement(where), 1) == ("drop", True)


def test_dropped_flag_wins():
    assert classify("crate", "", None, 1, dropped=True) == ("drop", True)


def test_overflow_beats_regrasp_but_not_wrong_target():
    assert classify("crate", "", Placement("crate", proud_of_rim=True), 3) == ("overflow", True)
    assert classify("rack_lane", "L1_1", Placement("crate", proud_of_rim=True), 1) == ("wrong_target", True)
