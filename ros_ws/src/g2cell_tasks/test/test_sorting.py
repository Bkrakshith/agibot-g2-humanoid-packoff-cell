import pytest

from g2cell_tasks.sorting import SortRules

RULES = SortRules(lanes={"reject": "L1_1", "sample": "L1_2", "SKU_ROCKET": "L2_1"}, default_lane="L2_2",
                  known_lanes=["L1_1", "L1_2", "L2_1", "L2_2"])


def test_good_goes_to_crate():
    assert RULES.target_for("good") == ("crate", "")


def test_reject_and_skus_go_to_lanes():
    assert RULES.target_for("reject") == ("rack_lane", "L1_1")
    assert RULES.target_for("SKU_ROCKET") == ("rack_lane", "L2_1")
    assert RULES.target_for("SKU_UNKNOWN") == ("rack_lane", "L2_2")


def test_resolve_uses_goal_when_given():
    assert RULES.resolve("", "", "good") == ("crate", "")
    assert RULES.resolve("crate", "", "reject") == ("crate", "")
    assert RULES.resolve("rack_lane", "L1_2", "good") == ("rack_lane", "L1_2")
    assert RULES.resolve("rack_lane", "", "reject") == ("rack_lane", "L1_1")
    assert RULES.resolve("rack_lane", "", "good") == ("rack_lane", "L2_2")


@pytest.mark.parametrize("target,lane", [("bin", ""), ("rack_lane", "L9_9")])
def test_resolve_rejects_bad_goal(target, lane):
    with pytest.raises(ValueError):
        RULES.resolve(target, lane, "good")


def test_bad_config():
    with pytest.raises(ValueError):
        SortRules(lanes={"reject": "L7"}, default_lane="L1_1", known_lanes=["L1_1"])
