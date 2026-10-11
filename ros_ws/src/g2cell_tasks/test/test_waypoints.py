import math

import pytest

from g2cell_tasks.waypoints import CellTargets, pick_waypoints, place_waypoints

CFG = CellTargets(crate_xy=(0.60, 0.0), crate_rim_z=1.044,
                  lanes={"L1_1": (-0.70, -0.33, 1.20)})


def test_pick_waypoints_above_grasp_lift():
    w = pick_waypoints((0.10, 0.56), 0.94)
    assert w["approach"] == pytest.approx((0.10, 0.56, 1.07))
    assert w["grasp"] == pytest.approx((0.10, 0.56, 0.93))
    assert w["lift"] == pytest.approx((0.10, 0.56, 1.11))


def test_place_waypoints_crate_match_demo():
    w = place_waypoints((0.10, 0.56), 1.11, CFG.point("crate", ""))
    assert w["via"] == pytest.approx((0.35, 0.28, max(1.11, 1.044 + 0.12)))
    assert w["above"] == pytest.approx((0.60, 0.0, 1.044 + 0.17))
    assert w["release"] == pytest.approx((0.60, 0.0, 1.044 + 0.03))
    assert w["retract"] == pytest.approx((0.60, 0.0, 1.044 + 0.20))


def test_lane_point_and_unknown():
    assert CFG.point("rack_lane", "L1_1") == (-0.70, -0.33, 1.20)
    with pytest.raises(KeyError):
        CFG.point("rack_lane", "L2_1")
    with pytest.raises(KeyError):
        CFG.point("bin", "")


def test_from_dict():
    c = CellTargets.from_dict({"crate_xy": [0.6, 0.0], "crate_rim_z": 1.044,
                               "lanes": {"L1_1": [-0.7, -0.33, 1.2]}})
    assert math.isclose(c.point("crate", "")[2], 1.044)
