"""Teleport reset requests, reset check and the placement scorer (pure; poses as the relay publishes them)."""
import math

import pytest

from g2cell_datagen.randomize import EpisodeSpec
from g2cell_datagen.scene_io import (CARTON, TOTES, check_reset, placement_of, score, teleport_requests)

I = (1.0, 0.0, 0.0, 0.0)
SPEC = EpisodeSpec(seed=3, bag_id="Bag_01", x=-0.2, y=0.7, yaw_deg=-4.7, label="good", target="crate", lane="",
                   parked={"Bag_02": (-0.4, 1.4, 0.0, 0.0), "Bag_03": (-0.1, 1.4, 0.0, 0.0)})
RESTORE = {CARTON: [0.6, 0.0, 0.82, 0.0]}


def rz(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def test_teleport_requests_place_every_bag_and_restore_the_carton():
    reqs = teleport_requests(SPEC, RESTORE)
    by = {r["path"]: r for r in reqs}
    assert by["/World/Bags/Bag_01"] == {"path": "/World/Bags/Bag_01", "xyz": [-0.2, 0.7, 0.87 + 0.002],
                                         "yaw_deg": -4.7, "stop": True}
    assert by["/World/Bags/Bag_02"]["xyz"] == [-0.4, 1.4, 0.0 + 0.002]
    assert by[CARTON] == {"path": CARTON, "xyz": [0.6, 0.0, 0.82 + 0.002], "yaw_deg": 0.0, "stop": True}
    assert len(reqs) == 4


def test_check_reset_compares_positions_and_yaw():
    reqs = teleport_requests(SPEC, RESTORE)
    poses = {r["path"]: ((r["xyz"][0], r["xyz"][1], r["xyz"][2] - 0.002), rz(r["yaw_deg"])) for r in reqs}
    ok, worst = check_reset(poses, reqs)
    assert ok and worst["err_m"] < 1e-9
    poses["/World/Bags/Bag_01"] = ((-0.2, 0.75, 0.87), rz(-4.7))
    ok, worst = check_reset(poses, reqs)
    assert not ok and worst["path"] == "/World/Bags/Bag_01" and math.isclose(worst["err_m"], 0.05, abs_tol=1e-9)
    ok, worst = check_reset({}, reqs)                 # nothing seen from the relay
    assert not ok and worst["err_m"] == float("inf")
    # a box bag is the same after 180 deg: yaw error is taken modulo 180
    poses = {r["path"]: ((r["xyz"][0], r["xyz"][1], r["xyz"][2]), rz(r["yaw_deg"] + 180.0)) for r in reqs}
    assert check_reset(poses, reqs)[0]


CARTON_POSE = ((0.6, 0.0, 0.82), I)
TOTE_POSE = ((0.0, -0.823, 0.817), (0.0, 0.0, 0.1736, 0.9848))    # turned 180 deg about Z, tilted 20 deg
POSES = {CARTON: CARTON_POSE, TOTES["L1_1"]: TOTE_POSE}


def test_placement_in_carton_tote_table_floor():
    p = placement_of(((0.6, 0.02, 0.83), rz(10)), POSES, spawn=(-0.2, 0.7))
    assert p["target"] == "crate" and not p["proud_of_rim"]
    p = placement_of(((0.6, 0.0, 1.06), I), POSES, spawn=(-0.2, 0.7))       # sticking out of the carton
    assert p["target"] == "crate" and p["proud_of_rim"]
    p = placement_of(((0.0, -0.80, 0.86), I), POSES, spawn=(-0.2, 0.7))      # in the tote
    assert p["target"] == "rack_lane" and p["lane"] == "L1_1"
    p = placement_of(((-0.2, 0.71, 0.87), I), POSES, spawn=(-0.2, 0.7))      # never left the table
    assert p["target"] == "table" and p["at_spawn"]
    p = placement_of(((0.9, 0.5, 0.0), I), POSES, spawn=(-0.2, 0.7))
    assert p["target"] == "floor"
    p = placement_of(((0.6, 1.0, 0.83), I), POSES, spawn=(-0.2, 0.7))        # on the belt, not in the carton
    assert p["target"] == "conveyor"


def test_score_uses_the_placement_only_when_the_motion_completed():
    crate = {"target": "crate", "lane": "", "proud_of_rim": False, "at_spawn": False}
    assert score("unverified", "crate", "", crate, 1) == ("success", True)
    assert score("unverified", "crate", "", crate, 2) == ("regrasp", True)
    assert score("unverified", "rack_lane", "L1_1", crate, 1) == ("wrong_target", True)
    table = {"target": "table", "lane": "", "proud_of_rim": False, "at_spawn": True}
    assert score("unverified", "crate", "", table, 1) == ("grasp_failed", True)
    floor = {"target": "floor", "lane": "", "proud_of_rim": False, "at_spawn": False}
    assert score("unverified", "crate", "", floor, 1) == ("drop", True)
    assert score("plan_failed", "crate", "", table, 1) == ("plan_failed", False)   # task's own failure wins
    assert score("unverified", "crate", "", None, 1) == ("unverified", False)      # nothing observed
    proud = {**crate, "proud_of_rim": True}
    assert score("unverified", "crate", "", proud, 1) == ("overflow", True)
