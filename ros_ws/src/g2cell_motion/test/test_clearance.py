import math

from g2cell_motion.clearance import head_points, min_distance, path_clearance


def test_head_points_include_link_origin_and_camera():
    pts = head_points((0.1, 0.0, 1.5), (0, 0, 0, 1), cam_offset=(0.1, 0.0, 0.1))
    assert pts[0] == (0.1, 0.0, 1.5) and all(math.isclose(a, b) for a, b in zip(pts[1], (0.2, 0.0, 1.6)))


def test_head_points_rotated():
    q = (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # 90 deg about Z
    pts = head_points((0, 0, 0), q, cam_offset=(1, 0, 0))
    assert all(abs(a - b) < 1e-9 for a, b in zip(pts[1], (0, 1, 0)))


def test_min_distance():
    assert math.isclose(min_distance([(0, 0, 0), (3, 0, 0)], [(0, 4, 0), (3, 1, 0)]), 1.0)


def test_path_clearance_samples_between_waypoints():
    # one joint q; link point at (q, 0, 0); head at (0.5, 0, 0): the path 0 -> 1 passes through it
    fk = lambda joints: [(joints["q"], 0.0, 0.0)]
    c = path_clearance([{"q": 0.0}, {"q": 1.0}], fk, [(0.5, 0.0, 0.0)], samples=8)
    assert c < 0.05
    # endpoints alone would have said 0.5
    c2 = path_clearance([{"q": 0.0}, {"q": 0.2}], fk, [(0.5, 0.0, 0.0)], samples=8)
    assert math.isclose(c2, 0.3, abs_tol=1e-9)


def test_path_clearance_fk_failure_is_zero():
    assert path_clearance([{"q": 0.0}, {"q": 1.0}], lambda j: None, [(0, 0, 0)]) == 0.0


def test_path_clearance_single_pose_checks_it():
    fk = lambda joints: [(joints["q"], 0.0, 0.0)]
    assert math.isclose(path_clearance([{"q": 0.0}], fk, [(0.5, 0, 0)]), 0.5)


def test_path_clearance_fk_moves_heads_with_each_sample():
    from g2cell_motion.clearance import path_clearance_fk

    # head turns with the waist "w": link point and head both rotate -> distance constant
    def fk(j):
        import math
        c, s = math.cos(j["w"]), math.sin(j["w"])
        return [(0.5 * c, 0.5 * s, 0.0)], [(0.1 * c, 0.1 * s, 0.0)]

    d = path_clearance_fk([{"w": 0.0}, {"w": 1.5}], fk)
    assert abs(d - 0.4) < 1e-9
    assert path_clearance_fk([{"w": 0.0}, {"w": 1.0}], lambda j: None) == 0.0


# ---- fixture boxes (cell_colliders.json OBBs): the arm must not sweep through the carton walls ----------
import json  # noqa: E402

from g2cell_motion.clearance import (box_distance, load_fixture_boxes, min_box_distance,  # noqa: E402
                                     path_clearance_fk)

WALL = {"name": "/World/Cartons/Carton_Pack/Colliders/WallPosY", "c": [0.6, 0.298, 0.949],
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "h": [0.196, 0.002, 0.121]}


def test_box_distance_outside_inside_and_rotated():
    assert math.isclose(box_distance((0.6, 0.298, 1.17), WALL), 1.17 - 1.07, abs_tol=1e-9)   # above the rim
    assert box_distance((0.6, 0.298, 1.0), WALL) < 0                                          # in the wall
    rot = {"c": [0, 0, 0], "axes": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "h": [1.0, 0.1, 0.1]}  # long along Y
    assert box_distance((0.0, 0.9, 0.0), rot) < 0 and math.isclose(box_distance((0.9, 0, 0), rot), 0.8)


def test_load_fixture_boxes_skips_movable_bags_and_far_boxes(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([WALL,
                             {**WALL, "name": "/World/Bags/Bag_01/Colliders/Box"},
                             {**WALL, "name": "/World/LineBags/Bag_InBin/Colliders/Box"},
                             {**WALL, "name": "/World/Palletiser/X", "c": [3.9, -1.8, 0.3]}]))
    boxes = load_fixture_boxes(str(p), radius=1.6)
    assert [b["name"] for b in boxes] == [WALL["name"]]


def test_min_box_distance():
    assert math.isclose(min_box_distance([(0.6, 0.298, 1.17), (0.6, 0.298, 1.12)], [WALL]), 0.05, abs_tol=1e-9)
    assert min_box_distance([(0, 0, 0)], []) == float("inf")


def test_path_through_a_fixture_is_unsafe_even_when_far_from_the_head():
    # link point at (0.6, y=q, 1.09): the path y 0 -> 0.6 crosses the carton's +Y wall 2 cm above its top
    fk = lambda j: ([(0.6, j["q"], 1.09)], [(5.0, 0.0, 0.0)])
    free = path_clearance_fk([{"q": 0.0}, {"q": 0.6}], fk)
    assert free > 4.0                                   # head-only: fine
    hit = path_clearance_fk([{"q": 0.0}, {"q": 0.6}], fk, fixtures=[WALL], fixture_margin=0.04)
    assert hit == 0.0
    high = lambda j: ([(0.6, j["q"], 1.20)], [(5.0, 0.0, 0.0)])
    assert path_clearance_fk([{"q": 0.0}, {"q": 0.6}], high, fixtures=[WALL], fixture_margin=0.04) > 4.0


def test_fixture_only_points_are_not_head_checked():
    # fk may return a third list: points checked against fixtures only (fingertips)
    fk = lambda j: ([(0.0, 0.0, 2.0)], [(0.0, 0.0, 0.0)], [(0.6, 0.298, 1.0)])
    assert path_clearance_fk([{"q": 0.0}], fk) == 2.0
    assert path_clearance_fk([{"q": 0.0}], fk, fixtures=[WALL]) == 0.0
