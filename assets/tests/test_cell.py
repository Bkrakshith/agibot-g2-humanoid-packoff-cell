"""Tests for the composed cell scene written by assets/cell/build.py.

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_cell.py
(build first: ../.venv-tools/bin/python assets/cell/build.py)
"""

import sys
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell"))
import build  # noqa: E402

CELL = build.DEFAULT_DEST / "cell.usda"
TOL = 0.020  # spec acceptance tolerance


@pytest.fixture(scope="module")
def stage():
    assert CELL.is_file(), f"missing {CELL}; run assets/cell/build.py"
    s = Usd.Stage.Open(str(CELL))
    assert s is not None
    return s


def _bbox(stage, path, purposes=("default", "render")):
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid(), path
    if path.__str__().endswith("InteriorVolume"):
        purposes = ("default", "render", "guide")
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), list(purposes))
    return bc.ComputeWorldBound(prim).ComputeAlignedRange()


def test_stage_metadata(stage):
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    ps = stage.GetPrimAtPath("/PhysicsScene")
    assert ps.IsA(UsdPhysics.Scene)
    assert {"NewtonSceneAPI", "PhysxSceneAPI"} <= set(ps.GetPrimTypeInfo().GetAppliedAPISchemas() or ps.GetAppliedSchemas())


@pytest.mark.parametrize("path,cx,cy", [
    ("/World/Conveyor", 0.60, 0.0),
    ("/World/PickTable", *build.LAYOUT["pick_table"][:2]),
    ("/World/SortRack", *build.LAYOUT["sort_rack"][:2]),
])
def test_fixture_centres(stage, path, cx, cy):
    r = _bbox(stage, path)
    c = (r.GetMin() + r.GetMax()) / 2
    assert abs(c[0] - cx) < TOL and abs(c[1] - cy) < TOL, (path, c)
    assert abs(r.GetMin()[2]) < TOL


# ---------------------------------------------------------------- Layout v2: the U
CLEAR = 0.10   # required clearance per side for driving in


def test_u_inner_width_leaves_drive_in_clearance(stage):
    table = _bbox(stage, "/World/PickTable")
    rack_inner = max(_bbox(stage, t.GetPath()).GetMax()[1] for t in stage.GetPrimAtPath("/World/Totes").GetChildren())
    rack_inner = max(rack_inner, _bbox(stage, "/World/SortRack").GetMax()[1])
    assert table.GetMin()[1] >= build.CHASSIS_HALF_W + CLEAR
    assert rack_inner <= -(build.CHASSIS_HALF_W + CLEAR)


def test_u_opens_toward_minus_x(stage):
    """Nothing but floor in the drive-in corridor |Y| <= 0.43 between the room wall and the conveyor."""
    conv = _bbox(stage, "/World/Conveyor")
    for p in stage.GetPrimAtPath("/World").GetChildren():
        if p.GetName() in ("Room", "Conveyor", "Props", "Cartons", "LineBags"):
            continue
        for k in ([p] if p.GetName() not in ("Totes", "Bags") else p.GetChildren()):
            r = _bbox(stage, k.GetPath())
            if r.IsEmpty() or r.GetMin()[0] > conv.GetMin()[0]:
                continue
            assert r.GetMin()[1] >= build.U_INNER_HALF - 1e-3 or r.GetMax()[1] <= -build.U_INNER_HALF + 1e-3, k.GetPath()


def test_rack_faces_robot_and_clears_conveyor(stage):
    rack = _bbox(stage, "/World/SortRack")
    conv = _bbox(stage, "/World/Conveyor")
    assert rack.GetMax()[0] < conv.GetMin()[0] - 0.02
    # open front (+X local) turned to +Y: every tote slot's down-slope axis points to +Y
    from pxr import Gf
    for t in stage.GetPrimAtPath("/World/Totes").GetChildren():
        m = UsdGeom.Xformable(t).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        down = -m.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized()   # tote asset -Y = slot +X (down-slope)
        assert down[1] > 0.9 and down[2] < -0.3, (t.GetPath(), down)   # toward the robot, tilted 20 deg


def test_four_totes_in_slots(stage):
    totes = stage.GetPrimAtPath("/World/Totes").GetChildren()
    assert len(totes) == 2      # Layout v2: one lane column x 2 levels
    rack = _bbox(stage, "/World/SortRack")
    for t in totes:
        assert t.HasAPI(UsdPhysics.RigidBodyAPI)
        r = _bbox(stage, t.GetPath())
        assert rack.GetMin()[0] - TOL < r.GetMin()[0] and r.GetMax()[0] < rack.GetMax()[0] + TOL
        assert rack.GetMin()[1] - TOL < r.GetMin()[1] and r.GetMax()[1] < rack.GetMax()[1] + 0.15


def test_table_clear_of_robot_chassis_and_conveyor(stage):
    r = _bbox(stage, "/World/PickTable")
    assert r.GetMin()[1] > 0.318 + 0.02          # G2 chassis half-width 0.318 m
    c = _bbox(stage, "/World/Conveyor")
    assert r.GetMax()[0] < c.GetMin()[0] - 0.02


# ---------------------------------------------------------------- Layout v2: cartons + end of line
def _point(stage, path):
    from pxr import Gf
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return m.Transform(Gf.Vec3d(0, 0, 0))


def test_no_crates_on_the_line(stage):
    assert not stage.GetPrimAtPath("/World/Crates").IsValid()


def test_pack_carton_on_belt(stage):
    r = _bbox(stage, "/World/Cartons/Carton_Pack")
    c = (r.GetMin() + r.GetMax()) / 2
    assert abs(c[0] - 0.60) < TOL and abs(c[1]) < TOL
    assert abs(r.GetMin()[2] - 0.82) < 0.002
    assert stage.GetPrimAtPath("/World/Cartons/Carton_Pack/InteriorVolume").IsValid()


def test_open_cartons_on_belt_do_not_touch(stage):
    belt = [c for c in stage.GetPrimAtPath("/World/Cartons").GetChildren()
            if c.GetName().startswith(("Carton_Up", "Carton_Pack", "Carton_Down"))]
    assert len(belt) >= 3
    rs = sorted((_bbox(stage, c.GetPath()) for c in belt), key=lambda r: r.GetMin()[1])
    for a, b in zip(rs, rs[1:]):
        assert a.GetMax()[1] < b.GetMin()[1]
    for r in rs:
        assert r.GetMin()[1] > -1.80 and r.GetMax()[1] < 1.80 and abs(r.GetMin()[2] - 0.82) < 0.002
        assert all(c.HasAPI(UsdPhysics.RigidBodyAPI) for c in belt)


def test_erector_feeds_conveyor_upstream_end(stage):
    sp = _point(stage, "/World/CaseErector/Outfeed/SpawnPoint")
    assert abs(sp[0] - 0.60) < TOL and sp[1] > 1.80 and abs(sp[2] - 0.82) < 0.002
    r = _bbox(stage, "/World/CaseErector")
    assert abs(r.GetMin()[1] - 1.80) < TOL
    c = _bbox(stage, "/World/Cartons/Carton_Erector")
    assert abs((c.GetMin()[1] + c.GetMax()[1]) / 2 - sp[1]) < TOL


def test_line_continuity_downstream(stage):
    """conveyor end -> drop gap (floor bin) -> transfer -> sealer -> palletiser, bed tops 0.82."""
    rat = _bbox(stage, "/World/TransferRAT")
    assert abs(rat.GetMax()[2] - 0.82) < 0.002
    assert abs(rat.GetMax()[1] - (-1.80 - build.DROP_GAP)) < TOL     # gap after the conveyor end
    entry = _point(stage, "/World/CaseSealer/Infeed/EntryPoint")
    exit_ = _point(stage, "/World/CaseSealer/Outfeed/ExitPoint")
    assert abs(entry[0] - rat.GetMax()[0]) < TOL and abs(entry[2] - 0.82) < 0.002
    assert exit_[0] > entry[0] + 1.0 and abs(exit_[1] - entry[1]) < 1e-6     # flow along +X
    pe = _point(stage, "/World/Palletiser/Infeed/EntryPoint")
    assert (pe - exit_).GetLength() < TOL
    assert stage.GetPrimAtPath("/World/Palletiser/StationA/Stack").IsValid()
    for name, path in (("Carton_Sealer", "/World/CaseSealer"), ("Carton_PalletInfeed", "/World/Palletiser")):
        c, m = _bbox(stage, f"/World/Cartons/{name}"), _bbox(stage, path)
        assert all(m.GetMin()[k] <= c.GetMin()[k] and c.GetMax()[k] <= m.GetMax()[k] for k in range(3)), name


def test_floor_bin_catches_the_drop_gap(stage):
    b = _bbox(stage, "/World/FloorBin")
    vol = _bbox(stage, "/World/FloorBin/InteriorVolume")
    conv = _bbox(stage, "/World/Conveyor")
    rat = _bbox(stage, "/World/TransferRAT")
    # the whole gap (conveyor end .. transfer edge) lies over the bin interior, across the belt width
    assert vol.GetMax()[1] >= conv.GetMin()[1] - 0.06 and vol.GetMin()[1] <= rat.GetMax()[1]
    assert vol.GetMin()[0] <= 0.60 - 0.225 and vol.GetMax()[0] >= 0.60 + 0.225   # 450 mm between guides
    assert b.GetMax()[2] < 0.72 - 1e-3                                             # under the transfer frame
    assert b.GetMax()[1] <= conv.GetMin()[1] + 1e-3                                # not under the conveyor legs


def test_end_of_line_inside_room_and_clear_of_each_other(stage):
    room_in = ((-4.4, 5.4), (-4.9, 4.9))
    boxes = {n: _bbox(stage, f"/World/{n}") for n in
             ("Conveyor", "PickTable", "SortRack", "CaseErector", "CaseSealer", "Palletiser", "TransferRAT")}
    for n, r in boxes.items():
        assert room_in[0][0] < r.GetMin()[0] and r.GetMax()[0] < room_in[0][1], n
        assert room_in[1][0] < r.GetMin()[1] and r.GetMax()[1] < room_in[1][1], n
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = boxes[a], boxes[b]
            assert not all(ra.GetMin()[k] < rb.GetMax()[k] - 1e-3 and rb.GetMin()[k] < ra.GetMax()[k] - 1e-3
                           for k in range(3)), (a, b)
    for p in stage.GetPrimAtPath("/World/Props").GetChildren():
        rp = _bbox(stage, p.GetPath())
        for n, r in boxes.items():
            assert not all(rp.GetMin()[k] < r.GetMax()[k] and r.GetMin()[k] < rp.GetMax()[k] for k in range(2)), (p.GetName(), n)


def test_robot_start_is_in_open_floor(stage):
    """Mobile scene start (-2.0, 0): chassis footprint clear of everything by >= 0.3 m."""
    x0, x1, y0, y1 = -2.0 - 0.374 - 0.3, -2.0 + 0.382 + 0.3, -0.318 - 0.3, 0.318 + 0.3
    for p in stage.GetPrimAtPath("/World").GetChildren():
        if p.GetName() == "Room":
            continue
        for k in (p.GetChildren() if p.GetName() in ("Props", "Totes", "Bags", "Cartons", "LineBags") else [p]):
            r = _bbox(stage, k.GetPath())
            if r.IsEmpty():
                continue
            assert not (r.GetMin()[0] < x1 and r.GetMax()[0] > x0 and r.GetMin()[1] < y1 and r.GetMax()[1] > y0), k.GetPath()


def test_bags_on_table(stage):
    bags = stage.GetPrimAtPath("/World/Bags").GetChildren()
    assert 2 <= len(bags) <= 6
    for b in bags:
        assert b.HasAPI(UsdPhysics.RigidBodyAPI)
        assert abs(UsdPhysics.MassAPI(b).GetMassAttr().Get() - 0.12) < 1e-6
        r = _bbox(stage, b.GetPath())
        size = r.GetSize()
        assert abs(size[2] - 0.07) < 0.002
        assert abs(r.GetMin()[2] - 0.87) < 0.002
        # inside the table top (turned -90: 1.00 (X) x 0.60 (Y) m around the table centre, minus the lips)
        tx, ty, _ = build.LAYOUT["pick_table"]
        assert r.GetMin()[0] > tx - 0.497 and r.GetMax()[0] < tx + 0.5
        assert r.GetMin()[1] > ty - 0.297 and r.GetMax()[1] < ty + 0.297


def test_line_bags_inside_their_containers(stage):
    for bag, box in (("Bag_InCarton", "/World/Cartons/Carton_Down_01/InteriorVolume"),
                     ("Bag_InBin", "/World/FloorBin/InteriorVolume")):
        r, v = _bbox(stage, f"/World/LineBags/{bag}"), _bbox(stage, box)
        assert all(v.GetMin()[k] - 1e-3 <= r.GetMin()[k] and r.GetMax()[k] <= v.GetMax()[k] + 1e-3 for k in range(3)), bag
