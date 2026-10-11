"""Tests for assets/sort_rack (angled gravity-flow tote rack behind the robot).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest assets/tests/test_sort_rack.py
"""
import math
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_DIR = Path(__file__).resolve().parents[1] / "sort_rack"
ROOT = ASSET_DIR / "sort_rack.usda"
TOL = 0.020

DEPTH_X = 0.65
WIDTH_Y = 0.55  # Layout v2: 1 lane column (was 1.35 / 2 lanes; the second column was behind the robot, unreachable)
LIP_HEIGHTS = [0.75, 1.30]
TILT_DEG = 20.0
LEVELS = [1, 2]
LANES = [1]


@pytest.fixture(scope="module")
def stage():
    assert ROOT.exists(), f"missing {ROOT} (run assets/sort_rack/build.py)"
    s = Usd.Stage.Open(str(ROOT))
    assert s is not None
    return s


def _bbox(prim, purposes=None):
    purposes = purposes or [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _world(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _slope(prim):
    """Return (tilt_deg, local +X axis in world) of a prim's frame."""
    m = _world(prim)
    ax = m.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
    tilt = math.degrees(math.asin(-ax[2]))  # positive = +X points downhill
    return tilt, ax


def test_layer_metadata(stage):
    dp = stage.GetDefaultPrim()
    assert dp and dp.GetPath() == "/SortRack"
    assert dp.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(dp).GetKind() == "component"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    for name in ("geo.usda", "physics.usda"):
        lyr = Usd.Stage.Open(str(ASSET_DIR / name))
        assert UsdGeom.GetStageMetersPerUnit(lyr) == 1.0
        assert UsdGeom.GetStageUpAxis(lyr) == UsdGeom.Tokens.z


def test_files_present():
    for name in ("build.py", "sort_rack.usda", "geo.usda", "physics.usda", "README.md"):
        assert (ASSET_DIR / name).exists(), name


def test_overall_bbox(stage):
    r = _bbox(stage.GetDefaultPrim())
    mn, mx = r.GetMin(), r.GetMax()
    assert abs((mx[0] - mn[0]) - DEPTH_X) <= TOL
    assert abs((mx[1] - mn[1]) - WIDTH_Y) <= TOL
    assert abs((mx[0] + mn[0]) / 2) <= TOL
    assert abs((mx[1] + mn[1]) / 2) <= TOL
    assert abs(mn[2]) <= 1e-6
    # taller than the upper lip, not absurdly tall
    assert LIP_HEIGHTS[-1] < mx[2] < 2.0


@pytest.mark.parametrize("level,lip", list(zip(LEVELS, LIP_HEIGHTS)))
def test_front_lip_heights(stage, level, lip):
    bar = stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/StopBar")
    assert bar.IsValid()
    r = _bbox(bar)
    assert abs(r.GetMax()[2] - lip) <= TOL
    # stop bar is at the front (+X, robot side)
    assert r.GetMax()[0] > DEPTH_X / 2 - 0.05


@pytest.mark.parametrize("level", LEVELS)
def test_shelf_tilt(stage, level):
    deck = stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/DeckCollider")
    assert deck.IsValid() and deck.IsA(UsdGeom.Cube)
    tilt, ax = _slope(deck)
    assert abs(tilt - TILT_DEG) <= 0.5, tilt
    assert ax[0] > 0, "slopes down toward +X (the robot)"
    # deck surface at the front edge sits below the stop-bar top
    r = _bbox(deck, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
    lip = LIP_HEIGHTS[level - 1]
    assert r.GetMin()[2] < lip
    # rear end higher than the front end
    assert r.GetMax()[2] > lip


@pytest.mark.parametrize("level", LEVELS)
def test_rollers_present(stage, level):
    for lane in LANES:
        rollers = stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/Lane{lane}Rollers")
        assert rollers.IsValid()
        kids = [c for c in rollers.GetChildren() if c.IsA(UsdGeom.Cylinder)]
        assert len(kids) >= 8
        for k in kids:
            assert not k.HasAPI(UsdPhysics.CollisionAPI), "rollers are visual only"


def test_lane_prims(stage):
    slots = []
    for level in LEVELS:
        for lane in LANES:
            base = f"/SortRack/Lanes/L{level}_{lane}"
            label = stage.GetPrimAtPath(f"{base}/Label")
            assert label.IsValid() and label.IsA(UsdGeom.Gprim)
            col = UsdGeom.Gprim(label).GetDisplayColorAttr().Get()
            assert col and len(col) == 1
            slot = stage.GetPrimAtPath(f"{base}/ToteSlot")
            assert slot.IsValid() and slot.IsA(UsdGeom.Xform)
            slots.append(slot)
            # label at the front face
            assert _bbox(label).GetMax()[0] > DEPTH_X / 2 - 0.02
    assert len(slots) == len(LEVELS) * len(LANES)
    colours = {tuple(UsdGeom.Gprim(stage.GetPrimAtPath(f"/SortRack/Lanes/L{lv}_{ln}/Label"))
                     .GetDisplayColorAttr().Get()[0]) for lv in LEVELS for ln in LANES}
    assert len(colours) == len(LEVELS) * len(LANES)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("lane", LANES)
def test_tote_slot_pose(stage, level, lane):
    slot = stage.GetPrimAtPath(f"/SortRack/Lanes/L{level}_{lane}/ToteSlot")
    tilt, ax = _slope(slot)
    assert abs(tilt - TILT_DEG) <= 0.5
    assert ax[0] > 0
    c = _world(slot).ExtractTranslation()
    # tote (600 along slope x 400 x 325) fits inside the rack footprint
    assert abs(c[0]) < DEPTH_X / 2
    assert abs(c[1]) < WIDTH_Y / 2 - 0.2
    if len(LANES) > 1:
        assert (c[1] < 0) == (lane == 1), "lane 1 is on -Y"
    # tote centre is 325/2 above its deck along the deck normal, above the lip height-ish
    lip = LIP_HEIGHTS[level - 1]
    assert lip - 0.05 < c[2] < lip + 0.35
    # tote bottom-front corner rests behind the stop bar
    front_bottom = _world(slot).Transform(Gf.Vec3d(0.3, 0, -0.1625))
    bar = _bbox(stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/StopBar"))
    assert front_bottom[0] <= bar.GetMin()[0] + 0.002


@pytest.mark.parametrize("level", LEVELS)
def test_shelf_colliders(stage, level):
    deck = stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/DeckCollider")
    assert deck.HasAPI(UsdPhysics.CollisionAPI)
    assert not deck.HasAPI(UsdPhysics.RigidBodyAPI)
    mat, _ = UsdShade.MaterialBindingAPI(deck).ComputeBoundMaterial("physics")
    assert mat
    m = UsdPhysics.MaterialAPI(mat.GetPrim())
    assert abs(m.GetStaticFrictionAttr().Get() - 0.1) < 1e-6
    assert abs(m.GetDynamicFrictionAttr().Get() - 0.1) < 1e-6
    bar = stage.GetPrimAtPath(f"/SortRack/Shelves/Level{level}/StopBar")
    assert bar.IsA(UsdGeom.Cube)
    assert bar.HasAPI(UsdPhysics.CollisionAPI)
    assert not bar.HasAPI(UsdPhysics.RigidBodyAPI)


def test_frame_colliders(stage):
    frame = stage.GetPrimAtPath("/SortRack/Frame")
    uprights = [c for c in frame.GetChildren() if c.GetName().startswith("Upright")]
    assert len(uprights) >= 4
    for u in uprights:
        assert u.HasAPI(UsdPhysics.CollisionAPI)


def test_no_rigid_bodies(stage):
    for p in stage.Traverse():
        assert not p.HasAPI(UsdPhysics.RigidBodyAPI)


def test_physics_is_payload(stage):
    s = Usd.Stage.Open(str(ROOT), Usd.Stage.LoadNone)
    deck = s.GetPrimAtPath("/SortRack/Shelves/Level1/DeckCollider")
    assert deck.IsValid()
    assert not deck.HasAPI(UsdPhysics.CollisionAPI)
