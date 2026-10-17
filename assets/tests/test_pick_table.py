"""Tests for assets/pick_table (infeed / accumulation table on the robot's left).

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_pick_table.py
"""
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_DIR = Path(__file__).resolve().parents[1] / "pick_table"
ROOT = ASSET_DIR / "pick_table.usda"
TOL = 0.020  # spec acceptance tolerance, metres

FOOTPRINT_X = 0.6
FOOTPRINT_Y = 1.0
TOP_Z = 0.87
LIP_H = 0.025


@pytest.fixture(scope="module")
def stage():
    assert ROOT.exists(), f"missing {ROOT} (run assets/pick_table/build.py)"
    s = Usd.Stage.Open(str(ROOT))
    assert s is not None
    return s


def _bbox(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def test_layer_metadata(stage):
    dp = stage.GetDefaultPrim()
    assert dp and dp.GetPath() == "/PickTable"
    assert dp.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(dp).GetKind() == "component"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    for name in ("geo.usda", "physics.usda"):
        lyr = Usd.Stage.Open(str(ASSET_DIR / name))
        assert UsdGeom.GetStageMetersPerUnit(lyr) == 1.0
        assert UsdGeom.GetStageUpAxis(lyr) == UsdGeom.Tokens.z


def test_files_present():
    for name in ("build.py", "pick_table.usda", "geo.usda", "physics.usda", "README.md"):
        assert (ASSET_DIR / name).exists(), name


def test_overall_bbox(stage):
    r = _bbox(stage.GetDefaultPrim())
    mn, mx = r.GetMin(), r.GetMax()
    assert abs((mx[0] - mn[0]) - FOOTPRINT_X) <= TOL
    assert abs((mx[1] - mn[1]) - FOOTPRINT_Y) <= TOL
    # origin = footprint centre on the floor
    assert abs((mx[0] + mn[0]) / 2) <= TOL
    assert abs((mx[1] + mn[1]) / 2) <= TOL
    assert abs(mn[2]) <= 1e-6
    assert abs(mx[2] - (TOP_Z + LIP_H)) <= TOL


def test_top_surface_height(stage):
    top = stage.GetPrimAtPath("/PickTable/Top")
    assert top.IsValid()
    r = _bbox(top)
    assert abs(r.GetMax()[2] - TOP_Z) <= 0.002
    assert abs((r.GetMax()[0] - r.GetMin()[0]) - FOOTPRINT_X) <= TOL
    assert abs((r.GetMax()[1] - r.GetMin()[1]) - FOOTPRINT_Y) <= TOL


@pytest.mark.parametrize("name", ["LipPosX", "LipNegX", "LipPosY"])
def test_lips(stage, name):
    lip = stage.GetPrimAtPath(f"/PickTable/Lips/{name}")
    assert lip.IsValid()
    r = _bbox(lip)
    assert abs(r.GetMax()[2] - (TOP_Z + LIP_H)) <= 0.002
    assert r.GetMin()[2] <= TOP_Z + 0.001


def test_open_side_faces_negative_y(stage):
    # no lip on the -Y edge (faces the robot when placed at (-0.05, +0.95))
    assert not stage.GetPrimAtPath("/PickTable/Lips/LipNegY").IsValid()
    lip_y = _bbox(stage.GetPrimAtPath("/PickTable/Lips/LipPosY"))
    assert lip_y.GetMin()[1] > 0.45
    for name in ("LipPosX", "LipNegX"):
        r = _bbox(stage.GetPrimAtPath(f"/PickTable/Lips/{name}"))
        assert abs(r.GetMax()[0]) > 0.28 or abs(r.GetMin()[0]) > 0.28


def test_legs_feet_brace(stage):
    for i in range(1, 5):
        assert stage.GetPrimAtPath(f"/PickTable/Frame/Leg{i}").IsValid()
        foot = stage.GetPrimAtPath(f"/PickTable/Frame/Foot{i}")
        assert foot.IsValid()
        assert abs(_bbox(foot).GetMin()[2]) <= 1e-6
    brace = stage.GetPrimAtPath("/PickTable/Frame/CrossBrace")
    assert brace.IsValid()
    assert _bbox(brace).GetMax()[2] < 0.4


def test_spawn_zone(stage):
    sz = stage.GetPrimAtPath("/PickTable/SpawnZone")
    assert sz.IsValid() and sz.IsA(UsdGeom.Xform)
    box = stage.GetPrimAtPath("/PickTable/SpawnZone/Volume")
    assert box.IsValid()
    assert UsdGeom.Imageable(box).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert not box.HasAPI(UsdPhysics.CollisionAPI)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.guide])
    r = cache.ComputeWorldBound(box).ComputeAlignedRange()
    size = r.GetSize()
    assert abs(size[0] - 0.5) <= 0.002
    assert abs(size[1] - 0.9) <= 0.002
    assert abs(size[2] - 0.15) <= 0.002
    assert abs(r.GetMin()[2] - (TOP_Z + 0.02)) <= 0.002


COLLIDERS = ["/PickTable/Top", "/PickTable/Lips/LipPosX", "/PickTable/Lips/LipNegX", "/PickTable/Lips/LipPosY"]


@pytest.mark.parametrize("path", COLLIDERS)
def test_static_colliders(stage, path):
    prim = stage.GetPrimAtPath(path)
    assert prim.IsA(UsdGeom.Cube), "colliders are analytic boxes"
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    assert mat, f"{path} has no physics material"
    mapi = UsdPhysics.MaterialAPI(mat.GetPrim())
    assert abs(mapi.GetStaticFrictionAttr().Get() - 0.3) < 1e-6
    assert abs(mapi.GetDynamicFrictionAttr().Get() - 0.3) < 1e-6


def test_no_rigid_bodies(stage):
    for p in stage.Traverse():
        assert not p.HasAPI(UsdPhysics.RigidBodyAPI)


def test_physics_is_payload(stage):
    s = Usd.Stage.Open(str(ROOT), Usd.Stage.LoadNone)
    top = s.GetPrimAtPath("/PickTable/Top")
    assert top.IsValid(), "geometry comes from the reference"
    assert not top.HasAPI(UsdPhysics.CollisionAPI), "physics must live in the payload"
