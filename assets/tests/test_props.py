"""Tests for the background props asset built by assets/props/build.py.

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest assets/tests/test_props.py
"""

from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

PROPS_DIR = Path(__file__).resolve().parents[1] / "props"
MM = 0.001
TOL = 20 * MM
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]

# name -> local bbox (X, Y, Z) in mm (see assets/props/README.md for sources)
PROPS = {
    "CrateStack": (400, 600, 5 * 224),
    "PalletStack": (1000, 1200, 144 + 4 * 224),
    "WasteBin": (580, 730, 1070),
    "HandwashStation": (400, 500, 1100),
    "LightCurtainPost": (200, 200, 1400),
}
# Room inner faces (assets/room) and the cell keep-out (conveyor, rack, robot).
ROOM_INNER = ((-4.4, 5.4), (-4.9, 4.9))
KEEP_OUT = ((-1.8, 2.0), (-2.8, 2.8))


@pytest.fixture(scope="module")
def stage():
    path = PROPS_DIR / "props.usda"
    assert path.is_file(), f"missing {path}"
    return Usd.Stage.Open(str(path))


def _local_range(prim, purposes=VISUAL):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeUntransformedBound(prim).ComputeAlignedRange()


def _world_range(prim, purposes=VISUAL):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def test_layers_units_and_default_prim(stage):
    root = stage.GetDefaultPrim()
    assert root and root.GetName() == "Props" and root.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(root).GetKind() == "assembly"
    for name in ("props.usda", "geo.usda", "physics.usda"):
        lst = Usd.Stage.Open(str(PROPS_DIR / name))
        assert UsdGeom.GetStageMetersPerUnit(lst) == 1.0, name
        assert UsdGeom.GetStageUpAxis(lst) == UsdGeom.Tokens.z, name
    spec = stage.GetRootLayer().GetPrimAtPath("/Props")
    assert [r.assetPath for r in spec.referenceList.GetAddedOrExplicitItems()] == ["./geo.usda"]
    assert [p.assetPath for p in spec.payloadList.GetAddedOrExplicitItems()] == ["./physics.usda"]
    assert (PROPS_DIR / "README.md").is_file()


def test_physics_layer_only_overs():
    layer = Sdf.Layer.FindOrOpen(str(PROPS_DIR / "physics.usda"))
    geo = Usd.Stage.Open(str(PROPS_DIR / "geo.usda"))
    bad = []

    def visit(path):
        if not path.IsPrimPath():
            return
        spec = layer.GetPrimAtPath(path)
        if spec.specifier == Sdf.SpecifierDef and "PhysicsMaterials" not in str(path):
            bad.append(str(path))
        if spec.specifier == Sdf.SpecifierOver and not geo.GetPrimAtPath(path):
            bad.append(f"dangling {path}")

    layer.Traverse("/", visit)
    assert not bad, bad


@pytest.mark.parametrize("name", list(PROPS))
def test_prop_xform_movable(stage, name):
    prim = stage.GetPrimAtPath(f"/Props/{name}")
    assert prim and prim.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(prim).GetKind() == "component"
    ops = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    assert ops == ["xformOp:translate", "xformOp:rotateZ"]


@pytest.mark.parametrize("name,size", list(PROPS.items()))
def test_prop_bbox(stage, name, size):
    rng = _local_range(stage.GetPrimAtPath(f"/Props/{name}"))
    for axis, want in enumerate(size):
        assert abs(rng.GetSize()[axis] - want * MM) <= TOL, (axis, rng.GetSize()[axis], want)
    # origin = centre of footprint on the floor
    assert abs(rng.GetMin()[2]) <= 1 * MM
    assert abs(rng.GetMidpoint()[0]) <= TOL and abs(rng.GetMidpoint()[1]) <= TOL


def test_pallet_and_crate_sizes(stage):
    pallet = _local_range(stage.GetPrimAtPath("/Props/PalletStack/Geo/Pallet"))
    for axis, want in enumerate((1000, 1200, 144)):
        assert abs(pallet.GetSize()[axis] - want * MM) <= TOL, axis
    crates = [p for p in stage.GetPrimAtPath("/Props/CrateStack/Geo").GetChildren()
              if p.GetName().startswith("Crate_")]
    assert len(crates) == 5
    for c in crates:
        r = _local_range(c)
        for axis, want in enumerate((400, 600, 224)):
            assert abs(r.GetSize()[axis] - want * MM) <= TOL, (c.GetName(), axis)
    post = stage.GetPrimAtPath("/Props/LightCurtainPost/Geo/SensorStrip")
    assert post and post.IsA(UsdGeom.Mesh)


@pytest.mark.parametrize("name", list(PROPS))
def test_static_colliders(stage, name):
    prim = stage.GetPrimAtPath(f"/Props/{name}")
    cols = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert cols, name
    for c in cols:
        assert UsdGeom.Imageable(c).ComputePurpose() == UsdGeom.Tokens.guide
        mat, _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial("physics")
        assert mat, c.GetPath()
    assert not [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    # colliders envelope the visual within tolerance
    vis = _local_range(prim)
    col = _local_range(prim, GUIDE)
    for axis in range(3):
        assert abs(col.GetSize()[axis] - vis.GetSize()[axis]) <= 0.1, (name, axis)


@pytest.mark.parametrize("name", list(PROPS))
def test_default_placement_in_room_and_clear_of_cell(stage, name):
    r = _world_range(stage.GetPrimAtPath(f"/Props/{name}"))
    lo, hi = r.GetMin(), r.GetMax()
    assert ROOM_INNER[0][0] <= lo[0] and hi[0] <= ROOM_INNER[0][1], name
    assert ROOM_INNER[1][0] <= lo[1] and hi[1] <= ROOM_INNER[1][1], name
    overlap_x = lo[0] < KEEP_OUT[0][1] and hi[0] > KEEP_OUT[0][0]
    overlap_y = lo[1] < KEEP_OUT[1][1] and hi[1] > KEEP_OUT[1][0]
    assert not (overlap_x and overlap_y), name


def test_triangle_budget(stage):
    tris = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            tris += sum(c - 2 for c in UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
    assert 0 < tris < 30_000, tris
