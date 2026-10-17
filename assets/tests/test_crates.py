"""Tests for the three crate assets built by assets/crates/build.py.

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_crates.py
"""

from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

CRATES_DIR = Path(__file__).resolve().parents[1] / "crates"

MM = 0.001
OUTER_TOL = 2 * MM   # catalogue dims, tighter than the spec's 20 mm
INNER_TOL = 3 * MM
MAX_TRIANGLES = 40_000

# folder, defaultPrim, outer (X, Y, Z) mm, inner (X, Y, Z) mm, mass kg.
# Long side along Y. Inner dims of balearm_199 and inner X/Y of deep_325 are
# estimates (see the crates' README.md); everything else is catalogue data.
VARIANTS = {
    "stacking_224": dict(prim="CrateStacking224", outer=(400, 600, 224),
                         inner=(368, 568, 210), mass=1.72),
    "balearm_199": dict(prim="CrateBaleArm199", outer=(400, 600, 199),
                        inner=(366, 566, 185), mass=1.5),
    "deep_325": dict(prim="CrateDeep325", outer=(400, 600, 325),
                     inner=(366, 566, 284), mass=2.0),
}

PARAMS = [pytest.param(k, v, id=k) for k, v in VARIANTS.items()]


def _open(name):
    path = CRATES_DIR / name / f"{name}.usda"
    assert path.is_file(), f"missing {path}"
    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    return stage


def _range(prim, purposes):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _colliders(stage, root):
    return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]


def _by_name(prims):
    return {p.GetName(): p for p in prims}


GUIDE = [UsdGeom.Tokens.guide]
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]


@pytest.mark.parametrize("name,spec", PARAMS)
def test_layers_units_and_default_prim(name, spec):
    stage = _open(name)
    root = stage.GetDefaultPrim()
    assert root and root.GetName() == spec["prim"]
    assert root.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(root).GetKind() == "component"
    for layer_name in (f"{name}.usda", "geo.usda", "physics.usda"):
        layer_path = CRATES_DIR / name / layer_name
        assert layer_path.is_file()
        lstage = Usd.Stage.Open(str(layer_path))
        assert UsdGeom.GetStageMetersPerUnit(lstage) == 1.0, layer_name
        assert UsdGeom.GetStageUpAxis(lstage) == UsdGeom.Tokens.z, layer_name
    # geometry is referenced, physics is a payload
    root_layer = stage.GetRootLayer()
    spec_prim = root_layer.GetPrimAtPath(root.GetPath())
    refs = [r.assetPath for r in spec_prim.referenceList.GetAddedOrExplicitItems()]
    pays = [p.assetPath for p in spec_prim.payloadList.GetAddedOrExplicitItems()]
    assert refs == ["./geo.usda"]
    assert pays == ["./physics.usda"]


def test_physics_layer_has_only_overs_for_existing_prims():
    for name in VARIANTS:
        layer = Sdf.Layer.FindOrOpen(str(CRATES_DIR / name / "physics.usda"))
        geo = Usd.Stage.Open(str(CRATES_DIR / name / "geo.usda"))
        bad = []

        def visit(spec):
            path = spec.path
            if spec.specifier == Sdf.SpecifierDef:
                if "PhysicsMaterials" not in str(path):
                    bad.append(str(path))
            elif not geo.GetPrimAtPath(path):
                bad.append(f"dangling over {path}")
            for child in spec.nameChildren:
                visit(child)

        for top in layer.rootPrims:
            visit(top)
        assert not bad, f"{name}: {bad}"


@pytest.mark.parametrize("name,spec", PARAMS)
def test_outer_bbox_matches_catalogue(name, spec):
    stage = _open(name)
    rng = _range(stage.GetDefaultPrim(), VISUAL)
    size = rng.GetSize()
    for axis, want in enumerate(spec["outer"]):
        assert abs(size[axis] - want * MM) <= OUTER_TOL, (axis, size[axis], want)
    # origin = centre of base footprint
    lo, hi = rng.GetMin(), rng.GetMax()
    assert abs(lo[2]) <= 1e-6
    assert abs((lo[0] + hi[0]) / 2) <= 1e-6
    assert abs((lo[1] + hi[1]) / 2) <= 1e-6


@pytest.mark.parametrize("name,spec", PARAMS)
def test_compound_box_collider(name, spec):
    stage = _open(name)
    root = stage.GetDefaultPrim()
    cols = _colliders(stage, root)
    assert len(cols) >= 5
    named = _by_name(cols)
    for n in ("Base", "WallNegX", "WallPosX", "WallNegY", "WallPosY"):
        assert n in named, f"{name}: collider {n} missing"
    for c in cols:
        # boxes, not hulls/meshes
        assert c.IsA(UsdGeom.Cube), c.GetPath()
        assert not c.HasAPI(UsdPhysics.MeshCollisionAPI)
        # contact offset 3 mm (raw PhysX attribute; PhysxSchema is not in usd-core)
        # (GetAppliedSchemas() hides schemas usd-core doesn't know, so read the list op)
        assert "PhysxCollisionAPI" in c.GetMetadata("apiSchemas").GetAppliedItems()
        assert c.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.003)
        # HDPE friction 0.5 via a physics-purpose material binding
        mat, _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial("physics")
        assert mat, c.GetPath()
        pm = UsdPhysics.MaterialAPI(mat.GetPrim())
        assert pm.GetStaticFrictionAttr().Get() == pytest.approx(0.5)
        assert pm.GetDynamicFrictionAttr().Get() == pytest.approx(0.5)
        # colliders are not visual geometry
        assert UsdGeom.Imageable(c).ComputePurpose() == UsdGeom.Tokens.guide

    r = {n: _range(named[n], GUIDE) for n in ("Base", "WallNegX", "WallPosX", "WallNegY", "WallPosY")}
    inner_x = r["WallPosX"].GetMin()[0] - r["WallNegX"].GetMax()[0]
    inner_y = r["WallPosY"].GetMin()[1] - r["WallNegY"].GetMax()[1]
    inner_z = r["WallPosX"].GetMax()[2] - r["Base"].GetMax()[2]
    want = [v * MM for v in spec["inner"]]
    assert abs(inner_x - want[0]) <= INNER_TOL, inner_x
    assert abs(inner_y - want[1]) <= INNER_TOL, inner_y
    assert abs(inner_z - want[2]) <= INNER_TOL, inner_z
    for w, axis in (("WallNegX", 0), ("WallPosX", 0), ("WallNegY", 1), ("WallPosY", 1)):
        t = r[w].GetSize()[axis]
        assert 3 * MM <= t <= 5 * MM, (w, t)
    # collider envelope never exceeds the outer catalogue box
    whole = Gf.Range3d()
    for c in cols:
        whole.UnionWith(_range(c, GUIDE))
    for axis, o in enumerate(spec["outer"]):
        assert whole.GetSize()[axis] <= o * MM + OUTER_TOL


@pytest.mark.parametrize("name,spec", PARAMS)
def test_rigid_body_and_mass(name, spec):
    stage = _open(name)
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Get() in (True, None)
    assert not UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(spec["mass"])
    # no nested rigid bodies
    nested = [p for p in Usd.PrimRange(root) if p != root and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert not nested


@pytest.mark.parametrize("name,spec", PARAMS)
def test_interior_volume(name, spec):
    stage = _open(name)
    root = stage.GetDefaultPrim()
    iv = stage.GetPrimAtPath(root.GetPath().AppendChild("InteriorVolume"))
    assert iv and iv.IsA(UsdGeom.Gprim)
    assert not iv.HasAPI(UsdPhysics.CollisionAPI)
    img = UsdGeom.Imageable(iv)
    assert img.ComputeVisibility() == UsdGeom.Tokens.invisible
    rng = _range(iv, GUIDE + VISUAL)
    want = [v * MM for v in spec["inner"]]
    for axis in range(3):
        assert abs(rng.GetSize()[axis] - want[axis]) <= 1 * MM, axis
    # sits on the base collider's top face, centred
    base = _by_name(_colliders(stage, root))["Base"]
    assert abs(rng.GetMin()[2] - _range(base, GUIDE).GetMax()[2]) <= 1 * MM
    assert abs(rng.GetMidpoint()[0]) <= 1e-6 and abs(rng.GetMidpoint()[1]) <= 1e-6


@pytest.mark.parametrize("name,spec", PARAMS)
def test_triangle_budget_and_colour(name, spec):
    stage = _open(name)
    root = stage.GetDefaultPrim()
    tris = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            counts = UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []
            tris += sum(c - 2 for c in counts)
    assert 0 < tris < MAX_TRIANGLES, tris
    geo = stage.GetPrimAtPath(root.GetPath().AppendChild("Geo"))
    pv = UsdGeom.PrimvarsAPI(geo).GetPrimvar("displayColor")
    assert pv and pv.GetInterpolation() == UsdGeom.Tokens.constant
    r, g, b = pv.Get()[0]
    assert b > r and b > g, "default colour is blue"


def test_documented_parts():
    st = _open("stacking_224")
    for p in ("Geo/Base", "Geo/Walls", "Geo/Ribs", "Geo/Rim"):
        assert st.GetPrimAtPath(f"/CrateStacking224/{p}"), p
    ba = _open("balearm_199")
    arms = [ba.GetPrimAtPath(f"/CrateBaleArm199/Geo/BaleArm{s}") for s in ("NegY", "PosY")]
    assert all(arms)
    for a in arms:
        assert a.IsA(UsdGeom.Mesh)
        assert not a.HasAPI(UsdPhysics.CollisionAPI)
    for name in VARIANTS:
        assert (CRATES_DIR / name / "README.md").is_file()
