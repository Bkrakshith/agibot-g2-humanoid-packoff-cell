"""Tests for assets/conveyor (powered roller conveyor, crates flow toward -Y).

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_conveyor.py
"""
from pathlib import Path

import pytest
from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics, UsdShade

ASSET_DIR = Path(__file__).resolve().parents[1] / "conveyor"
ROOT = ASSET_DIR / "conveyor.usda"

TOL = 0.020          # spec acceptance tolerance (+-20 mm)
ROLLER_TOP = 0.82
GUIDE_TOP = 0.86
LENGTH = 3.6


@pytest.fixture(scope="module")
def stage():
    assert ROOT.exists(), f"missing {ROOT} (run assets/conveyor/build.py)"
    st = Usd.Stage.Open(str(ROOT), Usd.Stage.LoadAll)
    assert st is not None
    return st


def world_range(prim):
    """World-space AABB of a single gprim from its extent (ignores visibility)."""
    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
    assert extent is not None, f"{prim.GetPath()} has no extent"
    box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1])),
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return box.ComputeAlignedRange()


def gprims_under(stage, path):
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.IsA(UsdGeom.Gprim)]


# --- layer / composition ---------------------------------------------------------------

def test_root_layer_metadata(stage):
    dp = stage.GetDefaultPrim()
    assert dp and dp.GetPath() == "/Conveyor"
    assert dp.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(dp).GetKind() == Kind.Tokens.component
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z


@pytest.mark.parametrize("name", ["geo.usda", "physics.usda"])
def test_sublayers_have_units(name):
    st = Usd.Stage.Open(str(ASSET_DIR / name))
    assert UsdGeom.GetStageMetersPerUnit(st) == 1.0
    assert UsdGeom.GetStageUpAxis(st) == UsdGeom.Tokens.z


def test_composition_reference_and_payload(stage):
    spec = stage.GetRootLayer().GetPrimAtPath("/Conveyor")
    refs = [r.assetPath for r in spec.referenceList.GetAddedOrExplicitItems()]
    pls = [p.assetPath for p in spec.payloadList.GetAddedOrExplicitItems()]
    assert refs == ["./geo.usda"]
    assert pls == ["./physics.usda"]


def test_physics_layer_only_overs_existing_geometry():
    layer = Usd.Stage.Open(str(ASSET_DIR / "physics.usda")).GetRootLayer()
    geo = Usd.Stage.Open(str(ASSET_DIR / "geo.usda"))
    from pxr import Sdf
    bad = []

    def visit(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None or path == Sdf.Path.absoluteRootPath:
            return
        # new physics-only prims (joints, physics materials) may be defs
        if spec.specifier == Sdf.SpecifierDef and spec.typeName not in (
                "PhysicsPrismaticJoint", "Material", "Scope"):
            bad.append(str(path))
        if spec.specifier == Sdf.SpecifierOver and not geo.GetPrimAtPath(path):
            bad.append(f"dangling over {path}")
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    assert not bad, bad


def test_readme_exists():
    assert (ASSET_DIR / "README.md").exists()
    assert (ASSET_DIR / "build.py").exists()


# --- overall dimensions ------------------------------------------------------------------

def test_overall_bbox(stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath("/Conveyor")).ComputeAlignedRange()
    size = r.GetSize()
    assert 0.55 - TOL <= size[0] <= 0.65 + TOL, size
    assert 0.55 <= size[0] <= 0.65, size
    assert abs(size[1] - LENGTH) <= TOL, size
    assert abs(r.GetMin()[2] - 0.0) <= TOL, r
    assert abs(r.GetMax()[2] - GUIDE_TOP) <= TOL, r
    # origin = centre of footprint
    c = r.GetMidpoint()
    assert abs(c[1]) <= TOL and abs(c[0]) <= TOL, c


# --- conveyance surface -----------------------------------------------------------------

def test_belt_surface(stage):
    belt = stage.GetPrimAtPath("/Conveyor/BeltSurface")
    assert belt.IsValid()
    assert belt.HasAPI(UsdPhysics.CollisionAPI)
    assert not belt.HasAPI(UsdPhysics.RigidBodyAPI), "must be static"
    r = world_range(belt)
    assert abs(r.GetMax()[2] - ROLLER_TOP) <= 0.002, r
    assert r.GetSize()[2] <= 0.02
    assert abs(r.GetSize()[1] - LENGTH) <= TOL
    assert abs(r.GetSize()[0] - 0.45) <= 0.01
    # surface-velocity API is applied in Stage 3, not now
    assert "PhysxSurfaceVelocityAPI" not in belt.GetAppliedSchemas()


def test_rollers_visual_only(stage):
    rollers = gprims_under(stage, "/Conveyor/Rollers")
    assert len(rollers) == 48
    ys = []
    for p in rollers:
        assert not p.HasAPI(UsdPhysics.CollisionAPI), p.GetPath()
        r = world_range(p)
        assert abs(r.GetMax()[2] - ROLLER_TOP) <= 0.002
        assert abs(r.GetSize()[2] - 0.050) <= 0.001        # diameter 50 mm
        assert abs(r.GetSize()[0] - 0.45) <= 0.01            # axis along X between frames
        ys.append(r.GetMidpoint()[1])
    ys.sort()
    pitches = [b - a for a, b in zip(ys, ys[1:])]
    assert all(abs(p - 0.075) < 1e-6 for p in pitches)


# --- frame and guides -------------------------------------------------------------------

def test_guides_inner_width_and_height(stage):
    near = gprims_under(stage, "/Conveyor/Guides/Near")   # -X, robot side
    far = gprims_under(stage, "/Conveyor/Guides/Far")     # +X
    assert near and far
    rails_far = [world_range(p) for p in far if "Rail" in p.GetName()]
    rails_near = [world_range(p) for p in near if "Rail" in p.GetName()]
    assert rails_far and rails_near
    inner = min(r.GetMin()[0] for r in rails_far) - max(r.GetMax()[0] for r in rails_near)
    assert abs(inner - 0.45) <= 0.01, inner
    for r in rails_far + rails_near:
        assert abs(r.GetMax()[2] - GUIDE_TOP) <= 0.002
    for p in near + far:
        assert p.HasAPI(UsdPhysics.CollisionAPI), p.GetPath()


def test_frame_static_colliders(stage):
    parts = gprims_under(stage, "/Conveyor/Frame")
    assert len(parts) > 10
    for p in parts:
        assert p.HasAPI(UsdPhysics.CollisionAPI), p.GetPath()
        assert not p.HasAPI(UsdPhysics.RigidBodyAPI)
    names = {p.GetName() for p in parts}
    assert any(n.startswith("Leg") for n in names)
    assert any(n.startswith("Foot") for n in names)
    assert any(n.startswith("Cross") for n in names)


def test_frame_inner_width(stage):
    webs = [world_range(p) for p in gprims_under(stage, "/Conveyor/Frame") if "Web" in p.GetName()]
    assert len(webs) == 2
    inner = max(r.GetMin()[0] for r in webs) - min(r.GetMax()[0] for r in webs)
    assert abs(inner - 0.45) <= 0.002, inner


# --- stop gate --------------------------------------------------------------------------

def test_stop_gate_joint(stage):
    gate = stage.GetPrimAtPath("/Conveyor/StopGate")
    assert gate.IsValid()
    assert gate.HasAPI(UsdPhysics.RigidBodyAPI)
    assert gate.HasAPI(UsdPhysics.MassAPI)
    joints = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/Conveyor"))
              if p.IsA(UsdPhysics.PrismaticJoint)]
    assert len(joints) == 1
    j = UsdPhysics.PrismaticJoint(joints[0])
    assert j.GetAxisAttr().Get() == UsdPhysics.Tokens.z
    assert j.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert j.GetUpperLimitAttr().Get() == pytest.approx(0.04)
    assert j.GetBody1Rel().GetTargets() == ["/Conveyor/StopGate"]
    assert j.GetBody0Rel().GetTargets() == ["/Conveyor/Frame"]
    drive = UsdPhysics.DriveAPI.Get(joints[0], UsdPhysics.Tokens.linear)
    assert drive and joints[0].HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.linear)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(0.0)
    assert drive.GetStiffnessAttr().Get() > 0


def test_stop_gate_geometry(stage):
    bars = gprims_under(stage, "/Conveyor/StopGate")
    assert bars
    for p in bars:
        assert p.HasAPI(UsdPhysics.CollisionAPI)
    r = world_range(bars[0])
    for p in bars[1:]:
        r = Gf.Range3d.GetUnion(r, world_range(p))
    assert abs(r.GetMidpoint()[1]) <= 0.001           # at the pack position Y = 0
    assert r.GetMax()[2] < ROLLER_TOP                  # lowered: below crate
    assert r.GetMax()[2] + 0.04 >= ROLLER_TOP + 0.03   # raised: blocks crate
    # fits in the gap between the two rollers either side of Y = 0
    assert r.GetSize()[1] < 0.075 - 0.050


def test_stop_gate_filtered_from_belt(stage):
    gate = stage.GetPrimAtPath("/Conveyor/StopGate")
    fp = UsdPhysics.FilteredPairsAPI(gate)
    assert "/Conveyor/BeltSurface" in [str(t) for t in fp.GetFilteredPairsRel().GetTargets()]


# --- photo-eye --------------------------------------------------------------------------

def test_photo_eye(stage):
    pe = stage.GetPrimAtPath("/Conveyor/PhotoEye")
    assert pe.IsValid()
    beam = stage.GetPrimAtPath("/Conveyor/PhotoEye/BeamOrigin")
    assert beam.IsValid() and beam.IsA(UsdGeom.Xform)
    m = UsdGeom.Xformable(beam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    o = m.ExtractTranslation()
    # crates flow toward -Y, so upstream of the gate (Y = 0) is +Y
    assert o[1] == pytest.approx(0.35, abs=1e-6)
    assert ROLLER_TOP < o[2] < GUIDE_TOP
    d = m.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
    assert abs(abs(d[0]) - 1.0) < 1e-6, d          # beam across X
    assert "beam" in (beam.GetDocumentation() or "").lower()
    assert gprims_under(stage, "/Conveyor/PhotoEye")  # bracket/body geometry


# --- materials --------------------------------------------------------------------------

def test_stainless_preview_material(stage):
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Conveyor/Looks/BrushedStainless/PreviewSurface"))
    assert shader
    assert shader.GetInput("metallic").Get() == pytest.approx(1.0)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.35, abs=0.05)
    # rollers and frame bound to it
    roller = gprims_under(stage, "/Conveyor/Rollers")[0]
    mat, _ = UsdShade.MaterialBindingAPI(roller).ComputeBoundMaterial()
    assert mat and mat.GetPath() == "/Conveyor/Looks/BrushedStainless"


def test_physics_material_stainless(stage):
    belt = stage.GetPrimAtPath("/Conveyor/BeltSurface")
    mat, _ = UsdShade.MaterialBindingAPI(belt).ComputeBoundMaterial("physics")
    assert mat
    pm = UsdPhysics.MaterialAPI(mat.GetPrim())
    assert pm.GetStaticFrictionAttr().Get() == pytest.approx(0.3)
    assert pm.GetDynamicFrictionAttr().Get() == pytest.approx(0.3)
