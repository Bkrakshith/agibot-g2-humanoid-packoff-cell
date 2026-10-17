"""Tests for assets/case_sealer (static top+bottom tape case sealer prop, flow toward -Y).

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_case_sealer.py
"""

import re
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from _brands import BRANDS

DIR = Path(__file__).resolve().parents[1] / "case_sealer"
ROOT = DIR / "case_sealer.usda"
NAME = "CaseSealer"
MM = 0.001
TOL = 20 * MM
BED = 0.82
BED_TOL = 5 * MM
SIZE = (1.0, 1.6, 1.7)          # X, Y (along flow), Z
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]


@pytest.fixture(scope="module")
def stage():
    assert ROOT.is_file(), f"missing {ROOT} (run assets/case_sealer/build.py)"
    st = Usd.Stage.Open(str(ROOT), Usd.Stage.LoadAll)
    assert st is not None
    return st


def _range(prim, purposes=VISUAL):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _colliders(prim):
    return [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]


def _pos(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()


def _get(stage, rel):
    p = stage.GetPrimAtPath(f"/{NAME}/{rel}")
    assert p and p.IsValid(), f"missing /{NAME}/{rel}"
    return p


def test_layers(stage):
    root = stage.GetDefaultPrim()
    assert root.GetName() == NAME and root.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(root).GetKind() == "component"
    for ln in ("case_sealer.usda", "geo.usda", "physics.usda"):
        ls = Usd.Stage.Open(str(DIR / ln))
        assert UsdGeom.GetStageMetersPerUnit(ls) == 1.0 and UsdGeom.GetStageUpAxis(ls) == "Z"
    ps = stage.GetRootLayer().GetPrimAtPath(root.GetPath())
    assert [r.assetPath for r in ps.referenceList.GetAddedOrExplicitItems()] == ["./geo.usda"]
    assert [p.assetPath for p in ps.payloadList.GetAddedOrExplicitItems()] == ["./physics.usda"]
    assert (DIR / "README.md").is_file() and (DIR / "build.py").is_file()


def test_physics_layer_only_overs():
    layer = Sdf.Layer.FindOrOpen(str(DIR / "physics.usda"))
    geo = Usd.Stage.Open(str(DIR / "geo.usda"))
    bad = []

    def visit(spec):
        if spec.specifier == Sdf.SpecifierDef and "PhysicsMaterials" not in str(spec.path):
            bad.append(str(spec.path))
        elif spec.specifier == Sdf.SpecifierOver and not geo.GetPrimAtPath(spec.path):
            bad.append(f"dangling {spec.path}")
        for c in spec.nameChildren:
            visit(c)

    for top in layer.rootPrims:
        visit(top)
    assert not bad, bad


def test_bbox_and_origin(stage):
    rng = _range(stage.GetDefaultPrim())
    for ax, want in enumerate(SIZE):
        if want is not None:
            assert abs(rng.GetSize()[ax] - want) <= TOL, (ax, rng.GetSize()[ax])
    assert abs(rng.GetMin()[2]) <= 1e-4
    assert abs(rng.GetMidpoint()[0]) <= TOL and abs(rng.GetMidpoint()[1]) <= TOL


def test_no_rigid_bodies(stage):
    assert not [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _check_colliders(prim, n_min=1):
    cols = _colliders(prim)
    assert len(cols) >= n_min, (prim.GetPath(), len(cols))
    for c in cols:
        assert UsdGeom.Imageable(c).ComputePurpose() == "guide", c.GetPath()
        mat, _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial("physics")
        assert mat, c.GetPath()
    return cols


def test_beds_at_conveyor_height(stage):
    for grp in ("Infeed", "Outfeed"):
        bed = _get(stage, f"{grp}/Colliders/Bed")
        assert bed.HasAPI(UsdPhysics.CollisionAPI)
        assert abs(_range(bed, GUIDE).GetMax()[2] - BED) <= BED_TOL
        assert abs(_range(_get(stage, f"{grp}/Rollers")).GetMax()[2] - BED) <= BED_TOL
    root_r = _range(stage.GetDefaultPrim())
    inf = _range(_get(stage, "Infeed/Colliders/Bed"), GUIDE)
    out = _range(_get(stage, "Outfeed/Colliders/Bed"), GUIDE)
    assert abs(inf.GetMax()[1] - root_r.GetMax()[1]) <= 2 * MM, "infeed at the +Y edge"
    assert abs(out.GetMin()[1] - root_r.GetMin()[1]) <= 2 * MM, "outfeed at the -Y edge"


def test_entry_exit_points(stage):
    _get(stage, "Infeed")
    e = _pos(_get(stage, "Infeed/EntryPoint"))
    x = _pos(_get(stage, "Outfeed/ExitPoint"))
    assert _get(stage, "Outfeed/ExitPoint").IsA(UsdGeom.Xform)
    for p in (e, x):
        assert abs(p[2] - BED) <= BED_TOL and abs(p[0]) <= 1 * MM
    assert e[1] > 0.7 and x[1] < -0.7


def test_tunnel_fits_400x250_carton(stage):
    tun = {c.GetName(): _range(c, GUIDE) for c in _check_colliders(_get(stage, "Tunnel"), 4)}
    assert {"BeltNegX", "BeltPosX", "HeadPlate"} <= set(tun), sorted(tun)
    width = tun["BeltPosX"].GetMin()[0] - tun["BeltNegX"].GetMax()[0]
    assert 0.400 <= width <= 0.440, width
    height = tun["HeadPlate"].GetMin()[2] - BED
    assert 0.250 <= height <= 0.280, height
    # the opening is clear: no tunnel collider intrudes into the 400 x 250 carton envelope
    for n, r in tun.items():
        inside_x = r.GetMax()[0] > -0.2 and r.GetMin()[0] < 0.2
        inside_z = r.GetMax()[2] > BED + 0.001 and r.GetMin()[2] < BED + 0.25
        assert not (inside_x and inside_z), n


def test_tape_roll_and_stack_light(stage):
    assert _get(stage, "TopHead/TapeRoll").IsA(UsdGeom.Mesh)
    _get(stage, "BottomHead")
    zs = [_range(_get(stage, f"StackLight/{n}")).GetMidpoint()[2] for n in ("Green", "Amber", "Red")]
    assert zs[0] < zs[1] < zs[2]


def test_no_lights_triangles_brands(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight) or p.HasAPI(UsdLux.LightAPI)]
    tris = sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    assert 0 < tris < 60_000, tris
    pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b", re.I)
    files = list(DIR.glob("*.usda")) + list(DIR.glob("*.md")) + list(DIR.glob("*.py"))
    hits = [(f.name, m.group(0)) for f in files for m in pat.finditer(f.read_text())]
    assert not hits, hits
