"""Tests for assets/case_erector (static automatic carton erector prop, outfeed -> conveyor +Y end).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest assets/tests/test_case_erector.py
"""

import re
from pathlib import Path

import pytest
from pxr import Pcp, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from _brands import BRANDS

DIR = Path(__file__).resolve().parents[1] / "case_erector"
ROOT = DIR / "case_erector.usda"
NAME = "CaseErector"
MM = 0.001
TOL = 20 * MM
BED = 0.82
BED_TOL = 5 * MM
SIZE = (1.4, 2.2, 1.9)          # X, Y (along flow), Z
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]


@pytest.fixture(scope="module")
def stage():
    assert ROOT.is_file(), f"missing {ROOT} (run assets/case_erector/build.py)"
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
    for ln in ("case_erector.usda", "geo.usda", "physics.usda"):
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
        assert abs(rng.GetSize()[ax] - want) <= TOL, (ax, rng.GetSize()[ax])
    assert abs(rng.GetMin()[2]) <= 1e-4
    assert abs(rng.GetMidpoint()[0]) <= TOL and abs(rng.GetMidpoint()[1]) <= TOL


def test_outfeed_bed_and_spawn_point(stage):
    bed = _get(stage, "Outfeed/Colliders/Bed")
    assert bed.HasAPI(UsdPhysics.CollisionAPI)
    r = _range(bed, GUIDE)
    assert abs(r.GetMax()[2] - BED) <= BED_TOL
    root_r = _range(stage.GetDefaultPrim())
    assert abs(r.GetMin()[1] - root_r.GetMin()[1]) <= 2 * MM, "bed must end at the -Y edge"
    assert abs(r.GetMidpoint()[0]) <= 1 * MM, "outfeed centred on X = 0"
    sp = _get(stage, "Outfeed/SpawnPoint")
    assert sp.IsA(UsdGeom.Xform)
    p = _pos(sp)
    assert abs(p[2] - BED) <= BED_TOL and abs(p[0]) <= 1 * MM
    assert r.GetMin()[1] <= p[1] - 0.3 and p[1] + 0.3 <= r.GetMax()[1], "600 mm carton fits"
    # guide rails let a 400 mm wide carton through
    g = {c.GetName(): _range(c, GUIDE) for c in _colliders(_get(stage, "Outfeed"))}
    assert {"GuideNegX", "GuidePosX"} <= set(g)
    gap = g["GuidePosX"].GetMin()[0] - g["GuideNegX"].GetMax()[0]
    assert 0.42 <= gap <= 0.47, gap
    # rollers (visual) top at the bed height
    rr = _range(_get(stage, "Outfeed/Rollers"))
    assert abs(rr.GetMax()[2] - BED) <= BED_TOL


def test_static_colliders_on_guards(stage):
    cols = _colliders(_get(stage, "Guards"))
    assert len(cols) >= 4
    for c in cols:
        assert c.IsA(UsdGeom.Cube) and UsdGeom.Imageable(c).ComputePurpose() == "guide"
        mat, _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial("physics")
        assert mat, c.GetPath()
    assert not [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def test_magazine_holds_flat_stack(stage):
    ms = _get(stage, "Magazine/BlankStack")
    arcs = [a.GetTargetLayer().identifier for a in Usd.PrimCompositionQuery(ms).GetCompositionArcs()
            if a.GetArcType() == Pcp.ArcTypeReference]
    assert any("carton_flat_stack" in a for a in arcs), arcs
    blanks = [p for p in Usd.PrimRange(ms) if p.GetName().startswith("Blank_") and p.IsA(UsdGeom.Mesh)]
    assert len(blanks) == 20
    assert Usd.ModelAPI(ms).GetKind() != "component"     # no component inside a component


def test_control_panel_and_stack_light(stage):
    _get(stage, "ControlPanel/HMI")
    for lamp in ("Green", "Amber", "Red"):
        p = _get(stage, f"StackLight/{lamp}")
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat
    zs = [_range(_get(stage, f"StackLight/{n}")).GetMidpoint()[2] for n in ("Green", "Amber", "Red")]
    assert zs[0] < zs[1] < zs[2], "red on top"


def test_polycarbonate_guards_are_transparent(stage):
    panels = _get(stage, "Guards/Panels")
    mat, _ = UsdShade.MaterialBindingAPI(panels).ComputeBoundMaterial()
    sh = [UsdShade.Shader(p) for p in Usd.PrimRange(mat.GetPrim())
          if UsdShade.Shader(p) and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"][0]
    assert sh.GetInput("opacity").Get() < 0.5


def test_no_lights_triangles_brands(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight) or p.HasAPI(UsdLux.LightAPI)]
    tris = sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    assert 0 < tris < 60_000, tris
    pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b", re.I)
    files = list(DIR.glob("*.usda")) + list(DIR.glob("*.md")) + list(DIR.glob("*.py"))
    hits = [(f.name, m.group(0)) for f in files for m in pat.finditer(f.read_text())]
    assert not hits, hits
