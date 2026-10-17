"""Tests for assets/floor_bin (static open-top red HDPE reject bin on castors).

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_floor_bin.py
"""

import re
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from _brands import BRANDS

DIR = Path(__file__).resolve().parents[1] / "floor_bin"
ROOT = DIR / "floor_bin.usda"
NAME = "FloorBin"
MM = 0.001
TOL = 20 * MM
BED = 0.82
BED_TOL = 5 * MM
SIZE = (0.6, 0.8, 0.7)          # X, Y, Z
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]


@pytest.fixture(scope="module")
def stage():
    assert ROOT.is_file(), f"missing {ROOT} (run assets/floor_bin/build.py)"
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
    for ln in ("floor_bin.usda", "geo.usda", "physics.usda"):
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


def test_compound_colliders_and_interior(stage):
    cols = {c.GetName(): c for c in _check_colliders(_get(stage, "Colliders"), 5)}
    assert {"Base", "WallNegX", "WallPosX", "WallNegY", "WallPosY"} <= set(cols)
    for c in cols.values():
        assert c.IsA(UsdGeom.Cube)
    iv = _get(stage, "InteriorVolume")
    assert not iv.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(iv).ComputeVisibility() == UsdGeom.Tokens.invisible
    r = _range(iv, GUIDE + VISUAL)
    base = _range(cols["Base"], GUIDE)
    assert abs(r.GetMin()[2] - base.GetMax()[2]) <= 1 * MM
    assert r.GetMax()[2] <= SIZE[2] + 1 * MM and r.GetSize()[2] > 0.45
    assert r.GetSize()[0] > 0.45 and r.GetSize()[1] > 0.65
    # walls bound the interior on all four sides
    assert _range(cols["WallNegX"], GUIDE).GetMax()[0] >= r.GetMin()[0] - 2 * MM
    assert _range(cols["WallPosX"], GUIDE).GetMin()[0] <= r.GetMax()[0] + 2 * MM


def test_castors_rim_colour(stage):
    castors = [p for p in _get(stage, "Geo").GetChildren() if p.GetName().startswith("Castor_")]
    assert len(castors) == 4
    for c in castors:
        assert _range(c).GetMin()[2] <= 1e-4
    rim = _range(_get(stage, "Geo/Rim"))
    body = _range(_get(stage, "Geo/Body"))
    assert abs(rim.GetSize()[0] - SIZE[0]) <= TOL and abs(rim.GetSize()[1] - SIZE[1]) <= TOL
    assert abs(rim.GetMax()[2] - SIZE[2]) <= TOL
    assert body.GetMin()[2] > 0.08, "body rides on castors"
    # flared: the tub is narrower at the bottom than at the rim
    mesh = UsdGeom.Mesh(_get(stage, "Geo/Body"))
    pts = mesh.GetPointsAttr().Get()
    zmin = min(p[2] for p in pts)
    bottom_x = max(abs(p[0]) for p in pts if abs(p[2] - zmin) < 1e-4)
    assert bottom_x <= rim.GetMax()[0] - 0.015
    mat, _ = UsdShade.MaterialBindingAPI(_get(stage, "Geo/Body")).ComputeBoundMaterial()
    sh = [UsdShade.Shader(p) for p in Usd.PrimRange(mat.GetPrim())
          if UsdShade.Shader(p) and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"][0]
    r, g, b = sh.GetInput("diffuseColor").Get()
    assert r > 3 * g and r > 3 * b, "red reject bin"


def test_no_lights_triangles_brands(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight) or p.HasAPI(UsdLux.LightAPI)]
    tris = sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    assert 0 < tris < 60_000, tris
    pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b", re.I)
    files = list(DIR.glob("*.usda")) + list(DIR.glob("*.md")) + list(DIR.glob("*.py"))
    hits = [(f.name, m.group(0)) for f in files for m in pat.finditer(f.read_text())]
    assert not hits, hits
