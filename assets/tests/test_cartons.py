"""Tests for the corrugated carton assets built by assets/cartons/build.py.

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest assets/tests/test_cartons.py
"""

import re
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from _brands import BRANDS

DIR = Path(__file__).resolve().parents[1] / "cartons"
MM = 0.001
TOL = 20 * MM          # spec acceptance tolerance
TIGHT = 2 * MM
MAX_TRIS = 60_000
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]

VARIANTS = {
    "carton_open": dict(prim="CartonOpen", body=(400, 600, 250), rigid=True, mass=0.55,
                        colliders=("Base", "WallNegX", "WallPosX", "WallNegY", "WallPosY")),
    "carton_sealed": dict(prim="CartonSealed", body=(400, 600, 250), rigid=True, mass=0.55,
                          colliders=("Box",)),
    "carton_flat": dict(prim="CartonFlat", body=(650, 1000, 8), rigid=False,
                        colliders=("Box",)),
    "carton_flat_stack": dict(prim="CartonFlatStack", body=(650, 1000, 160), rigid=False,
                              colliders=("Box",)),
}
PARAMS = [pytest.param(k, v, id=k) for k, v in VARIANTS.items()]


# ---------------------------------------------------------------- helpers
def _open(name):
    path = DIR / name / f"{name}.usda"
    assert path.is_file(), f"missing {path} (run assets/cartons/build.py)"
    st = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    assert st is not None
    return st


def _range(prim, purposes=VISUAL):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _colliders(root):
    return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]


def _surface_input(mat_prim, name):
    for sh in Usd.PrimRange(mat_prim):
        s = UsdShade.Shader(sh)
        if s and s.GetIdAttr().Get() == "UsdPreviewSurface":
            inp = s.GetInput(name)
            return inp.Get() if inp else None
    return None


def _bound(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert mat, f"no material on {prim.GetPath()}"
    return mat.GetPrim()


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("name,spec", PARAMS)
def test_layers_units_default_prim(name, spec):
    st = _open(name)
    root = st.GetDefaultPrim()
    assert root and root.GetName() == spec["prim"] and root.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(root).GetKind() == "component"
    for ln in (f"{name}.usda", "geo.usda", "physics.usda"):
        ls = Usd.Stage.Open(str(DIR / name / ln))
        assert UsdGeom.GetStageMetersPerUnit(ls) == 1.0, ln
        assert UsdGeom.GetStageUpAxis(ls) == UsdGeom.Tokens.z, ln
    ps = st.GetRootLayer().GetPrimAtPath(root.GetPath())
    assert [r.assetPath for r in ps.referenceList.GetAddedOrExplicitItems()] == ["./geo.usda"]
    assert [p.assetPath for p in ps.payloadList.GetAddedOrExplicitItems()] == ["./physics.usda"]
    assert (DIR / name / "README.md").is_file()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_physics_layer_only_overs(name):
    layer = Sdf.Layer.FindOrOpen(str(DIR / name / "physics.usda"))
    geo = Usd.Stage.Open(str(DIR / name / "geo.usda"))
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


@pytest.mark.parametrize("name,spec", PARAMS)
def test_body_bbox_and_origin(name, spec):
    st = _open(name)
    root = st.GetDefaultPrim()
    body = st.GetPrimAtPath(root.GetPath().AppendPath("Geo/Body"))
    assert body, "Geo/Body missing"
    rng = _range(body)
    for ax, want in enumerate(spec["body"]):
        assert abs(rng.GetSize()[ax] - want * MM) <= TIGHT, (ax, rng.GetSize()[ax], want)
    whole = _range(root)
    assert abs(whole.GetMin()[2]) <= 1e-6
    mid = whole.GetMidpoint()     # (flat stack blanks are misaligned by <= 0.8 mm)
    assert abs(mid[0]) <= 1 * MM and abs(mid[1]) <= 1 * MM
    # anything beyond the body (flaps, tape) stays within the spec tolerance in Z
    assert whole.GetSize()[2] <= spec["body"][2] * MM + TOL


@pytest.mark.parametrize("name,spec", PARAMS)
def test_colliders(name, spec):
    st = _open(name)
    root = st.GetDefaultPrim()
    cols = {p.GetName(): p for p in _colliders(root)}
    assert set(spec["colliders"]) <= set(cols), sorted(cols)
    assert len(cols) == len(spec["colliders"])
    env = Gf.Range3d()
    for c in cols.values():
        assert c.IsA(UsdGeom.Cube), c.GetPath()
        assert UsdGeom.Imageable(c).ComputePurpose() == UsdGeom.Tokens.guide
        assert "PhysxCollisionAPI" in c.GetMetadata("apiSchemas").GetAppliedItems()
        assert c.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.003)
        mat, _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial("physics")
        assert mat, c.GetPath()
        assert UsdPhysics.MaterialAPI(mat.GetPrim()).GetStaticFrictionAttr().Get() > 0
        env.UnionWith(_range(c, GUIDE))
    for ax, want in enumerate(spec["body"]):
        assert abs(env.GetSize()[ax] - want * MM) <= TIGHT, (ax, env.GetSize()[ax])


@pytest.mark.parametrize("name,spec", PARAMS)
def test_rigid_body_only_on_movable_cartons(name, spec):
    st = _open(name)
    root = st.GetDefaultPrim()
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if spec["rigid"]:
        assert bodies == [root]
        assert root.HasAPI(UsdPhysics.MassAPI)
        assert not UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
    else:
        assert not bodies


def test_open_carton_walls_flaps_and_interior():
    st = _open("carton_open")
    root = st.GetDefaultPrim()
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(0.55)
    cols = {p.GetName(): p for p in _colliders(root)}
    for w, ax in (("WallNegX", 0), ("WallPosX", 0), ("WallNegY", 1), ("WallPosY", 1)):
        t = _range(cols[w], GUIDE).GetSize()[ax]
        assert 3 * MM <= t <= 5 * MM, (w, t)
    # four top flaps, folded out and down: below the rim, outside the body
    for f in ("FlapNegX", "FlapPosX", "FlapNegY", "FlapPosY"):
        p = st.GetPrimAtPath(f"/CartonOpen/Geo/Flaps/{f}")
        assert p and p.IsA(UsdGeom.Mesh), f
        assert not p.HasAPI(UsdPhysics.CollisionAPI)
        r = _range(p)
        assert r.GetMin()[2] < 0.25 - 0.15        # hangs down most of its 200 mm length
        assert r.GetMax()[2] <= 0.25 + 2 * MM
    whole = _range(root)
    assert 0.40 < whole.GetSize()[0] <= 0.52 and 0.60 < whole.GetSize()[1] <= 0.72
    # interior volume: invisible guide box, no collision, on the base, inside the walls
    iv = st.GetPrimAtPath("/CartonOpen/InteriorVolume")
    assert iv and iv.IsA(UsdGeom.Gprim) and not iv.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(iv).ComputeVisibility() == UsdGeom.Tokens.invisible
    r = _range(iv, GUIDE + VISUAL)
    assert abs(r.GetMin()[2] - _range(cols["Base"], GUIDE).GetMax()[2]) <= 1 * MM
    assert abs(r.GetMax()[2] - 0.25) <= 1 * MM
    assert abs(r.GetSize()[0] - (0.400 - 0.008)) <= 1 * MM
    assert abs(r.GetSize()[1] - (0.600 - 0.008)) <= 1 * MM


@pytest.mark.parametrize("name", ["carton_open", "carton_sealed", "carton_flat"])
def test_kraft_look(name):
    st = _open(name)
    body = st.GetPrimAtPath(st.GetDefaultPrim().GetPath().AppendPath("Geo/Body"))
    mat = _bound(body)
    assert _surface_input(mat, "roughness") == pytest.approx(0.85, abs=0.05)
    pv = UsdGeom.PrimvarsAPI(body).GetPrimvar("displayColor")
    assert pv and pv.HasValue()
    cols = {tuple(round(c, 4) for c in v) for v in pv.Get()}
    assert len(cols) > 1, "slight colour variation expected"
    for r, g, b in cols:
        assert r > g > b, "kraft brown"


def test_sealed_tape_and_mass():
    st = _open("carton_sealed")
    root = st.GetDefaultPrim()
    contents = root.GetAttribute("packoff:contentsMass")
    assert contents and contents.Get() >= 0.0
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(0.55 + contents.Get())
    tape = st.GetPrimAtPath("/CartonSealed/Geo/Tape")
    assert tape and not tape.HasAPI(UsdPhysics.CollisionAPI)
    r = _range(tape)
    assert abs(r.GetSize()[0] - 0.048) <= 1 * MM          # 48 mm tape across X
    assert r.GetSize()[1] >= 0.600                        # runs the full long axis
    assert r.GetMax()[2] >= 0.250
    tmat, kmat = _bound(tape), _bound(st.GetPrimAtPath("/CartonSealed/Geo/Body"))
    assert _surface_input(tmat, "roughness") < _surface_input(kmat, "roughness")


def test_flat_stack_has_20_blanks():
    st = _open("carton_flat_stack")
    blanks = [p for p in st.GetPrimAtPath("/CartonFlatStack/Geo/Body").GetChildren()
              if p.IsA(UsdGeom.Mesh)]
    assert len(blanks) == 20


@pytest.mark.parametrize("name", list(VARIANTS))
def test_no_lights_and_triangle_budget(name):
    st = _open(name)
    assert not [p for p in st.Traverse() if p.IsA(UsdLux.DomeLight) or p.HasAPI(UsdLux.LightAPI)]
    tris = sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in st.Traverse() if p.IsA(UsdGeom.Mesh))
    assert 0 < tris < MAX_TRIS, tris


def test_no_brand_strings():
    files = list(DIR.rglob("*.usda")) + list(DIR.rglob("*.md")) + list(DIR.rglob("*.py"))
    assert files
    pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b", re.I)
    hits = [(f.name, m.group(0)) for f in files for m in pat.finditer(f.read_text())]
    assert not hits, hits
