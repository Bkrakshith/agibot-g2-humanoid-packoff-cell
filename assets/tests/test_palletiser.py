"""Tests for assets/palletiser (static compact cobot palletising cell prop, generic).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python -m pytest assets/tests/test_palletiser.py
"""

import re
from pathlib import Path

import pytest
from pxr import Gf, Pcp, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from _brands import BRANDS

DIR = Path(__file__).resolve().parents[1] / "palletiser"
ROOT = DIR / "palletiser.usda"
NAME = "Palletiser"
MM = 0.001
TOL = 20 * MM
BED = 0.82
BED_TOL = 5 * MM
SIZE = (3.2, 2.2, None)         # X, Y, Z (height not specified)
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]


@pytest.fixture(scope="module")
def stage():
    assert ROOT.is_file(), f"missing {ROOT} (run assets/palletiser/build.py)"
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
    for ln in ("palletiser.usda", "geo.usda", "physics.usda"):
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


PALLET = (0.8, 1.2, 0.144)


def test_infeed(stage):
    _check_colliders(_get(stage, "Infeed"), 3)
    bed = _get(stage, "Infeed/Colliders/Bed")
    r = _range(bed, GUIDE)
    assert abs(r.GetMax()[2] - BED) <= BED_TOL
    e = _pos(_get(stage, "Infeed/EntryPoint"))
    root_r = _range(stage.GetDefaultPrim())
    assert abs(e[2] - BED) <= BED_TOL and abs(e[1] - root_r.GetMax()[1]) <= TOL
    assert abs(r.GetMax()[1] - root_r.GetMax()[1]) <= 2 * MM, "infeed starts at the +Y edge"
    pk = _pos(_get(stage, "Infeed/PickPoint"))
    assert abs(pk[2] - BED) <= BED_TOL and r.GetMin()[1] <= pk[1] - 0.3


@pytest.mark.parametrize("st_name", ["StationA", "StationB"])
def test_pallet_stations(stage, st_name):
    pal = _get(stage, f"{st_name}/Pallet")
    r = _range(pal)
    for ax, want in enumerate(PALLET):
        assert abs(r.GetSize()[ax] - want) <= 2 * MM, (ax, r.GetSize()[ax])
    assert abs(r.GetMin()[2]) <= 1e-4
    _check_colliders(pal, 1)
    so = _get(stage, f"{st_name}/StackOrigin")
    p = _pos(so)
    assert abs(p[2] - PALLET[2]) <= 1 * MM
    assert abs(p[0] - r.GetMidpoint()[0]) <= 1 * MM and abs(p[1] - r.GetMidpoint()[1]) <= 1 * MM


def test_stations_either_side_of_robot(stage):
    a = _pos(_get(stage, "StationA/StackOrigin"))
    b = _pos(_get(stage, "StationB/StackOrigin"))
    rb = _range(_get(stage, "Robot/Pedestal"))
    assert a[0] > rb.GetMax()[0] and b[0] < rb.GetMin()[0]


def test_station_a_stack_and_b_empty(stage):
    stack = _get(stage, "StationA/Stack")
    cartons = [p for p in stack.GetChildren() if p.GetName().startswith("Carton_")]
    assert len(cartons) == 12, "4 per layer x 3 layers"
    pal = _range(_get(stage, "StationA/Pallet"))
    ranges = [_range(c) for c in cartons]
    zs = sorted({round(r.GetMin()[2], 3) for r in ranges})
    assert len(zs) == 3 and abs(zs[0] - PALLET[2]) <= 1 * MM
    for r in ranges:
        assert r.GetMin()[0] >= pal.GetMin()[0] - TOL and r.GetMax()[0] <= pal.GetMax()[0] + TOL
        assert r.GetMin()[1] >= pal.GetMin()[1] - TOL and r.GetMax()[1] <= pal.GetMax()[1] + TOL
    for i in range(len(ranges)):                                    # no interpenetration
        for j in range(i + 1, len(ranges)):
            ov = Gf.Range3d.GetIntersection(ranges[i], ranges[j])
            if not ov.IsEmpty():
                s = ov.GetSize()
                assert min(s[0], s[1], s[2]) <= 1.5 * MM, (cartons[i].GetName(), cartons[j].GetName())
    for c in cartons:
        arcs = [a.GetTargetLayer().identifier for a in Usd.PrimCompositionQuery(c).GetCompositionArcs()
                if a.GetArcType() == Pcp.ArcTypeReference]
        assert any("carton_sealed" in a for a in arcs), arcs
        assert _colliders(c), f"static collider on {c.GetName()}"
    b = stage.GetPrimAtPath(f"/{NAME}/StationB/Stack")
    assert not b or not [p for p in Usd.PrimRange(b) if p.IsA(UsdGeom.Mesh)]


def test_robot(stage):
    _check_colliders(_get(stage, "Robot/Pedestal"), 1)
    ped = _range(_get(stage, "Robot/Pedestal"))
    assert 0.5 <= ped.GetMax()[2] <= 0.7, "~0.6 m pedestal"
    for link in ("Base", "Shoulder", "UpperArm", "Elbow", "Forearm", "Wrist1", "Wrist2", "Wrist3", "Tool"):
        _get(stage, f"Robot/Arm/{link}")
    # tool is posed mid-pick over the carton at the infeed pick point
    tool = _range(_get(stage, "Robot/Arm/Tool"))
    pk = _pos(_get(stage, "Infeed/PickPoint"))
    mid = tool.GetMidpoint()
    assert abs(mid[0] - pk[0]) <= 0.05 and abs(mid[1] - pk[1]) <= 0.05
    assert BED + 0.25 <= tool.GetMin()[2] <= BED + 0.25 + 0.10
    # reach: shoulder-to-tool-flange distance stays within a ~1.3 m cobot envelope
    base = _range(_get(stage, "Robot/Arm/Base"))
    assert base.GetMin()[2] >= ped.GetMax()[2] - 1 * MM
    import math
    far = [(x, y) for x in (0.4, 1.2) for y in (-0.6, 0.6)]
    reach = stage.GetPrimAtPath(f"/{NAME}/Robot").GetAttribute("packoff:reach").Get()
    assert 1.25 <= reach <= 1.35
    bc = base.GetMidpoint()
    assert max(math.hypot(x - bc[0], y - bc[1]) for x, y in far) <= reach + 0.1


def test_perimeter_and_cabinet(stage):
    _get(stage, "ControlCabinet")
    bollards = [p for p in _get(stage, "Perimeter").GetChildren() if p.GetName().startswith("Bollard_")]
    scanners = [p for p in _get(stage, "Perimeter").GetChildren() if p.GetName().startswith("ScannerPost_")]
    assert len(bollards) >= 4 and len(scanners) >= 2


def test_no_lights_triangles_brands(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight) or p.HasAPI(UsdLux.LightAPI)]
    tris = sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    assert 0 < tris < 60_000, tris
    pat = re.compile(r"\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b", re.I)
    files = list(DIR.glob("*.usda")) + list(DIR.glob("*.md")) + list(DIR.glob("*.py"))
    hits = [(f.name, m.group(0)) for f in files for m in pat.finditer(f.read_text())]
    assert not hits, hits
