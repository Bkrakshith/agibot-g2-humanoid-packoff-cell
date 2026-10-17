"""Tests for the room (environment) asset built by assets/room/build.py.

Run: ../.venv-tools/bin/python -m pytest assets/tests/test_room.py
"""

from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOM_DIR = Path(__file__).resolve().parents[1] / "room"
MM = 0.001
TOL = 20 * MM
VISUAL = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
GUIDE = [UsdGeom.Tokens.guide]
WALLS = ("Wall_PosX", "Wall_NegX", "Wall_PosY", "Wall_NegY")
WORK_Z = 0.85
# Cell layout the room must accommodate (cell frame, robot base at origin).
CONVEYOR_Y_HALF = 1.8
RACK_X = -0.78
MAX_TRIANGLES = 60_000


@pytest.fixture(scope="module")
def stage():
    path = ROOM_DIR / "room.usda"
    assert path.is_file(), f"missing {path}"
    st = Usd.Stage.Open(str(path))
    assert st is not None
    return st


def _range(prim, purposes=VISUAL):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _p(stage, path):
    prim = stage.GetPrimAtPath(path)
    assert prim and prim.IsValid(), f"missing {path}"
    return prim


def test_layers_units_and_default_prim(stage):
    root = stage.GetDefaultPrim()
    assert root and root.GetName() == "Room"
    assert root.IsA(UsdGeom.Xform)
    assert Usd.ModelAPI(root).GetKind() == "component"
    for name in ("room.usda", "geo.usda", "physics.usda"):
        lst = Usd.Stage.Open(str(ROOM_DIR / name))
        assert UsdGeom.GetStageMetersPerUnit(lst) == 1.0, name
        assert UsdGeom.GetStageUpAxis(lst) == UsdGeom.Tokens.z, name
    spec = stage.GetRootLayer().GetPrimAtPath("/Room")
    assert [r.assetPath for r in spec.referenceList.GetAddedOrExplicitItems()] == ["./geo.usda"]
    assert [p.assetPath for p in spec.payloadList.GetAddedOrExplicitItems()] == ["./physics.usda"]
    assert (ROOM_DIR / "README.md").is_file()
    assert (ROOM_DIR / "build.py").is_file()


def test_physics_layer_only_overs():
    layer = Sdf.Layer.FindOrOpen(str(ROOM_DIR / "physics.usda"))
    geo = Usd.Stage.Open(str(ROOM_DIR / "geo.usda"))
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


def test_floor_top_at_zero_and_single_box_collider(stage):
    slab = _p(stage, "/Room/Geo/Floor/Slab")
    assert abs(_range(slab).GetMax()[2]) <= 1 * MM
    cols = [p for p in Usd.PrimRange(_p(stage, "/Room/Colliders/Floor"))
            if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(cols) == 1
    floor = cols[0]
    assert floor.IsA(UsdGeom.Cube)
    assert UsdGeom.Imageable(floor).ComputePurpose() == UsdGeom.Tokens.guide
    rng = _range(floor, GUIDE)
    assert abs(rng.GetMax()[2]) <= 1 * MM
    assert rng.GetSize()[0] >= 10 - TOL and rng.GetSize()[1] >= 10 - TOL
    mat, _ = UsdShade.MaterialBindingAPI(floor).ComputeBoundMaterial("physics")
    assert mat
    pm = UsdPhysics.MaterialAPI(mat.GetPrim())
    assert pm.GetStaticFrictionAttr().Get() == pytest.approx(0.6)
    assert pm.GetDynamicFrictionAttr().Get() == pytest.approx(0.6)


def test_floor_material_epoxy(stage):
    mat, _ = UsdShade.MaterialBindingAPI(_p(stage, "/Room/Geo/Floor/Slab")).ComputeBoundMaterial()
    assert mat and "Epoxy" in mat.GetPrim().GetName()
    shader = UsdShade.Shader(_p(stage, mat.GetPath().AppendChild("Shader")))
    assert shader.GetInput("roughness").Get() == pytest.approx(0.25, abs=0.05)


def test_everything_static(stage):
    rb = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert not rb


def test_room_extents_and_walls(stage):
    walls = _range(_p(stage, "/Room/Geo/Walls"))
    assert abs(walls.GetSize()[0] - 10.0) <= TOL
    assert abs(walls.GetSize()[1] - 10.0) <= TOL
    whole = _range(stage.GetDefaultPrim())
    assert abs(whole.GetSize()[0] - 10.0) <= TOL
    assert abs(whole.GetSize()[1] - 10.0) <= TOL
    for w in WALLS:
        r = _range(_p(stage, f"/Room/Geo/Walls/{w}"))
        assert abs(r.GetMin()[2]) <= 1 * MM, w
        assert abs(r.GetSize()[2] - 4.0) <= TOL, w
        col = _p(stage, f"/Room/Colliders/{w}")
        assert col.HasAPI(UsdPhysics.CollisionAPI), w
    # robot cell fits with >= 1 m clearance from the inner wall faces
    inner_neg_x = _range(_p(stage, "/Room/Geo/Walls/Wall_NegX")).GetMax()[0]
    inner_pos_x = _range(_p(stage, "/Room/Geo/Walls/Wall_PosX")).GetMin()[0]
    inner_neg_y = _range(_p(stage, "/Room/Geo/Walls/Wall_NegY")).GetMax()[1]
    inner_pos_y = _range(_p(stage, "/Room/Geo/Walls/Wall_PosY")).GetMin()[1]
    assert RACK_X - inner_neg_x >= 1.0
    assert inner_pos_x - 1.6 >= 1.0  # drainage channel / conveyor side
    assert CONVEYOR_Y_HALF + 1.0 <= min(-inner_neg_y, inner_pos_y)


def test_coves_and_kick_plates(stage):
    for w in WALLS:
        cove = _p(stage, f"/Room/Geo/Walls/Coves/Cove_{w.split('_')[1]}")
        assert cove.IsA(UsdGeom.Mesh)
        r = _range(cove)
        assert abs(r.GetSize()[2] - 0.1) <= 5 * MM  # radius ~0.1 m
        kick = _p(stage, f"/Room/Geo/Walls/KickPlates/Kick_{w.split('_')[1]}")
        assert abs(_range(kick).GetSize()[2] - 0.2) <= 2 * MM


def test_drainage_channel(stage):
    r = _range(_p(stage, "/Room/Geo/Floor/Drain"))
    assert abs(r.GetMidpoint()[0] - 1.6) <= 5 * MM
    assert abs(r.GetSize()[0] - 0.15) <= 5 * MM
    assert r.GetSize()[1] >= 9.0  # runs along Y
    assert r.GetMax()[2] <= 5 * MM
    assert _p(stage, "/Room/Geo/Floor/Drain/Grating").IsA(UsdGeom.Mesh)


def test_walkway_lines(stage):
    lines = [p for p in _p(stage, "/Room/Geo/Floor/Walkway").GetChildren() if p.IsA(UsdGeom.Mesh)]
    assert len(lines) >= 2
    for ln in lines:
        r = _range(ln)
        assert r.GetMin()[2] == pytest.approx(0.001, abs=1e-5)
        assert r.GetSize()[2] <= 1e-6  # coplanar strip
        c = UsdGeom.Mesh(ln).GetDisplayColorAttr().Get()[0]
        assert c[0] > 0.8 and c[1] > 0.6 and c[2] < 0.2  # safety yellow


def test_ceiling_and_cable_tray(stage):
    ceil = _range(_p(stage, "/Room/Geo/Ceiling/Slab"))
    assert abs(ceil.GetMin()[2] - 4.0) <= TOL
    tray = _range(_p(stage, "/Room/Geo/CableTray/Tray"))
    assert abs(tray.GetMin()[2] - 3.2) <= TOL
    assert tray.GetSize()[1] >= 9.0 and tray.GetSize()[1] > tray.GetSize()[0]


def _rect_lights(stage):
    return [p for p in Usd.PrimRange(_p(stage, "/Room/Lights")) if p.IsA(UsdLux.RectLight)]


def test_troffers(stage):
    troffers = [p for p in _rect_lights(stage) if p.GetName().startswith("Troffer_")]
    assert len(troffers) >= 8
    names = sorted(p.GetName() for p in troffers)
    assert names == [f"Troffer_{i:02d}" for i in range(len(names))]
    for p in troffers:
        L = UsdLux.RectLight(p)
        assert L.GetWidthAttr().Get() == pytest.approx(1.2)
        assert L.GetHeightAttr().Get() == pytest.approx(0.3)
        assert L.GetEnableColorTemperatureAttr().Get() is True
        assert L.GetColorTemperatureAttr().Get() == pytest.approx(5000)
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        down = xf.TransformDir(Gf.Vec3d(0, 0, -1))
        assert down[2] == pytest.approx(-1.0, abs=1e-6)  # emits downwards
        assert 3.8 <= xf.ExtractTranslation()[2] <= 4.0


def test_no_dome_light_and_fill_present(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)
                or "Dome" in str(p.GetTypeName())]
    fill = _p(stage, "/Room/Lights/Fill")
    assert fill.IsA(UsdLux.RectLight) or fill.IsA(UsdLux.SphereLight)
    assert stage.GetRootLayer().GetPrimAtPath("/Room") is not None


def _illuminance(stage, point):
    """Direct illuminance (lux) from the RectLights, treating radiance = intensity *
    2^exposure nits (Lambertian emitter, colour temperature normalised to luminance 1)."""
    total = 0.0
    n_u, n_v = 24, 24
    for p in _rect_lights(stage):
        L = UsdLux.RectLight(p)
        radiance = L.GetIntensityAttr().Get() * 2 ** (L.GetExposureAttr().Get() or 0.0)
        w, h = L.GetWidthAttr().Get(), L.GetHeightAttr().Get()
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        us = ((np.arange(n_u) + 0.5) / n_u - 0.5) * w
        vs = ((np.arange(n_v) + 0.5) / n_v - 0.5) * h
        dA = w * h / (n_u * n_v)
        for u in us:
            for v in vs:
                q = xf.Transform(Gf.Vec3d(u, v, 0))
                d = np.array(q) - np.array(point)
                r2 = float(d @ d)
                cos_e = d[2] / np.sqrt(r2)  # emitter normal is -Z
                if cos_e <= 0:
                    continue
                total += radiance * dA * cos_e * cos_e / r2
    return total


@pytest.mark.parametrize("xy", [(0.0, 0.0), (0.75, 1.5), (0.75, -1.5), (RACK_X, 0.0)])
def test_work_surface_illuminance(stage, xy):
    lux = _illuminance(stage, (xy[0], xy[1], WORK_Z))
    # LIGHT_SCALE 2.2 + fill 40 nits (2026-10-11 brighter-room request): ~1300-1400 lux at the work surface
    assert 1100 <= lux <= 1700, lux


def test_triangle_budget(stage):
    tris = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            counts = UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []
            tris += sum(c - 2 for c in counts)
            assert UsdGeom.Mesh(p).GetSubdivisionSchemeAttr().Get() == "none", p.GetPath()
    assert 0 < tris < MAX_TRIANGLES, tris
