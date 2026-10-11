"""Generate the chilled-factory room asset (assets/room/*.usda).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/room/build.py [--light-scale 1.0]

Origin = cell origin (G2 robot base) on the floor. The 10 x 10 m room spans
X [-4.5, 5.5], Y [-5, 5] (outer wall faces), so the rack at X -0.78, the conveyor
(Y +-1.8) and the drainage channel at X +1.6 all have >= 1 m clearance to the walls.
"""

import argparse
import math
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent

# ---- layout (m) -------------------------------------------------------------
X_MIN, X_MAX = -4.5, 5.5          # outer wall faces
Y_MIN, Y_MAX = -5.0, 5.0
WALL_T = 0.10
WALL_H = 4.0
COVE_R = 0.10
KICK_H, KICK_T, KICK_Z0 = 0.20, 0.003, COVE_R   # kick plate sits on top of the cove
CEIL_Z, CEIL_T = 4.0, 0.05
FLOOR_COL_T = 0.20                # floor collider thickness (top at Z = 0)
DRAIN_X, DRAIN_W = 1.6, 0.15
WALKWAY_X = (2.6, 4.4)            # aisle between drain and the +X wall
WALKWAY_W = 0.10
WALKWAY_DZ = 0.001
TRAY_X, TRAY_Z, TRAY_W, TRAY_H = -2.0, 3.2, 0.30, 0.08

# ---- lighting -----------------------------------------------------------------
# Default light scale; multiply every light intensity (exposed as --light-scale).
LIGHT_SCALE = 2.2   # 2026-10-11 user request: noticeably brighter hall (was 1.0)
TROFFER_W, TROFFER_H = 1.2, 0.3   # RectLight width (local X) x height (local Y)
TROFFER_NITS = 2700.0             # radiance, see README (~3050 lm per fitting)
TROFFER_XS = (-3.5, -1.5, 0.5, 2.5, 4.5)
TROFFER_YS = (-4.0, -2.0, 0.0, 2.0, 4.0)
TROFFER_Z = 3.968                 # just below the diffuser face (3.97)
FILL_SIZE, FILL_Z, FILL_NITS = 8.0, 3.9, 40.0   # fill doubled (was 20) to lift the corners
CCT = 5000.0

IX0, IX1 = X_MIN + WALL_T, X_MAX - WALL_T   # inner wall faces
IY0, IY1 = Y_MIN + WALL_T, Y_MAX - WALL_T
CX, CY = (X_MIN + X_MAX) / 2, (Y_MIN + Y_MAX) / 2


# ---- mesh helpers ------------------------------------------------------------
class MeshBuilder:
    def __init__(self):
        self.pts, self.counts, self.idx = [], [], []

    def _face(self, ids):
        self.counts.append(len(ids))
        self.idx.extend(ids)

    def box(self, lo, hi):
        b = len(self.pts)
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        for z in (z0, z1):
            for y in (y0, y1):
                for x in (x0, x1):
                    self.pts.append((x, y, z))
        # vertex i = x + 2y + 4z
        for f in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self._face([b + i for i in f])
        return self

    def quad(self, a, b, c, d):
        base = len(self.pts)
        self.pts.extend([a, b, c, d])
        self._face([base, base + 1, base + 2, base + 3])
        return self

    def cylinder(self, p0, axis, radius, length, n=12):
        """Capped cylinder starting at p0 along axis 'x'|'y'|'z'."""
        a = "xyz".index(axis)
        u, v = [i for i in range(3) if i != a]
        b = len(self.pts)
        for end in (0.0, length):
            for k in range(n):
                t = 2 * math.pi * k / n
                p = list(p0)
                p[a] += end
                p[u] += radius * math.cos(t)
                p[v] += radius * math.sin(t)
                self.pts.append(tuple(p))
        for k in range(n):
            k1 = (k + 1) % n
            self._face([b + k, b + k1, b + n + k1, b + n + k])
        self._face([b + k for k in reversed(range(n))])
        self._face([b + n + k for k in range(n)])
        return self

    def strip(self, rows):
        """rows: list of point-lists (same length); quads between consecutive rows."""
        b = len(self.pts)
        m = len(rows[0])
        for r in rows:
            self.pts.extend(r)
        for i in range(len(rows) - 1):
            for j in range(m - 1):
                p = b + i * m + j
                self._face([p, p + 1, p + m + 1, p + m])
        return self

    def emit(self, stage, path, material, color):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in self.pts]))
        mesh.CreateFaceVertexCountsAttr(self.counts)
        mesh.CreateFaceVertexIndicesAttr(self.idx)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return mesh


def box(stage, path, lo, hi, mat):
    return MeshBuilder().box(lo, hi).emit(stage, path, *mat)


def triangles(stage):
    return sum(sum(c - 2 for c in UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))


# ---- materials ----------------------------------------------------------------
def make_material(stage, name, color, roughness, metallic=0.0, emissive=None):
    mat = UsdShade.Material.Define(stage, f"/Room/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, color


def collider_cube(stage, path, lo, hi):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    s = [b - a for a, b in zip(lo, hi)]
    cube.AddTranslateOp().Set(Gf.Vec3d(*c))
    cube.AddScaleOp().Set(Gf.Vec3f(*s))
    cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    return cube


def new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    return stage


# ---- geometry layer -----------------------------------------------------------
def build_geo(path, light_scale):
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Room")
    stage.SetDefaultPrim(root.GetPrim())
    for p in ("/Room/Geo", "/Room/Geo/Floor", "/Room/Geo/Walls", "/Room/Geo/Ceiling",
              "/Room/Geo/CableTray", "/Room/Lights", "/Room/Colliders"):
        UsdGeom.Xform.Define(stage, p)
    UsdGeom.Scope.Define(stage, "/Room/Looks")

    epoxy = make_material(stage, "EpoxyGrey", (0.42, 0.43, 0.44), 0.25)
    pvc = make_material(stage, "PVCWhite", (0.92, 0.92, 0.90), 0.35)
    steel = make_material(stage, "Stainless", (0.75, 0.75, 0.76), 0.30, 1.0)
    dark = make_material(stage, "DrainDark", (0.04, 0.04, 0.04), 0.8)
    yellow = make_material(stage, "SafetyYellow", (0.95, 0.75, 0.05), 0.5)
    ceil_mat = make_material(stage, "CeilingWhite", (0.88, 0.88, 0.88), 0.6)
    diffuser = make_material(stage, "TrofferDiffuser", (0.95, 0.95, 0.95), 0.4, 0.0, (1.0, 1.0, 1.0))
    galv = make_material(stage, "GalvanisedSteel", (0.62, 0.63, 0.64), 0.45, 1.0)
    cable = make_material(stage, "CableBlack", (0.03, 0.03, 0.03), 0.6)

    # Floor: epoxy slab (visual), drain, walkway lines
    box(stage, "/Room/Geo/Floor/Slab", (X_MIN, Y_MIN, -0.05), (X_MAX, Y_MAX, 0.0), epoxy)
    UsdGeom.Xform.Define(stage, "/Room/Geo/Floor/Drain")
    hw = DRAIN_W / 2
    dy0, dy1 = IY0, IY1
    slot = MeshBuilder().quad((DRAIN_X - hw + 0.012, dy0, 0.0005), (DRAIN_X + hw - 0.012, dy0, 0.0005),
                              (DRAIN_X + hw - 0.012, dy1, 0.0005), (DRAIN_X - hw + 0.012, dy1, 0.0005))
    slot.emit(stage, "/Room/Geo/Floor/Drain/Slot", *dark)
    frame = MeshBuilder()
    frame.box((DRAIN_X - hw, dy0, 0.0), (DRAIN_X - hw + 0.012, dy1, 0.003))
    frame.box((DRAIN_X + hw - 0.012, dy0, 0.0), (DRAIN_X + hw, dy1, 0.003))
    frame.emit(stage, "/Room/Geo/Floor/Drain/Frame", *steel)
    grating = MeshBuilder()
    inner = DRAIN_W - 2 * 0.012
    for k in range(4):   # bearing bars along Y
        x = DRAIN_X - inner / 2 + inner * (k + 0.5) / 4
        grating.box((x - 0.0015, dy0, 0.001), (x + 0.0015, dy1, 0.003))
    y = dy0 + 0.05
    while y < dy1 - 0.02:  # cross bars every 0.1 m
        grating.box((DRAIN_X - inner / 2, y - 0.002, 0.0015), (DRAIN_X + inner / 2, y + 0.002, 0.0028))
        y += 0.1
    grating.emit(stage, "/Room/Geo/Floor/Drain/Grating", *steel)

    UsdGeom.Xform.Define(stage, "/Room/Geo/Floor/Walkway")
    for i, x in enumerate(WALKWAY_X):
        z = WALKWAY_DZ
        MeshBuilder().quad((x - WALKWAY_W / 2, IY0 + 0.2, z), (x + WALKWAY_W / 2, IY0 + 0.2, z),
                           (x + WALKWAY_W / 2, IY1 - 0.2, z), (x - WALKWAY_W / 2, IY1 - 0.2, z)) \
            .emit(stage, f"/Room/Geo/Floor/Walkway/Line_{i:02d}", *yellow)

    # Walls (full outer footprint, 4 m high), coves, kick plates
    walls = {
        "PosX": ((IX1, Y_MIN, 0), (X_MAX, Y_MAX, WALL_H)),
        "NegX": ((X_MIN, Y_MIN, 0), (IX0, Y_MAX, WALL_H)),
        "PosY": ((IX0, IY1, 0), (IX1, Y_MAX, WALL_H)),
        "NegY": ((IX0, Y_MIN, 0), (IX1, IY0, WALL_H)),
    }
    UsdGeom.Xform.Define(stage, "/Room/Geo/Walls/Coves")
    UsdGeom.Xform.Define(stage, "/Room/Geo/Walls/KickPlates")
    n_arc = 8
    for name, (lo, hi) in walls.items():
        box(stage, f"/Room/Geo/Walls/Wall_{name}", lo, hi, pvc)
        collider_cube(stage, f"/Room/Colliders/Wall_{name}", lo, hi)
        # local frame: n = inward normal axis, face = inner face coordinate, sign = inward dir
        axis, sign, face = {"PosX": (0, -1, IX1), "NegX": (0, 1, IX0),
                            "PosY": (1, -1, IY1), "NegY": (1, 1, IY0)}[name]
        along = 1 - axis
        a0, a1 = (IY0, IY1) if axis == 0 else (IX0, IX1)

        def pt(n_off, z, a):
            p = [0.0, 0.0, z]
            p[axis] = face + sign * n_off
            p[along] = a
            return tuple(p)

        rows = []
        for k in range(n_arc + 1):
            t = -0.5 * math.pi * k / n_arc   # 0 -> on wall at z=r, -90 deg -> on floor
            n_off = COVE_R - COVE_R * math.cos(t)
            z = COVE_R + COVE_R * math.sin(t)
            rows.append([pt(n_off, z, a0), pt(n_off, z, a1)])
        MeshBuilder().strip(rows).emit(stage, f"/Room/Geo/Walls/Coves/Cove_{name}", *pvc)
        k0 = [0, 0, KICK_Z0]
        k1 = [0, 0, KICK_Z0 + KICK_H]
        k0[axis], k1[axis] = sorted((face, face + sign * KICK_T))
        k0[along], k1[along] = a0, a1
        box(stage, f"/Room/Geo/Walls/KickPlates/Kick_{name}", k0, k1, steel)

    # Ceiling + troffer housings
    box(stage, "/Room/Geo/Ceiling/Slab", (X_MIN, Y_MIN, CEIL_Z), (X_MAX, Y_MAX, CEIL_Z + CEIL_T), ceil_mat)
    housings, diffusers = MeshBuilder(), MeshBuilder()
    i = 0
    for x in TROFFER_XS:
        for y in TROFFER_YS:
            housings.box((x - 0.62, y - 0.165, 3.975), (x + 0.62, y + 0.165, CEIL_Z))
            diffusers.box((x - TROFFER_W / 2, y - TROFFER_H / 2, 3.97), (x + TROFFER_W / 2, y + TROFFER_H / 2, 3.975))
            light = UsdLux.RectLight.Define(stage, f"/Room/Lights/Troffer_{i:02d}")
            light.CreateWidthAttr(TROFFER_W)
            light.CreateHeightAttr(TROFFER_H)
            light.CreateIntensityAttr(TROFFER_NITS * light_scale)
            light.CreateExposureAttr(0.0)
            light.CreateNormalizeAttr(False)
            light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(CCT)
            light.AddTranslateOp().Set(Gf.Vec3d(x, y, TROFFER_Z))
            i += 1
    housings.emit(stage, "/Room/Geo/Ceiling/TrofferHousings", *ceil_mat)
    diffusers.emit(stage, "/Room/Geo/Ceiling/TrofferDiffusers", *diffuser)

    fill = UsdLux.RectLight.Define(stage, "/Room/Lights/Fill")
    fill.CreateWidthAttr(FILL_SIZE)
    fill.CreateHeightAttr(FILL_SIZE)
    fill.CreateIntensityAttr(FILL_NITS * light_scale)
    fill.CreateExposureAttr(0.0)
    fill.CreateNormalizeAttr(False)
    fill.CreateEnableColorTemperatureAttr(True)
    fill.CreateColorTemperatureAttr(CCT)
    fill.AddTranslateOp().Set(Gf.Vec3d(CX, CY, FILL_Z))
    fill.GetPrim().SetCustomDataByKey("note", "weak fill standing in for wall/ceiling bounce; no dome/HDRI")

    # Overhead ladder cable tray along Y, tray bottom at Z 3.2
    y0, y1 = IY0 + 0.05, IY1 - 0.05
    tray = MeshBuilder()
    for sx in (-1, 1):
        x = TRAY_X + sx * (TRAY_W / 2 - 0.001)
        tray.box((x - 0.0015, y0, TRAY_Z), (x + 0.0015, y1, TRAY_Z + TRAY_H))
        tray.box((x - 0.012 * (sx < 0) - 0.0015 * (sx > 0), y0, TRAY_Z),
                 (x + 0.012 * (sx > 0) + 0.0015 * (sx < 0), y1, TRAY_Z + 0.003))
    y = y0 + 0.15
    while y < y1:
        tray.box((TRAY_X - TRAY_W / 2, y - 0.012, TRAY_Z), (TRAY_X + TRAY_W / 2, y + 0.012, TRAY_Z + 0.015))
        y += 0.3
    tray.emit(stage, "/Room/Geo/CableTray/Tray", *galv)
    hangers = MeshBuilder()
    y = y0 + 0.5
    while y < y1:
        hangers.box((TRAY_X - TRAY_W / 2 - 0.05, y - 0.02, TRAY_Z - 0.04), (TRAY_X + TRAY_W / 2 + 0.05, y + 0.02, TRAY_Z))
        for sx in (-1, 1):
            hangers.cylinder((TRAY_X + sx * (TRAY_W / 2 + 0.03), y, TRAY_Z), "z", 0.005, CEIL_Z - TRAY_Z, n=8)
        y += 1.5
    hangers.emit(stage, "/Room/Geo/CableTray/Hangers", *galv)
    cables = MeshBuilder()
    for k, (dx, r) in enumerate(((-0.08, 0.015), (-0.04, 0.012), (0.0, 0.018), (0.05, 0.01), (0.09, 0.014))):
        cables.cylinder((TRAY_X + dx, y0 + 0.02, TRAY_Z + 0.015 + r), "y", r, y1 - y0 - 0.04, n=10)
    cables.emit(stage, "/Room/Geo/CableTray/Cables", *cable)

    # Static colliders (guide purpose; physics.usda adds the APIs)
    collider_cube(stage, "/Room/Colliders/Floor/Box", (X_MIN, Y_MIN, -FLOOR_COL_T), (X_MAX, Y_MAX, 0.0))
    stage.GetRootLayer().Save()
    return stage


# ---- physics layer ------------------------------------------------------------
def physics_material(stage, name, static, dynamic, restitution=0.0):
    mat = UsdShade.Material.Define(stage, f"/Room/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    return mat


def build_physics(path):
    stage = new_stage(path)
    root = stage.OverridePrim("/Room")
    stage.SetDefaultPrim(root)
    floor_mat = physics_material(stage, "EpoxyFloor", 0.6, 0.6)
    pvc_mat = physics_material(stage, "PVCWall", 0.5, 0.5)
    for p, mat in [("/Room/Colliders/Floor/Box", floor_mat)] + \
            [(f"/Room/Colliders/Wall_{n}", pvc_mat) for n in ("PosX", "NegX", "PosY", "NegY")]:
        prim = stage.OverridePrim(p)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    # Material.Define promotes ancestors to 'def'; keep /Room a pure 'over'
    stage.GetRootLayer().GetPrimAtPath("/Room").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()


def build_root(path):
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Room").GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.GetReferences().AddReference("./geo.usda")
    root.GetPayloads().AddPayload("./physics.usda")
    stage.GetRootLayer().Save()
    return stage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--light-scale", type=float, default=LIGHT_SCALE,
                    help="multiplies every light intensity (default %(default)s)")
    args = ap.parse_args()
    for f in ("room.usda", "geo.usda", "physics.usda"):
        (HERE / f).unlink(missing_ok=True)
    geo = build_geo(HERE / "geo.usda", args.light_scale)
    build_physics(HERE / "physics.usda")
    build_root(HERE / "room.usda")
    print(f"room: {triangles(geo)} triangles, "
          f"{len([p for p in geo.Traverse() if p.IsA(UsdLux.RectLight)])} RectLights")


if __name__ == "__main__":
    main()
