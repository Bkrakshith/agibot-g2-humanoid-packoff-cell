"""Generate the background-clutter props asset (assets/props/*.usda).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/props/build.py

One asset, defaultPrim /Props (kind=assembly) at the cell origin. Each prop is its own
Xform /Props/<Name> (kind=component, ops translate + rotateZ) whose origin is the centre
of its footprint on the floor, so a randomiser only edits those two ops.
Visual meshes live under /Props/<Name>/Geo, guide-purpose box colliders under
/Props/<Name>/Colliders (CollisionAPI + friction added by physics.usda; all static).
"""

import math
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent

CRATE = (0.40, 0.60, 0.224)          # 600 x 400 x 224, long side along Y
CRATE_WALL = 0.010
PALLET = (1.00, 1.20, 0.144)         # EUR pallet, 1200 along Y
PALLET_LAYERS = 4
BIN = (0.58, 0.73, 1.07)             # 240 L wheelie bin (EN 840)
SINK = (0.40, 0.50, 1.10)            # depth X, width Y, height incl. splashback
POST = (0.20, 0.20, 1.40)            # base plate footprint, post height

# Default placements in the cell frame (x, y, yaw deg); room inner faces X[-4.4,5.4] Y[-4.9,4.9]
PLACEMENT = {
    "CrateStack": (-3.6, 3.9, 0.0),
    "PalletStack": (-3.3, -3.8, 0.0),
    "WasteBin": (4.6, 4.2, 0.0),
    "HandwashStation": (-4.08, 2.2, 0.0),
    "LightCurtainPost": (2.3, -3.1, 0.0),
}


class MeshBuilder:
    def __init__(self):
        self.pts, self.counts, self.idx = [], [], []

    def _face(self, ids):
        self.counts.append(len(ids))
        self.idx.extend(ids)

    def box(self, lo, hi):
        return self.frustum(lo, hi, hi[0] - lo[0], hi[1] - lo[1])

    def frustum(self, lo, hi, top_x, top_y):
        """Box from lo to hi whose top face is top_x x top_y (centred)."""
        b = len(self.pts)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        for z, (sx, sy) in ((lo[2], (hi[0] - lo[0], hi[1] - lo[1])), (hi[2], (top_x, top_y))):
            for j in (-1, 1):
                for i in (-1, 1):
                    self.pts.append((cx + i * sx / 2, cy + j * sy / 2, z))
        for f in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self._face([b + i for i in f])
        return self

    def cylinder(self, p0, axis, radius, length, n=12):
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

    def open_box(self, lo, hi, t):
        """Floor + 4 walls of thickness t (open top)."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        self.box((x0, y0, z0), (x1, y1, z0 + t))
        self.box((x0, y0, z0), (x0 + t, y1, z1))
        self.box((x1 - t, y0, z0), (x1, y1, z1))
        self.box((x0 + t, y0, z0), (x1 - t, y0 + t, z1))
        self.box((x0 + t, y1 - t, z0), (x1 - t, y1, z1))
        return self

    def emit(self, stage, path, look):
        material, color = look
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in self.pts]))
        mesh.CreateFaceVertexCountsAttr(self.counts)
        mesh.CreateFaceVertexIndicesAttr(self.idx)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return mesh


def make_material(stage, name, color, roughness, metallic=0.0):
    mat = UsdShade.Material.Define(stage, f"/Props/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, color


COLLIDERS = []   # (path, physics material name) filled while building geo


def collider(stage, path, lo, hi, pmat):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[b - a for a, b in zip(lo, hi)]))
    cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    COLLIDERS.append((path, pmat))


def new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    return stage


def prop_root(stage, name):
    x, y, yaw = PLACEMENT[name]
    xf = UsdGeom.Xform.Define(stage, f"/Props/{name}")
    Usd.ModelAPI(xf.GetPrim()).SetKind(Kind.Tokens.component)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
    xf.AddRotateZOp().Set(yaw)
    UsdGeom.Xform.Define(stage, f"/Props/{name}/Geo")
    UsdGeom.Xform.Define(stage, f"/Props/{name}/Colliders")
    return f"/Props/{name}"


def crate(stage, path, z0, look):
    sx, sy, sz = CRATE
    m = MeshBuilder().open_box((-sx / 2, -sy / 2, z0), (sx / 2, sy / 2, z0 + sz), CRATE_WALL)
    # top rim lip on the long sides
    m.box((-sx / 2, -sy / 2, z0 + sz - 0.02), (-sx / 2 + 0.02, sy / 2, z0 + sz))
    m.box((sx / 2 - 0.02, -sy / 2, z0 + sz - 0.02), (sx / 2, sy / 2, z0 + sz))
    return m.emit(stage, path, look)


def build_geo(path):
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Props")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Props/Looks")
    hdpe_blue = make_material(stage, "HDPEBlue", (0.08, 0.25, 0.65), 0.55)
    hdpe_green = make_material(stage, "HDPEGreen", (0.10, 0.40, 0.18), 0.6)
    wood = make_material(stage, "PalletWood", (0.66, 0.52, 0.34), 0.8)
    steel = make_material(stage, "Stainless", (0.75, 0.75, 0.76), 0.3, 1.0)
    rubber = make_material(stage, "RubberBlack", (0.03, 0.03, 0.03), 0.9)
    yellow = make_material(stage, "PostYellow", (0.95, 0.75, 0.05), 0.45)
    sensor = make_material(stage, "SensorDark", (0.02, 0.02, 0.025), 0.3)

    # CrateStack: 5 empty crates
    p = prop_root(stage, "CrateStack")
    for i in range(5):
        z0 = i * CRATE[2]
        crate(stage, f"{p}/Geo/Crate_{i:02d}", z0, hdpe_blue)
    collider(stage, f"{p}/Colliders/Stack", (-CRATE[0] / 2, -CRATE[1] / 2, 0),
             (CRATE[0] / 2, CRATE[1] / 2, 5 * CRATE[2]), "HDPE")

    # PalletStack: EUR-style pallet (22 top boards / 22 stringer boards / 78 blocks / 22 bottom)
    p = prop_root(stage, "PalletStack")
    px, py, pz = PALLET
    m = MeshBuilder()
    for k in range(3):   # bottom boards along Y
        x = -px / 2 + 0.05 + k * (px - 0.10) / 2
        m.box((x - 0.05, -py / 2, 0.0), (x + 0.05, py / 2, 0.022))
    for k in range(3):   # blocks 3x3
        for j in range(3):
            x = -px / 2 + 0.05 + k * (px - 0.10) / 2
            y = -py / 2 + 0.072 + j * (py - 0.144) / 2
            m.box((x - 0.05, y - 0.072, 0.022), (x + 0.05, y + 0.072, 0.100))
    for k in range(3):   # stringer boards along Y
        x = -px / 2 + 0.05 + k * (px - 0.10) / 2
        m.box((x - 0.05, -py / 2, 0.100), (x + 0.05, py / 2, 0.122))
    for j in range(5):   # top deck boards along X
        y = -py / 2 + 0.0725 + j * (py - 0.145) / 4
        m.box((-px / 2, y - 0.0725, 0.122), (px / 2, y + 0.0725, pz))
    m.emit(stage, f"{p}/Geo/Pallet", wood)
    collider(stage, f"{p}/Colliders/Pallet", (-px / 2, -py / 2, 0), (px / 2, py / 2, pz), "Wood")
    UsdGeom.Xform.Define(stage, f"{p}/Geo/Crates")
    n = 0
    for layer in range(PALLET_LAYERS):
        for ix in (-1, 1):
            for iy in (-1, 1):
                xf = UsdGeom.Xform.Define(stage, f"{p}/Geo/Crates/Crate_{n:02d}")
                xf.AddTranslateOp().Set(Gf.Vec3d(ix * CRATE[0] / 2, iy * CRATE[1] / 2, 0.0))
                crate(stage, f"{p}/Geo/Crates/Crate_{n:02d}/Mesh", pz + layer * CRATE[2], hdpe_blue)
                n += 1
    collider(stage, f"{p}/Colliders/Crates", (-2 * CRATE[0] / 2, -CRATE[1], pz),
             (CRATE[0], CRATE[1], pz + PALLET_LAYERS * CRATE[2]), "HDPE")

    # WasteBin: 240 L wheelie bin (body tapers to the bottom), lid, axle + wheels at -Y
    p = prop_root(stage, "WasteBin")
    bx, by, bz = BIN
    body_top = bz - 0.05
    m = MeshBuilder()
    m.frustum((-0.24, -0.29, 0.06), (0.24, 0.31, body_top), 0.56, 0.68)
    m.box((-bx / 2, -0.33, body_top), (bx / 2, by / 2, bz - 0.03))             # top rim
    m.box((-bx / 2 + 0.02, -by / 2 + 0.02, bz - 0.06), (bx / 2 - 0.02, -0.30, bz - 0.03))  # handle bar
    m.emit(stage, f"{p}/Geo/Body", hdpe_green)
    MeshBuilder().box((-bx / 2, -0.33, bz - 0.03), (bx / 2, by / 2, bz)).emit(stage, f"{p}/Geo/Lid", hdpe_green)
    w = MeshBuilder()
    for sx in (-1, 1):
        w.cylinder((sx * bx / 2 - (0.05 if sx > 0 else 0.0), -by / 2 + 0.1, 0.1), "x", 0.1, 0.05, n=16)
    w.cylinder((-bx / 2 + 0.05, -by / 2 + 0.1, 0.1), "x", 0.012, bx - 0.10, n=8)
    w.emit(stage, f"{p}/Geo/Wheels", rubber)
    collider(stage, f"{p}/Colliders/Body", (-bx / 2, -by / 2, 0), (bx / 2, by / 2, bz), "HDPE")

    # HandwashStation: stainless pedestal sink, splashback + knee valve at -X (wall side)
    p = prop_root(stage, "HandwashStation")
    sx, sy, sz = SINK
    basin_z0, basin_z1 = 0.75, 0.90
    m = MeshBuilder()
    m.box((-0.15, -0.15, 0.0), (0.15, 0.15, 0.01))                       # floor plate
    m.box((-0.09 - 0.05, -0.09, 0.01), (0.09 - 0.05, 0.09, basin_z0))   # pedestal column
    m.box((-0.13, -0.12, 0.45), (-0.06, 0.12, 0.55))                     # knee valve box
    m.emit(stage, f"{p}/Geo/Pedestal", steel)
    MeshBuilder().open_box((-sx / 2, -sy / 2, basin_z0), (sx / 2, sy / 2, basin_z1), 0.012) \
        .emit(stage, f"{p}/Geo/Basin", steel)
    MeshBuilder().box((-sx / 2, -sy / 2, basin_z1), (-sx / 2 + 0.03, sy / 2, sz)) \
        .emit(stage, f"{p}/Geo/Splashback", steel)
    tap = MeshBuilder()
    tap.cylinder((-sx / 2 + 0.06, 0.0, basin_z1), "z", 0.012, 0.16, n=10)
    tap.cylinder((-sx / 2 + 0.06, 0.0, basin_z1 + 0.15), "x", 0.01, 0.14, n=10)
    tap.emit(stage, f"{p}/Geo/Tap", steel)
    collider(stage, f"{p}/Colliders/Pedestal", (-0.15, -0.15, 0.0), (0.15, 0.15, basin_z0), "Stainless")
    collider(stage, f"{p}/Colliders/Basin", (-sx / 2, -sy / 2, basin_z0), (sx / 2, sy / 2, basin_z1), "Stainless")
    collider(stage, f"{p}/Colliders/Splashback", (-sx / 2, -sy / 2, basin_z1), (-sx / 2 + 0.03, sy / 2, sz), "Stainless")

    # LightCurtainPost: yellow 1.4 m post on a base plate, dark sensor strip on the +Y face
    p = prop_root(stage, "LightCurtainPost")
    bx, by, bz = POST
    m = MeshBuilder()
    m.box((-bx / 2, -by / 2, 0.0), (bx / 2, by / 2, 0.012))
    m.box((-0.03, -0.03, 0.012), (0.03, 0.03, bz))
    m.emit(stage, f"{p}/Geo/Post", yellow)
    MeshBuilder().box((-0.015, 0.03, 0.15), (0.015, 0.036, bz - 0.05)).emit(stage, f"{p}/Geo/SensorStrip", sensor)
    collider(stage, f"{p}/Colliders/BasePlate", (-bx / 2, -by / 2, 0.0), (bx / 2, by / 2, 0.012), "PaintedSteel")
    collider(stage, f"{p}/Colliders/Post", (-0.03, -0.03, 0.012), (0.03, 0.036, bz), "PaintedSteel")

    stage.GetRootLayer().Save()
    return stage


PHYSICS_MATERIALS = {"HDPE": 0.5, "Wood": 0.5, "Stainless": 0.3, "PaintedSteel": 0.4}


def build_physics(path):
    stage = new_stage(path)
    root = stage.OverridePrim("/Props")
    stage.SetDefaultPrim(root)
    mats = {}
    for name, mu in PHYSICS_MATERIALS.items():
        mat = UsdShade.Material.Define(stage, f"/Props/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(mu)
        api.CreateDynamicFrictionAttr(mu)
        api.CreateRestitutionAttr(0.0)
        mats[name] = mat
    for cpath, pm in COLLIDERS:
        prim = stage.OverridePrim(cpath)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mats[pm], UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().GetPrimAtPath("/Props").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()


def build_root(path):
    stage = new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/Props").GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.assembly)
    root.GetReferences().AddReference("./geo.usda")
    root.GetPayloads().AddPayload("./physics.usda")
    stage.GetRootLayer().Save()


def main():
    for f in ("props.usda", "geo.usda", "physics.usda"):
        (HERE / f).unlink(missing_ok=True)
    COLLIDERS.clear()
    geo = build_geo(HERE / "geo.usda")
    build_physics(HERE / "physics.usda")
    build_root(HERE / "props.usda")
    for prop in PLACEMENT:
        tris = sum(sum(c - 2 for c in UsdGeom.Mesh(q).GetFaceVertexCountsAttr().Get())
                   for q in Usd.PrimRange(geo.GetPrimAtPath(f"/Props/{prop}")) if q.IsA(UsdGeom.Mesh))
        print(f"{prop}: {tris} triangles")


if __name__ == "__main__":
    main()
