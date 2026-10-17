"""Generate the three Euro-footprint crate assets (pxr only).

Run:
    ../.venv-tools/bin/python assets/crates/build.py

Writes, for each variant in VARIANTS:
    assets/crates/<name>/<name>.usda   root: defaultPrim /<Prim> (Xform, kind=component),
                                        references ./geo.usda, payloads ./physics.usda
    assets/crates/<name>/geo.usda      visual meshes, UsdPreviewSurface looks, collider boxes
                                        (purpose=guide), InteriorVolume (invisible, guide)
    assets/crates/<name>/physics.usda  overs only (+ PhysicsMaterials): RigidBodyAPI + MassAPI on
                                        the root, CollisionAPI + PhysX contact offset on boxes
    assets/crates/<name>/README.md     spec vs modelled table

Frame: metres, Z up, origin at the centre of the base footprint (Z = 0), long side along Y.
The .usda/README files are outputs; edit this script, not them.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
MM = 0.001

WALL_T = 4 * MM            # collider + visual wall thickness
RIM_H = 20 * MM            # top rim band height
SKIRT_H = 20 * MM          # bottom skirt band height
RIB_T = 4 * MM             # external rib thickness
PLATE_T = 4 * MM           # visual base plate thickness
SLOT_W, SLOT_H = 120 * MM, 30 * MM   # handle slot in each short end
SLOT_TOP_BELOW_RIM = 6 * MM
CONTACT_OFFSET = 0.003     # spec 2-5 mm
REST_OFFSET = 0.0
HDPE_FRICTION = 0.5
ARM_R = 4 * MM             # bale-arm wire radius (8 mm wire)

# displayColor presets (linear RGB). Stage 6 randomises among these.
COLOURS = {
    "blue": (0.02, 0.15, 0.55),
    "green": (0.05, 0.35, 0.10),
    "grey": (0.35, 0.36, 0.37),
    "black": (0.02, 0.02, 0.02),
}
DEFAULT_COLOUR = "blue"
ZINC = (0.62, 0.63, 0.64)

# All lengths in mm. Outer = (X, Y, Z) with Y the 600 mm long side.
VARIANTS = {
    "stacking_224": dict(
        prim="CrateStacking224",
        title="Perforated HDPE stacking crate 600x400x224",
        outer=(400, 600, 224), inner=(368, 568, 210), mass=1.72,
        perforated=True, bale_arms=False,
        sources=dict(outer="Transoplast catalogue", inner="Transoplast catalogue",
                     mass="Transoplast catalogue"),
        notes=["Base perforation is a visual lattice of 8 mm square holes on a 16 mm pitch; "
               "the collider base is a solid slab (bags cannot fall through anyway)."],
    ),
    "balearm_199": dict(
        prim="CrateBaleArm199",
        title="Supermarket bale-arm crate 600x400x199",
        outer=(400, 600, 199), inner=(366, 566, 185), mass=1.5,
        perforated=False, bale_arms=True,
        sources=dict(outer="BiGDUG catalogue",
                     inner="ESTIMATE: catalogue gives outer only; 17 mm wall+rib, 14 mm base",
                     mass="ESTIMATE: typical HDPE bale-arm crate of this size"),
        notes=["Real bale-arm crates taper for nesting; modelled with straight walls at the "
               "estimated inner size.",
               "Bale arms (8 mm zinc wire, U-shaped) are visual only, lying folded on the rim of "
               "the long sides with the crossbar across the opening; they sink 4 mm into a "
               "rim groove so they do not raise the 199 mm height. Prims: Geo/BaleArmNegY, "
               "Geo/BaleArmPosY. No collision (add one if stacking on arms is ever simulated)."],
    ),
    "deep_325": dict(
        prim="CrateDeep325",
        title="Deep produce tote 600x400x325",
        outer=(400, 600, 325), inner=(366, 566, 284), mass=2.0,
        perforated=False, bale_arms=False,
        sources=dict(outer="Utz catalogue",
                     inner="X/Y ESTIMATE (17 mm wall+rib); usable height 284 mm from Utz",
                     mass="ESTIMATE: typical HDPE 600x400x325 tote"),
        notes=["Used on the angled sort rack with the open top (+Z in the asset frame) facing "
               "the robot; the scene applies the tilt, the asset stays upright.",
               "Usable height 284 mm is modelled as a 41 mm solid base collider so the interior "
               "matches the catalogue; the visual base is a plate on support ribs."],
    ),
}


# ---------------------------------------------------------------- mesh helpers
class MeshBuilder:
    """Accumulates quads/polys into one mesh."""

    def __init__(self):
        self.points: list[Gf.Vec3f] = []
        self.counts: list[int] = []
        self.indices: list[int] = []

    def box(self, lo, hi):
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        if x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9 or z1 - z0 <= 1e-9:
            return
        b = len(self.points)
        for z in (z0, z1):
            for y in (y0, y1):
                for x in (x0, x1):
                    self.points.append(Gf.Vec3f(x, y, z))
        # vertex index = b + ix + 2*iy + 4*iz ; outward CCW quads
        for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
                  (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.counts.append(4)
            self.indices.extend(b + i for i in q)

    def cylinder(self, p0, p1, r, sides=8):
        p0, p1 = Gf.Vec3d(*p0), Gf.Vec3d(*p1)
        axis = (p1 - p0).GetNormalized()
        ref = Gf.Vec3d(0, 0, 1) if abs(axis[2]) < 0.9 else Gf.Vec3d(1, 0, 0)
        u = Gf.Cross(axis, ref).GetNormalized()
        v = Gf.Cross(axis, u)
        b = len(self.points)
        for end in (p0, p1):
            for k in range(sides):
                a = 2 * math.pi * k / sides
                self.points.append(Gf.Vec3f(end + r * (math.cos(a) * u + math.sin(a) * v)))
        for k in range(sides):
            k1 = (k + 1) % sides
            self.counts.append(4)
            self.indices.extend((b + k, b + k1, b + sides + k1, b + sides + k))
        # end caps
        self.counts.extend((sides, sides))
        self.indices.extend(b + k for k in reversed(range(sides)))
        self.indices.extend(b + sides + k for k in range(sides))

    def sphere_joint(self, p, r):
        """Cheap joint filler at polyline corners: a small cube."""
        self.box(tuple(c - r for c in p), tuple(c + r for c in p))

    def write(self, stage, path):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(self.points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(self.counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(self.indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        return mesh

    @property
    def triangles(self):
        return sum(c - 2 for c in self.counts)


def define_box_cube(stage, path, lo, hi, purpose=UsdGeom.Tokens.guide):
    """A UsdGeom.Cube (size 1) translated/scaled to the AABB lo..hi."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    centre = Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)])
    size = Gf.Vec3f(*[b - a for a, b in zip(lo, hi)])
    xf = UsdGeom.XformCommonAPI(cube)
    xf.SetTranslate(centre)
    xf.SetScale(size)
    cube.CreatePurposeAttr(purpose)
    return cube


def new_stage(path: Path, default_prim: str):
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetMetadata("comment", "Generated by assets/crates/build.py - do not edit by hand.")
    return stage


# ---------------------------------------------------------------- dimensions
def dims(v):
    W, L, H = (d * MM for d in v["outer"])
    Wi, Li, Hi = (d * MM for d in v["inner"])
    return dict(W=W, L=L, H=H, Wi=Wi, Li=Li, Hi=Hi, base_top=H - Hi,
                xi=Wi / 2, yi=Li / 2, xo=W / 2, yo=L / 2)


def collider_boxes(d):
    """name -> (lo, hi). Base slab + 4 thin walls at the inner faces + 4 top rim bands
    reaching the outer dims (so stacking / guide-rail contact uses the true envelope)."""
    t = WALL_T
    H, bt = d["H"], d["base_top"]
    xi, yi, xo, yo = d["xi"], d["yi"], d["xo"], d["yo"]
    rz = H - RIM_H
    return {
        "Base": ((-xo, -yo, 0.0), (xo, yo, bt)),
        "WallNegX": ((-xi - t, -yi - t, bt), (-xi, yi + t, H)),
        "WallPosX": ((xi, -yi - t, bt), (xi + t, yi + t, H)),
        "WallNegY": ((-xi, -yi - t, bt), (xi, -yi, H)),
        "WallPosY": ((-xi, yi, bt), (xi, yi + t, H)),
        "RimNegX": ((-xo, -yo, rz), (-xi - t, yo, H)),
        "RimPosX": ((xi + t, -yo, rz), (xo, yo, H)),
        "RimNegY": ((-xi - t, -yo, rz), (xi + t, -yi - t, H)),
        "RimPosY": ((-xi - t, yi + t, rz), (xi + t, yo, H)),
    }


# ---------------------------------------------------------------- visual geometry
def build_visuals(v, d):
    t = WALL_T
    H, bt = d["H"], d["base_top"]
    xi, yi, xo, yo = d["xi"], d["yi"], d["xo"], d["yo"]
    plate_z0 = bt - PLATE_T
    parts = {}

    # Walls: thin inner shell; short ends (+-Y) have a handle slot.
    walls = MeshBuilder()
    walls.box((-xi - t, -yi - t, plate_z0), (-xi, yi + t, H))
    walls.box((xi, -yi - t, plate_z0), (xi + t, yi + t, H))
    slot_top = H - RIM_H - SLOT_TOP_BELOW_RIM
    slot_bot = slot_top - SLOT_H
    sx = SLOT_W / 2
    for y0, y1 in ((-yi - t, -yi), (yi, yi + t)):
        walls.box((-xi, y0, plate_z0), (xi, y1, slot_bot))      # below slot
        walls.box((-xi, y0, slot_top), (xi, y1, H))             # above slot
        walls.box((-xi, y0, slot_bot), (-sx, y1, slot_top))     # left of slot
        walls.box((sx, y0, slot_bot), (xi, y1, slot_top))       # right of slot
    parts["Walls"] = walls

    # Rim: top band ring out to the outer dims.
    rim = MeshBuilder()
    rz = H - RIM_H
    # (bale arms, when present, sink 4 mm into the long-side rims: a visual groove)
    rim.box((-xo, -yo, rz), (-xi - t, yo, H))
    rim.box((xi + t, -yo, rz), (xo, yo, H))
    rim.box((-xi - t, -yo, rz), (xi + t, -yi - t, H))
    rim.box((-xi - t, yi + t, rz), (xi + t, yo, H))
    # bottom skirt ring at the outer footprint
    rim.box((-xo, -yo, 0.0), (-xo + RIB_T, yo, SKIRT_H))
    rim.box((xo - RIB_T, -yo, 0.0), (xo, yo, SKIRT_H))
    rim.box((-xo + RIB_T, -yo, 0.0), (xo - RIB_T, -yo + RIB_T, SKIRT_H))
    rim.box((-xo + RIB_T, yo - RIB_T, 0.0), (xo - RIB_T, yo, SKIRT_H))
    parts["Rim"] = rim

    # Ribs: vertical external ribs between wall and outer face, plus a mid-height band.
    ribs = MeshBuilder()
    z0, z1 = SKIRT_H, rz
    n_long = 7
    for k in range(n_long):
        y = -yi + (k + 0.5) * (2 * yi) / n_long
        for x0, x1 in ((-xo, -xi - t), (xi + t, xo)):
            ribs.box((x0, y - RIB_T / 2, z0), (x1, y + RIB_T / 2, z1))
    for x in (-xi + 0.03, -sx - 0.03, sx + 0.03, xi - 0.03):
        for y0, y1 in ((-yo, -yi - t), (yi + t, yo)):
            ribs.box((x - RIB_T / 2, y0, z0), (x + RIB_T / 2, y1, z1))
    zb = (z0 + min(z1, slot_bot)) / 2
    for x0, x1 in ((-xo, -xi - t), (xi + t, xo)):
        ribs.box((x0, -yo, zb - RIB_T / 2), (x1, yo, zb + RIB_T / 2))
    for y0, y1 in ((-yo, -yi - t), (yi + t, yo)):
        ribs.box((-xi - t, y0, zb - RIB_T / 2), (xi + t, y1, zb + RIB_T / 2))
    # corner posts close the outer corners
    for sxn in (-1, 1):
        for syn in (-1, 1):
            xa, xb = sorted((sxn * (xi + t), sxn * xo))
            ya, yb = sorted((syn * (yi + t), syn * yo))
            ribs.box((xa, ya, 0.0), (xb, yb, H - RIM_H))
    parts["Ribs"] = ribs

    # Base: plate (perforated lattice or solid) on under-floor support ribs.
    base = MeshBuilder()
    if v["perforated"]:
        hole, pitch = 8 * MM, 16 * MM
        bar = pitch - hole
        nx = int((2 * xi) // pitch)
        ny = int((2 * yi) // pitch)
        mx = (2 * xi - nx * pitch + bar) / 2       # border so the lattice is centred
        my = (2 * yi - ny * pitch + bar) / 2
        # border frame
        base.box((-xi, -yi, plate_z0), (-xi + mx, yi, bt))
        base.box((xi - mx, -yi, plate_z0), (xi, yi, bt))
        base.box((-xi + mx, -yi, plate_z0), (xi - mx, -yi + my, bt))
        base.box((-xi + mx, yi - my, plate_z0), (xi - mx, yi, bt))
        x_start, y_start = -xi + mx, -yi + my
        for i in range(nx - 1):          # bars between hole columns
            x = x_start + i * pitch + hole
            base.box((x, y_start, plate_z0), (x + bar, yi - my, bt))
        for j in range(ny - 1):          # bars between hole rows
            y = y_start + j * pitch + hole
            base.box((x_start, y, plate_z0), (xi - mx, y + bar, bt))
    else:
        base.box((-xi, -yi, plate_z0), (xi, yi, bt))
    for x in (-xi / 2, 0.0, xi / 2):
        base.box((x - RIB_T / 2, -yi, 0.0), (x + RIB_T / 2, yi, plate_z0))
    for y in (-yi * 2 / 3, -yi / 3, 0.0, yi / 3, yi * 2 / 3):
        base.box((-xi, y - RIB_T / 2, 0.0), (xi, y + RIB_T / 2, plate_z0))
    parts["Base"] = base

    arms = {}
    if v["bale_arms"]:
        zc = H - ARM_R
        xr = (xi + t + xo) / 2               # legs lie on the middle of the long-side rims
        for name, s in (("BaleArmNegY", -1), ("BaleArmPosY", 1)):
            m = MeshBuilder()
            pivot_y, cross_y = s * (yi - 0.010), s * 0.13
            pts = [(-xr, pivot_y, zc), (-xr, cross_y, zc), (xr, cross_y, zc), (xr, pivot_y, zc)]
            for a, b in zip(pts, pts[1:]):
                m.cylinder(a, b, ARM_R)
            for p in pts[1:3]:
                m.sphere_joint(p, ARM_R * 0.95)
            arms[name] = m
    return parts, arms


# ---------------------------------------------------------------- layers
def preview_material(stage, path, colour, roughness, metallic, use_primvar):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04))
    diffuse = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if use_primvar:
        rd = UsdShade.Shader.Define(stage, path.AppendChild("DisplayColorReader"))
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*colour))
        diffuse.ConnectToSource(rd.ConnectableAPI(), "result", UsdShade.AttributeType.Output,
                                Sdf.ValueTypeNames.Float3)
    else:
        diffuse.Set(Gf.Vec3f(*colour))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def write_geo(folder, v, d):
    stage = new_stage(folder / "geo.usda", v["prim"])
    root = UsdGeom.Xform.Define(stage, f"/{v['prim']}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    rp = root.GetPath()

    looks = UsdGeom.Scope.Define(stage, rp.AppendChild("Looks"))
    hdpe = preview_material(stage, looks.GetPath().AppendChild("HDPE"),
                            COLOURS[DEFAULT_COLOUR], roughness=0.55, metallic=0.0,
                            use_primvar=True)
    zinc = preview_material(stage, looks.GetPath().AppendChild("ZincWire"),
                            ZINC, roughness=0.35, metallic=1.0, use_primvar=False)

    geo = UsdGeom.Xform.Define(stage, rp.AppendChild("Geo"))
    # Single colour knob: constant primvar inherited by every HDPE mesh below Geo.
    pv = UsdGeom.PrimvarsAPI(geo).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                UsdGeom.Tokens.constant)
    pv.Set(Vt.Vec3fArray([Gf.Vec3f(*COLOURS[DEFAULT_COLOUR])]))
    UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(hdpe)

    parts, arms = build_visuals(v, d)
    tris = 0
    for name, mb in parts.items():
        mb.write(stage, geo.GetPath().AppendChild(name))
        tris += mb.triangles
    for name, mb in arms.items():
        m = mb.write(stage, geo.GetPath().AppendChild(name))
        m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*ZINC)]))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(zinc)
        tris += mb.triangles

    cols = UsdGeom.Scope.Define(stage, rp.AppendChild("Colliders"))
    for name, (lo, hi) in collider_boxes(d).items():
        define_box_cube(stage, cols.GetPath().AppendChild(name), lo, hi)

    iv = define_box_cube(stage, rp.AppendChild("InteriorVolume"),
                         (-d["xi"], -d["yi"], d["base_top"]), (d["xi"], d["yi"], d["H"]))
    iv.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    iv.GetPrim().SetDocumentation("Inner volume of the crate for 'bag fully inside' tests. "
                                  "Invisible, guide purpose, no collision.")
    stage.GetRootLayer().Save()
    return tris


def write_physics(folder, v, d):
    stage = new_stage(folder / "physics.usda", v["prim"])
    rp = Sdf.Path(f"/{v['prim']}")
    root = stage.OverridePrim(rp)
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(v["mass"])

    mats = UsdGeom.Scope.Define(stage, rp.AppendChild("PhysicsMaterials"))
    mat = UsdShade.Material.Define(stage, mats.GetPath().AppendChild("HDPE"))
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(HDPE_FRICTION)
    pm.CreateDynamicFrictionAttr(HDPE_FRICTION)
    pm.CreateRestitutionAttr(0.0)

    stage.OverridePrim(rp.AppendChild("Colliders"))
    for name in collider_boxes(d):
        p = stage.OverridePrim(rp.AppendChild("Colliders").AppendChild(name))
        UsdPhysics.CollisionAPI.Apply(p)
        # PhysxSchema is not shipped with usd-core: author the applied-schema token and raw
        # attributes by name; Isaac Sim's PhysxSchema picks them up on load.
        p.AddAppliedSchema("PhysxCollisionAPI")
        p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
        p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    # Defining PhysicsMaterials promoted the root to `def`; this layer only overrides it.
    stage.GetRootLayer().GetPrimAtPath(rp).specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()


def write_root(folder, name, v):
    stage = new_stage(folder / f"{name}.usda", v["prim"])
    root = UsdGeom.Xform.Define(stage, f"/{v['prim']}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.GetPrim().GetReferences().AddReference("./geo.usda")
    root.GetPrim().GetPayloads().AddPayload("./physics.usda")
    stage.GetRootLayer().Save()


def write_readme(folder, name, v, d, tris):
    src = v["sources"]
    W, L, H = v["outer"]
    Wi, Li, Hi = v["inner"]
    rows = [
        ("Outer X x Y x Z", f"{W} x {L} x {H} mm", f"{W} x {L} x {H} mm (visual bbox)", src["outer"]),
        ("Inner X x Y x Z", f"{Wi} x {Li} x {Hi} mm", f"{Wi} x {Li} x {Hi} mm (collider inner faces)", src["inner"]),
        ("Mass", f"{v['mass']} kg", f"{v['mass']} kg (MassAPI on root)", src["mass"]),
        ("Body", "movable", "RigidBodyAPI + MassAPI on /" + v["prim"], "spec"),
        ("Collider", "compound boxes, not a hull",
         f"{len(collider_boxes(d))} boxes: Base, WallNegX/PosX/NegY/PosY (4 mm), RimNegX/PosX/NegY/PosY", "spec"),
        ("Friction", "HDPE 0.5", "0.5 static/dynamic, restitution 0", "CONVENTIONS.md"),
        ("Contact offset", "2-5 mm", "3 mm, rest offset 0 (raw `physxCollision:*` attrs)",
         "spec"),
        ("Colour", "matte-satin HDPE", f"`primvars:displayColor` on Geo, default {DEFAULT_COLOUR}; roughness 0.55", "spec"),
        ("Triangles", "< 40k", f"{tris}", "build.py"),
    ]
    lines = [f"# {v['title']}  (`{name}`)", "",
             "Generated by `assets/crates/build.py`; do not edit the .usda files by hand.", "",
             "| item | spec | modelled | source |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {s} |" for a, b, c, s in rows]
    lines += ["", "## Prim paths", "",
              f"- `/{v['prim']}`: rigid body root (origin = centre of base footprint, Z=0, long side +Y)",
              f"- `/{v['prim']}/Geo/{{Base,Walls,Ribs,Rim}}`: visual meshes (HDPE look)",
              f"- `/{v['prim']}/Colliders/*`: collider boxes (UsdGeom.Cube, purpose=guide)",
              f"- `/{v['prim']}/InteriorVolume`: invisible guide box = inner volume, no collision",
              f"- `/{v['prim']}/PhysicsMaterials/HDPE`: physics material (physics.usda)", "",
              "## Colour", "",
              f"The HDPE look reads the constant primvar `primvars:displayColor` authored on "
              f"`/{v['prim']}/Geo` (inherited by all HDPE meshes; the shader's "
              "`UsdPrimvarReader_float3` fallback is the same blue). To switch, set it in the scene:",
              "", "```python",
              f"UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('<crate>/Geo')).GetPrimvar('displayColor')"
              ".Set([(r, g, b)])", "```", "",
              "| preset | linear RGB |", "|---|---|"]
    lines += [f"| {k} | {c} |" for k, c in COLOURS.items()]
    lines += ["", "Stage 6 replaces the preview look with MDL; keep reading the same primvar.",
              "", "## Notes", ""]
    lines += [f"- {n}" for n in v["notes"]]
    lines += ["- `PhysxSchema` is not in usd-core, so `PhysxCollisionAPI` is added as an "
              "applied-schema token with raw `physxCollision:contactOffset`/`restOffset` "
              "attributes; Isaac Sim reads them natively.",
              "- Mass is spread by PhysX over the collider boxes (no explicit inertia)."]
    (folder / "README.md").write_text("\n".join(lines) + "\n")


def build(name, v):
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    d = dims(v)
    tris = write_geo(folder, v, d)
    write_physics(folder, v, d)
    write_root(folder, name, v)
    write_readme(folder, name, v, d, tris)
    return tris


def main():
    for name, v in VARIANTS.items():
        tris = build(name, v)
        print(f"{name}: {tris} triangles")


if __name__ == "__main__":
    main()
