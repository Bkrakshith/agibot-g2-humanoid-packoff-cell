"""Small pxr-only toolkit shared by the pack-off line generators.

Used by assets/{cartons,case_erector,case_sealer,palletiser,floor_bin}/build.py (each adds
assets/cartons to sys.path). Follows assets/CONVENTIONS.md: metres, Z up, kg; root layer
references ./geo.usda and payloads ./physics.usda; colliders are UsdGeom.Cube prims with
purpose=guide in geo.usda that physics.usda turns into colliders with `over`s only.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

CONTACT_OFFSET = 0.003
REST_OFFSET = 0.0

# Physics materials (static, dynamic friction). CONVENTIONS.md: stainless 0.3, HDPE 0.5,
# rubber/pads 1.0; the rest are assumptions documented in each README.
PHYS_MATERIALS = {
    "Stainless": (0.3, 0.3),
    "HDPE": (0.5, 0.5),
    "Rubber": (1.0, 1.0),
    "Cardboard": (0.5, 0.45),      # ASSUMPTION: corrugated board on steel/plastic
    "PaintedSteel": (0.4, 0.4),    # ASSUMPTION (same as props/LightCurtainPost)
    "Wood": (0.5, 0.5),            # ASSUMPTION (same as props/PalletStack)
    "Polycarbonate": (0.4, 0.35),  # ASSUMPTION
}


# ---------------------------------------------------------------- meshes
class MeshBuilder:
    """Accumulates polygons into one mesh (optionally with a per-face colour)."""

    def __init__(self):
        self.points: list[Gf.Vec3f] = []
        self.counts: list[int] = []
        self.indices: list[int] = []

    # generic parallelepiped: centre + three (unit) axes + half sizes
    def obox(self, centre, axes, half):
        c = Gf.Vec3d(*centre)
        ax = [Gf.Vec3d(*a) for a in axes]
        b = len(self.points)
        for iz in (-1, 1):
            for iy in (-1, 1):
                for ix in (-1, 1):
                    p = c + ax[0] * (ix * half[0]) + ax[1] * (iy * half[1]) + ax[2] * (iz * half[2])
                    self.points.append(Gf.Vec3f(p))
        for q in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
                  (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.counts.append(4)
            self.indices.extend(b + i for i in q)

    def box(self, lo, hi):
        if any(h - l <= 1e-9 for l, h in zip(lo, hi)):
            return
        c = [(l + h) / 2 for l, h in zip(lo, hi)]
        half = [(h - l) / 2 for l, h in zip(lo, hi)]
        self.obox(c, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), half)

    def cylinder(self, p0, p1, r, sides=16, caps=True, r1=None):
        """Cylinder (or cone frustum with r1) from p0 to p1."""
        p0, p1 = Gf.Vec3d(*p0), Gf.Vec3d(*p1)
        r1 = r if r1 is None else r1
        axis = (p1 - p0).GetNormalized()
        ref = Gf.Vec3d(0, 0, 1) if abs(axis[2]) < 0.9 else Gf.Vec3d(1, 0, 0)
        u = Gf.Cross(axis, ref).GetNormalized()
        v = Gf.Cross(axis, u)
        b = len(self.points)
        for end, rr in ((p0, r), (p1, r1)):
            for k in range(sides):
                a = 2 * math.pi * k / sides
                self.points.append(Gf.Vec3f(end + rr * (math.cos(a) * u + math.sin(a) * v)))
        for k in range(sides):
            k1 = (k + 1) % sides
            self.counts.append(4)
            self.indices.extend((b + k, b + k1, b + sides + k1, b + sides + k))
        if caps:
            self.counts.extend((sides, sides))
            self.indices.extend(b + k for k in reversed(range(sides)))
            self.indices.extend(b + sides + k for k in range(sides))

    def sphere(self, c, r, seg=12, rings=6):
        c = Gf.Vec3d(*c)
        b = len(self.points)
        self.points.append(Gf.Vec3f(c + Gf.Vec3d(0, 0, -r)))
        for i in range(1, rings):
            phi = -math.pi / 2 + math.pi * i / rings
            for k in range(seg):
                t = 2 * math.pi * k / seg
                self.points.append(Gf.Vec3f(c + Gf.Vec3d(r * math.cos(phi) * math.cos(t),
                                                         r * math.cos(phi) * math.sin(t),
                                                         r * math.sin(phi))))
        top = len(self.points)
        self.points.append(Gf.Vec3f(c + Gf.Vec3d(0, 0, r)))
        ring = lambda i, k: b + 1 + (i - 1) * seg + (k % seg)
        for k in range(seg):
            self.counts.append(3)
            self.indices.extend((b, ring(1, k + 1), ring(1, k)))
        for i in range(1, rings - 1):
            for k in range(seg):
                self.counts.append(4)
                self.indices.extend((ring(i, k), ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k)))
        for k in range(seg):
            self.counts.append(3)
            self.indices.extend((ring(rings - 1, k), ring(rings - 1, k + 1), top))

    def capsule(self, p0, p1, r, sides=16):
        self.cylinder(p0, p1, r, sides, caps=False)
        self.sphere(p0, r, sides, 6)
        self.sphere(p1, r, sides, 6)

    @property
    def triangles(self):
        return sum(c - 2 for c in self.counts)

    def write(self, stage, path, material=None, colour=None, jitter=0.0, seed=0, purpose=None):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray(self.points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(self.counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(self.indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        if colour is not None:
            if jitter > 0:
                rnd = random.Random(seed)
                cols = []
                for _ in self.counts:
                    s = 1.0 + rnd.uniform(-jitter, jitter)
                    cols.append(Gf.Vec3f(*(min(1.0, c * s) for c in colour)))
                pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform)
                pv.Set(Vt.Vec3fArray(cols))
            else:
                mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        if material is not None:
            bind(mesh, material)
        if purpose:
            mesh.CreatePurposeAttr(purpose)
        return mesh


# ---------------------------------------------------------------- stage / looks
def new_stage(path: Path, generator: str):
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetMetadata("comment", f"Generated by {generator} - do not edit by hand.")
    return stage


def root_xform(stage, name, kind=Kind.Tokens.component):
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(kind)
    return root


def preview_material(stage, path, colour, roughness=0.5, metallic=0.0, opacity=1.0,
                     ior=None, use_primvar=False, emissive=None):
    path = Sdf.Path(str(path))
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if ior is not None:
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
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


def bind(prim_or_schema, mat):
    prim = prim_or_schema.GetPrim() if hasattr(prim_or_schema, "GetPrim") else prim_or_schema
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def xform(stage, path, translate=(0, 0, 0), rotate=None):
    x = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(x)
    if any(translate):
        api.SetTranslate(Gf.Vec3d(*translate))
    if rotate is not None:
        api.SetRotate(Gf.Vec3f(*rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return x


def collider_box(stage, path, lo, hi, rotate=None):
    """Guide-purpose UsdGeom.Cube (size 1) spanning the AABB lo..hi (rotate: XYZ degrees
    about the box centre)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)]))
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
    if rotate is not None:
        api.SetRotate(Gf.Vec3f(*rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale(Gf.Vec3f(*[b - a for a, b in zip(lo, hi)]))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return cube


def guide_point(stage, path, translate, doc):
    x = xform(stage, path, translate)
    x.GetPrim().SetDocumentation(doc)
    return x


# ---------------------------------------------------------------- layers
def write_root(folder: Path, file_stem: str, prim: str, generator: str):
    stage = new_stage(folder / f"{file_stem}.usda", generator)
    root = root_xform(stage, prim)
    root.GetPrim().GetReferences().AddReference("./geo.usda")
    root.GetPrim().GetPayloads().AddPayload("./physics.usda")
    stage.GetRootLayer().Save()


def write_physics(folder: Path, prim: str, generator: str, colliders: dict,
                  rigid_mass: float | None = None, extra=None):
    """physics.usda: `over`s only (+ PhysicsMaterials defs).

    colliders: {prim path (str/Sdf.Path, absolute): physics material name}.
    rigid_mass: RigidBodyAPI + MassAPI on the root when given, otherwise everything is static.
    extra(stage, root_path): optional hook for more overs.
    """
    stage = new_stage(folder / "physics.usda", generator)
    rp = Sdf.Path(f"/{prim}")
    root = stage.OverridePrim(rp)
    stage.SetDefaultPrim(root)
    if rigid_mass is not None:
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(rigid_mass)

    UsdGeom.Scope.Define(stage, rp.AppendChild("PhysicsMaterials"))
    mats = {}
    for name in sorted(set(colliders.values())):
        sf, df = PHYS_MATERIALS[name]
        m = UsdShade.Material.Define(stage, rp.AppendChild("PhysicsMaterials").AppendChild(name))
        pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        pm.CreateStaticFrictionAttr(sf)
        pm.CreateDynamicFrictionAttr(df)
        pm.CreateRestitutionAttr(0.0)
        mats[name] = m

    for path, mname in colliders.items():
        p = stage.OverridePrim(Sdf.Path(str(path)))
        UsdPhysics.CollisionAPI.Apply(p)
        # PhysxSchema is not in usd-core: applied-schema token + raw attrs (Isaac Sim reads them)
        p.AddAppliedSchema("PhysxCollisionAPI")
        p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
        p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(
            mats[mname], UsdShade.Tokens.weakerThanDescendants, "physics")
    if extra:
        extra(stage, rp)
    # Defining PhysicsMaterials promoted ancestors to `def`; this layer only overrides them.
    layer = stage.GetRootLayer()

    def demote(spec):
        if spec.name != "PhysicsMaterials":
            spec.specifier = Sdf.SpecifierOver
            for c in spec.nameChildren:
                demote(c)

    for top in layer.rootPrims:
        demote(top)
    layer.Save()


def count_triangles(stage):
    return sum(sum(c - 2 for c in (UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []))
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))


def readme_table(rows):
    lines = ["| item | spec | modelled | source / assumption |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return lines


# ---------------------------------------------------------------- shared machine parts
def rollers(y0, y1, top, half_len, d=0.050, pitch=0.075, sides=12, x0=0.0):
    """Rollers with axis X, tops at `top`, spread over [y0, y1] (pitch centred in the span)."""
    mb = MeshBuilder()
    n = max(1, int((y1 - y0 - d) // pitch) + 1)
    span = (n - 1) * pitch
    ys = [(y0 + y1) / 2 - span / 2 + i * pitch for i in range(n)]
    zc = top - d / 2
    for y in ys:
        mb.cylinder((x0 - half_len, y, zc), (x0 + half_len, y, zc), d / 2, sides)
    return mb


LAMP_COLOURS = {"Green": (0.04, 0.55, 0.08), "Amber": (0.95, 0.45, 0.02), "Red": (0.75, 0.03, 0.02)}


def stack_light(stage, path, base, looks_path, lit="Green", pole_h=0.10, seg_h=0.075, r=0.035):
    """Stack light Xform at `path`: Pole, Green (bottom), Amber, Red (top), Cap.
    Returns the top Z. Only the `lit` lamp gets an emissive colour (static prop)."""
    path = Sdf.Path(str(path))
    UsdGeom.Xform.Define(stage, path)
    x, y, z = base
    dark = preview_material(stage, Sdf.Path(str(looks_path)).AppendChild("StackLightBody"),
                            (0.03, 0.03, 0.03), roughness=0.5)
    pole = MeshBuilder()
    pole.cylinder((x, y, z), (x, y, z + pole_h), 0.012, 12)
    pole.cylinder((x, y, z), (x, y, z + 0.015), 0.04, 16)
    pole.write(stage, path.AppendChild("Pole"), dark)
    zc = z + pole_h
    for name in ("Green", "Amber", "Red"):
        col = LAMP_COLOURS[name]
        mat = preview_material(stage, Sdf.Path(str(looks_path)).AppendChild(f"Lamp{name}"), col,
                               roughness=0.2, opacity=0.85,
                               emissive=tuple(c * 4 for c in col) if name == lit else None)
        mb = MeshBuilder()
        mb.cylinder((x, y, zc + 0.003), (x, y, zc + seg_h - 0.003), r, 20)
        mb.write(stage, path.AppendChild(name), mat, col)
        ring = MeshBuilder()
        ring.cylinder((x, y, zc + seg_h - 0.003), (x, y, zc + seg_h + 0.003), r + 0.002, 20)
        ring.cylinder((x, y, zc - 0.003), (x, y, zc + 0.003), r + 0.002, 20)
        ring.write(stage, path.AppendChild(f"{name}Ring"), dark)
        zc += seg_h
    cap = MeshBuilder()
    cap.cylinder((x, y, zc), (x, y, zc + 0.015), r + 0.002, 20)
    cap.write(stage, path.AppendChild("Cap"), dark)
    return zc + 0.015
