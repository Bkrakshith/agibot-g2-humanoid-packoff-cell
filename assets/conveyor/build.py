#!/usr/bin/env python
"""Generator for the `conveyor` asset (powered roller conveyor, zero-pressure accumulation).

Run:  ../.venv-tools/bin/python assets/conveyor/build.py
Writes conveyor.usda, geo.usda, physics.usda next to this script (never hand-edit them).

Asset frame (see assets/CONVENTIONS.md): origin = centre of footprint on the floor, Z up,
metres. Conveyor runs along Y; crates flow toward -Y, so upstream is +Y.
-X side ("Near") faces the robot, +X side ("Far") is the far side.
"""
from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- dimensions (m)
LENGTH = 3.6                 # along Y, centred on origin
HALF_L = LENGTH / 2
INNER_W = 0.45               # between side-frame webs and between guide rails
HALF_W = INNER_W / 2
ROLLER_TOP = 0.82
ROLLER_D = 0.050
ROLLER_R = ROLLER_D / 2
ROLLER_AXIS_Z = ROLLER_TOP - ROLLER_R
PITCH = 0.075
N_ROLLERS = int(round(LENGTH / PITCH))           # 48, centred -> gap straddles Y = 0
GUIDE_RISE = 0.040
GUIDE_TOP = ROLLER_TOP + GUIDE_RISE              # 0.86

CH_DEPTH = 0.060             # channel flange width (outward along X)
CH_T = 0.004                 # sheet thickness
CH_TOP = 0.810
CH_BOT = 0.700
CH_OUT = HALF_W + CH_DEPTH   # 0.285

LEG = 0.040                  # square tube
LEG_X = HALF_W + 0.030       # leg centre, 0.255
LEG_YS = (-1.75, -0.60, 0.60, 1.75)
FOOT_PAD_D, FOOT_PAD_H = 0.080, 0.020
FOOT_STEM_D, FOOT_STEM_TOP = 0.020, 0.060

RAIL_T, RAIL_H = 0.012, 0.025
POST = 0.012
PE_Y = 0.35                  # photo-eye: upstream (+Y) of the stop gate
PE_GAP = 0.03                # half-gap in the guide rails for the beam
POST_YS = (-1.70, -1.10, -0.50, 0.10, 0.60, 1.10, 1.70)

GATE_Y = 0.0
GATE_LEN_X = 0.40
GATE_T_Y = 0.015
GATE_H = 0.030
GATE_HOME_Z = 0.800          # StopGate frame origin (bar centre) when lowered
GATE_STROKE = 0.040

BELT_T = 0.010

CYL_SEG = 24


# ---------------------------------------------------------------- mesh helpers
def _set_mesh(mesh: UsdGeom.Mesh, pts, counts, idx, normals):
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))


TRI_COUNT = 0


def box(stage, path, lo, hi):
    """Axis-aligned box from world-ish min/max corners (in parent frame), 12 tris."""
    global TRI_COUNT
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    h = [(b - a) / 2 for a, b in zip(lo, hi)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    faces = [  # (normal, 4 corners as sign tuples, CCW seen from outside)
        ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
        ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)]),
    ]
    pts, idx, nrm = [], [], []
    for n, corners in faces:
        for s in corners:
            idx.append(len(pts))
            pts.append(tuple(s[i] * h[i] for i in range(3)))
            nrm.append(n)
    _set_mesh(mesh, pts, [4] * 6, idx, nrm)
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*c))
    TRI_COUNT += 12
    return mesh


def cylinder(stage, path, center, radius, length, axis="X", seg=CYL_SEG):
    """Closed cylinder mesh; 2*seg + 2*(seg-2) tris."""
    global TRI_COUNT
    a = "XYZ".index(axis)
    u, v = [i for i in range(3) if i != a]

    def vec(along, cu, cv):
        p = [0.0, 0.0, 0.0]
        p[a], p[u], p[v] = along, cu, cv
        return tuple(p)

    ring = [(math.cos(2 * math.pi * k / seg), math.sin(2 * math.pi * k / seg)) for k in range(seg)]
    hl = length / 2
    pts, idx, nrm, counts = [], [], [], []
    for k in range(seg):  # side quads
        k2 = (k + 1) % seg
        quad = [(-hl, k), (-hl, k2), (hl, k2), (hl, k)]
        for along, kk in quad:
            cu, cv = ring[kk]
            idx.append(len(pts))
            pts.append(vec(along, radius * cu, radius * cv))
            nrm.append(vec(0.0, cu, cv))
        counts.append(4)
    for sign in (1, -1):  # caps
        order = range(seg) if sign > 0 else range(seg - 1, -1, -1)
        for kk in order:
            cu, cv = ring[kk]
            idx.append(len(pts))
            pts.append(vec(sign * hl, radius * cu, radius * cv))
            nrm.append(vec(float(sign), 0.0, 0.0))
        counts.append(seg)
    mesh = UsdGeom.Mesh.Define(stage, path)
    _set_mesh(mesh, pts, counts, idx, nrm)
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*center))
    TRI_COUNT += 2 * seg + 2 * (seg - 2)
    return mesh


def new_stage(path: Path):
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    st.SetMetadata("kilogramsPerUnit", 1.0)
    return st


def preview_material(stage, path, color, metallic, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim_or_schema, mat):
    prim = prim_or_schema.GetPrim() if hasattr(prim_or_schema, "GetPrim") else prim_or_schema
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# ---------------------------------------------------------------- geo.usda
def build_geo(path: Path):
    st = new_stage(path)
    root = UsdGeom.Xform.Define(st, "/Conveyor")
    st.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)

    UsdGeom.Scope.Define(st, "/Conveyor/Looks")
    steel = preview_material(st, "/Conveyor/Looks/BrushedStainless", (0.78, 0.78, 0.77), 1.0, 0.35)
    black = preview_material(st, "/Conveyor/Looks/BlackPolymer", (0.03, 0.03, 0.03), 0.0, 0.6)
    refl = preview_material(st, "/Conveyor/Looks/RedReflector", (0.7, 0.02, 0.02), 0.0, 0.2)

    # ---- frame: two C-channels (open outward, hygienic), legs, feet, cross members
    frame = UsdGeom.Xform.Define(st, "/Conveyor/Frame")
    bind(frame, steel)
    for side, s in (("Near", -1), ("Far", 1)):
        base = f"/Conveyor/Frame/Side{side}"
        UsdGeom.Xform.Define(st, base)

        def xr(a, b):
            return (min(s * a, s * b), max(s * a, s * b))
        x0, x1 = xr(HALF_W, HALF_W + CH_T)
        box(st, f"{base}/Web", (x0, -HALF_L, CH_BOT), (x1, HALF_L, CH_TOP))
        x0, x1 = xr(HALF_W, CH_OUT)
        box(st, f"{base}/FlangeTop", (x0, -HALF_L, CH_TOP - CH_T), (x1, HALF_L, CH_TOP))
        box(st, f"{base}/FlangeBottom", (x0, -HALF_L, CH_BOT), (x1, HALF_L, CH_BOT + CH_T))

        for i, y in enumerate(LEG_YS):
            x = s * LEG_X
            box(st, f"{base}/Leg{i}", (x - LEG / 2, y - LEG / 2, FOOT_STEM_TOP), (x + LEG / 2, y + LEG / 2, CH_BOT))
            cylinder(st, f"{base}/FootStem{i}", (x, y, (FOOT_PAD_H + FOOT_STEM_TOP) / 2),
                     FOOT_STEM_D / 2, FOOT_STEM_TOP - FOOT_PAD_H, axis="Z", seg=12)
            pad = cylinder(st, f"{base}/FootPad{i}", (x, y, FOOT_PAD_H / 2),
                           FOOT_PAD_D / 2, FOOT_PAD_H, axis="Z")
            bind(pad, black)

    xs = LEG_X - LEG / 2
    for i, y in enumerate(LEG_YS):
        box(st, f"/Conveyor/Frame/CrossUpper{i}", (-xs, y - 0.015, 0.650), (xs, y + 0.015, 0.680))
        box(st, f"/Conveyor/Frame/CrossLower{i}", (-xs, y - 0.015, 0.150), (xs, y + 0.015, 0.180))
    # gate actuator mount: cross member under Y = 0 plus a pneumatic cylinder body
    box(st, "/Conveyor/Frame/CrossGateMount", (-HALF_W, GATE_Y - 0.020, 0.660), (HALF_W, GATE_Y + 0.020, CH_BOT))
    cylinder(st, "/Conveyor/Frame/GateActuator", (0.0, GATE_Y, 0.730), 0.020, 0.060, axis="Z")

    # ---- rollers (visual only)
    rollers = UsdGeom.Xform.Define(st, "/Conveyor/Rollers")
    bind(rollers, steel)
    y0 = -HALF_L + PITCH / 2
    for i in range(N_ROLLERS):
        cylinder(st, f"/Conveyor/Rollers/Roller{i:02d}", (0.0, y0 + i * PITCH, ROLLER_AXIS_Z),
                 ROLLER_R, INNER_W, axis="X")

    # ---- side guides: rails on posts, gap at the photo-eye beam
    guides = UsdGeom.Xform.Define(st, "/Conveyor/Guides")
    bind(guides, steel)
    rail_z0 = GUIDE_TOP - RAIL_H
    segs = ((-HALF_L, PE_Y - PE_GAP), (PE_Y + PE_GAP, HALF_L))
    for side, s in (("Near", -1), ("Far", 1)):
        base = f"/Conveyor/Guides/{side}"
        UsdGeom.Xform.Define(st, base)
        xa, xb = sorted((s * HALF_W, s * (HALF_W + RAIL_T)))
        for j, (ya, yb) in enumerate(segs):
            box(st, f"{base}/Rail{j}", (xa, ya, rail_z0), (xb, yb, GUIDE_TOP))
        for k, y in enumerate(POST_YS):
            box(st, f"{base}/Post{k}", (xa, y - POST / 2, CH_TOP), (xb, y + POST / 2, rail_z0))

    # ---- belt surface: single static collider, surface-velocity carrier (Stage 3)
    belt = box(st, "/Conveyor/BeltSurface", (-HALF_W, -HALF_L, ROLLER_TOP - BELT_T), (HALF_W, HALF_L, ROLLER_TOP))
    belt.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    belt.GetPrim().SetDocumentation(
        "Static collision box whose top is exactly the roller top (0.82 m). Stage 3 applies "
        "PhysxSurfaceVelocityAPI here (surfaceVelocity along -Y) to convey crates; rollers are visual only.")

    # ---- stop gate: pop-up bar between the two rollers either side of Y = 0
    gate = UsdGeom.Xform.Define(st, "/Conveyor/StopGate")
    UsdGeom.XformCommonAPI(gate).SetTranslate(Gf.Vec3d(0.0, GATE_Y, GATE_HOME_Z))
    gate.GetPrim().SetDocumentation(
        "Pop-up stop bar at the pack position (Y = 0). Driven by /Conveyor/Joints/StopGatePrismatic "
        "(axis Z, 0..0.04 m): drive targetPosition 0 = lowered (top 0.815 m, clear of crates), "
        "0.04 = raised (top 0.855 m, blocks crate).")
    bind(gate, steel)
    box(st, "/Conveyor/StopGate/Bar", (-GATE_LEN_X / 2, -GATE_T_Y / 2, -GATE_H / 2),
        (GATE_LEN_X / 2, GATE_T_Y / 2, GATE_H / 2))
    cylinder(st, "/Conveyor/StopGate/Rod", (0.0, 0.0, -GATE_H / 2 - 0.020), 0.006, 0.040, axis="Z", seg=12)

    # ---- photo-eye (+X, far side) with retro-reflector (-X, near side), upstream at Y = +0.35
    pe = UsdGeom.Xform.Define(st, "/Conveyor/PhotoEye")
    pe.GetPrim().SetDocumentation(
        "Retro-reflective photo-eye 0.35 m upstream (+Y) of the stop gate; crates flow toward -Y. "
        "Geometry only; the beam is /Conveyor/PhotoEye/BeamOrigin.")
    sens_x0 = HALF_W + RAIL_T          # 0.237: emitter face, flush with the outside of the rails
    body_z0, body_z1 = 0.830, 0.865
    beam_z = (body_z0 + body_z1) / 2
    yb0, yb1 = PE_Y - 0.015, PE_Y + 0.015
    for side, s, part in (("Far", 1, "Sensor"), ("Near", -1, "Reflector")):
        grp = UsdGeom.Xform.Define(st, f"/Conveyor/PhotoEye/{part}")
        bracket = box(st, f"/Conveyor/PhotoEye/{part}/Bracket",
                      (min(s * sens_x0, s * CH_OUT), yb0, CH_TOP), (max(s * sens_x0, s * CH_OUT), yb1, body_z0))
        bind(bracket, steel)
        if part == "Sensor":
            body = box(st, f"/Conveyor/PhotoEye/{part}/Body", (sens_x0, yb0, body_z0), (sens_x0 + 0.030, yb1, body_z1))
            bind(body, black)
        else:
            holder = box(st, f"/Conveyor/PhotoEye/{part}/Holder",
                         (-sens_x0 - 0.030, yb0, body_z0), (-sens_x0 - 0.003, yb1, body_z1))
            bind(holder, black)
            disc = box(st, f"/Conveyor/PhotoEye/{part}/Prism",
                       (-sens_x0 - 0.003, yb0 + 0.002, body_z0 + 0.003), (-sens_x0, yb1 - 0.002, body_z1 - 0.003))
            bind(disc, refl)
    beam = UsdGeom.Xform.Define(st, "/Conveyor/PhotoEye/BeamOrigin")
    beam.AddTranslateOp().Set(Gf.Vec3d(sens_x0, PE_Y, beam_z))
    beam.AddRotateZOp().Set(180.0)
    beam.GetPrim().SetDocumentation(
        "Photo-eye beam origin at the emitter face (+X side). Beam direction = local +X, which is "
        "world -X (across the conveyor) toward the reflector at x = -0.237. Beam height 0.8475 m "
        "(27.5 mm above roller top). Beam length = packoff:beamLength.")
    beam.GetPrim().CreateAttribute("packoff:beamLength", Sdf.ValueTypeNames.Float, custom=True).Set(
        float(2 * sens_x0))
    beam.GetPrim().CreateAttribute("packoff:beamAxis", Sdf.ValueTypeNames.Token, custom=True).Set("+X")

    st.GetRootLayer().Save()
    return st


# ---------------------------------------------------------------- physics.usda
def build_physics(path: Path, geo: Usd.Stage):
    st = new_stage(path)
    root = st.OverridePrim("/Conveyor")
    st.SetDefaultPrim(root)

    UsdGeom.Scope.Define(st, "/Conveyor/PhysicsMaterials")
    pmat = UsdShade.Material.Define(st, "/Conveyor/PhysicsMaterials/Stainless")
    pm = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    pm.CreateStaticFrictionAttr(0.3)
    pm.CreateDynamicFrictionAttr(0.3)
    pm.CreateRestitutionAttr(0.0)

    def collide(p, approx="convexHull"):
        o = st.OverridePrim(p)
        UsdPhysics.CollisionAPI.Apply(o)
        UsdPhysics.MeshCollisionAPI.Apply(o).CreateApproximationAttr(approx)

    def bind_phys(p):
        o = st.OverridePrim(p)
        UsdShade.MaterialBindingAPI.Apply(o).Bind(pmat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # static colliders: frame, guides, belt surface (no RigidBodyAPI => static)
    for grp in ("/Conveyor/Frame", "/Conveyor/Guides", "/Conveyor/BeltSurface", "/Conveyor/StopGate"):
        bind_phys(grp)
        for prim in Usd.PrimRange(geo.GetPrimAtPath(grp)):
            if prim.IsA(UsdGeom.Mesh):
                collide(prim.GetPath())

    # stop gate: dynamic body held by a stiff position drive on a prismatic joint
    gate = st.OverridePrim("/Conveyor/StopGate")
    UsdPhysics.RigidBodyAPI.Apply(gate).CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(gate).CreateMassAttr(0.4)
    UsdPhysics.FilteredPairsAPI.Apply(gate).CreateFilteredPairsRel().SetTargets(
        # GateActuator: the rod slides inside the cylinder body; without this filter PhysX
        # depenetrates the gate 15 mm upward at start (found in the Stage 2 cell settle test).
        [Sdf.Path("/Conveyor/BeltSurface"), Sdf.Path("/Conveyor/Frame/GateActuator")])

    UsdGeom.Scope.Define(st, "/Conveyor/Joints")
    j = UsdPhysics.PrismaticJoint.Define(st, "/Conveyor/Joints/StopGatePrismatic")
    j.CreateAxisAttr(UsdPhysics.Tokens.z)
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(GATE_STROKE)
    j.CreateBody0Rel().SetTargets([Sdf.Path("/Conveyor/Frame")])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/Conveyor/StopGate")])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, GATE_Y, GATE_HOME_Z))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    j.GetPrim().SetDocumentation(
        "Stop gate lift. Stage 3: set drive:linear:physics:targetPosition to 0.04 (raise, block crate) "
        "or 0.0 (lower).")
    # Define() of children promotes undefined ancestors to `def`; keep the root a pure `over`
    st.GetRootLayer().GetPrimAtPath("/Conveyor").specifier = Sdf.SpecifierOver
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), UsdPhysics.Tokens.linear)
    d.CreateTypeAttr(UsdPhysics.Tokens.force)
    d.CreateTargetPositionAttr(0.0)
    d.CreateStiffnessAttr(2.0e4)
    d.CreateDampingAttr(1.0e3)
    d.CreateMaxForceAttr(500.0)

    st.GetRootLayer().Save()


# ---------------------------------------------------------------- root layer
def build_root(path: Path):
    st = new_stage(path)
    root = UsdGeom.Xform.Define(st, "/Conveyor")
    st.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.GetPrim().GetReferences().AddReference("./geo.usda")
    root.GetPrim().GetPayloads().AddPayload("./physics.usda")
    st.GetRootLayer().documentation = (
        "Powered roller conveyor, 3.6 m along Y, roller top 0.82 m, crates flow toward -Y. "
        "Generated by build.py - do not hand-edit.")
    st.GetRootLayer().Save()


def main():
    geo = build_geo(HERE / "geo.usda")
    build_physics(HERE / "physics.usda", geo)
    build_root(HERE / "conveyor.usda")
    print(f"conveyor: {N_ROLLERS} rollers, {TRI_COUNT} triangles")


if __name__ == "__main__":
    main()
