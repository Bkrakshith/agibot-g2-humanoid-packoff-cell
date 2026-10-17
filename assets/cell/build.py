"""Compose the Stage 2 static cell scene for Genie Sim (pxr only).

Run:
    ../.venv-tools/bin/python assets/cell/build.py [--dest DIR]

DIR defaults to ../geniesim_assets/scenes/g2cell (mounted in the container as
/geniesim_assets/scenes/g2cell). It writes:
    DIR/assets/<name>/...         copies of the Stage 2 asset folders (.usda only)
    DIR/assets/bag_rigid/...      rigid placeholder bag asset (also kept in assets/cell/bag_rigid/)
    DIR/cell.usda                 the composed scene: defaultPrim /World + /PhysicsScene

Cell frame: origin at the G2 base on the floor, +X toward the conveyor, +Y robot's left, Z up.
Layout values live in LAYOUT below; assets/cell/README.md documents them and every deviation.
The scene mirrors scenes/GDK/demo.usda: everything under /World (Xform), a root-level
/PhysicsScene with NewtonSceneAPI + PhysxSceneAPI applied and default values, metres, Z up.
Genie Sim opens this file as the physics stage root, references the robot at /genie, and the
OVRTX render node opens it again as the render stage.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DEFAULT_DEST = ASSETS.parents[1] / "geniesim_assets" / "scenes" / "g2cell"

# ---------------------------------------------------------------- layout (cell frame, metres)
# Layout v2 (2026-10-11): U-shaped work cell the G2 DRIVES INTO from -X. Pick table on the robot's
# left (+Y), conveyor in front (+X), sort rack on the right (-Y) turned so its open front faces +Y.
# The U opens toward -X. End of line: case erector feeding the conveyor's +Y end, a drop gap with
# a floor bin at the -Y end, a right-angle transfer, then the case sealer and a cobot palletiser
# running along +X (a straight line does not fit the 9.8 m room: 2.2 + 3.6 + 0.3 + 1.6 + 2.2 m).
# Every number is documented in assets/cell/README.md ("Layout v2").
BELT_TOP = 0.820            # conveyor BeltSurface top (= erector/sealer/palletiser bed tops)
CHASSIS_HALF_W = 0.318      # G2 base_link mesh |Y| max (robot.urdf base_link.dae)
U_INNER_HALF = 0.43         # table inner edge = +0.43, rack inner edge (tote lips) = -0.43
DROP_GAP = 0.30             # conveyor end (Y -1.80) -> transfer infeed (Y -2.10)
LAYOUT = {
    "conveyor": (0.60, 0.0),
    # (x, y, yaw_deg). Turned -90 deg so its 1.00 m side runs along X (footprint X -0.715..0.285, clear of
    # the conveyor frame at X 0.305; Y 0.43..1.03, inner edge 0.112 m from the chassis). The spawn zone is
    # then X -0.665..0.235 x Y 0.48..0.98, inside the waist-turned left arm's reach. The lip-less side
    # faces the U opening (-X); the 25 mm lip on the robot side is reached over.
    "pick_table": (-0.215, U_INNER_HALF + 0.30, -90.0),
    # (x, y, yaw_deg). yaw +90: local +X (open front) -> world +Y. ONE lane column x 2 levels (the second
    # column of the 2-lane rack sat behind the robot at X -0.74, unreachable at any waist yaw).
    # Front-face Y -0.495; the tilted totes' top-front edges protrude to Y -0.43 (inner edge).
    # X range -0.275..0.275 (conveyor frame starts at X 0.305); tote column at X 0.0.
    "sort_rack": (0.0, -0.82, 90.0),
    # erector outfeed edge (local Y -1.10) butts the conveyor's +Y end (Y 1.80)
    "case_erector": (0.60, 2.90, 0.0),
    # floor bin centred under the drop gap, from the conveyor end (Y -1.803) to under the transfer
    "floor_bin": (0.60, -2.205, 0.0),
    # right-angle transfer placeholder (this script): 0.70 x 0.70 bed, top 0.82, centre
    "transfer": (0.60, -2.10 - 0.35),
    # sealer and palletiser turned +90 deg (flow local -Y -> world +X): sealer EntryPoint at the
    # transfer's +X edge (0.95, -2.45); palletiser EntryPoint at the sealer ExitPoint (2.55, -2.45)
    "case_sealer": (1.75, -2.45, 90.0),
    "palletiser": (3.65, -2.45, 90.0),
    # props moved out of the way of the line (props asset defaults in assets/props/README.md)
    "prop_moves": {"LightCurtainPost": (2.0, -3.6)},
    # cartons (name, asset, x, y, yaw_deg); bottoms on the bed tops (BELT_TOP). Open-carton pitch
    # 0.75 m on the belt (flaps flare to 0.69 m along Y).
    "cartons": [
        ("Carton_Erector", "carton_open", 0.60, 2.20, 0.0),     # erector outfeed SpawnPoint
        ("Carton_Up_02", "carton_open", 0.60, 1.45, 0.0),
        ("Carton_Up_01", "carton_open", 0.60, 0.75, 0.0),
        ("Carton_Pack", "carton_open", 0.60, 0.00, 0.0),        # pack position (stop gate)
        ("Carton_Down_01", "carton_open", 0.60, -0.90, 0.0),    # packed, heading to the sealer
        ("Carton_Sealer", "carton_sealed", 1.75, -2.45, 90.0),  # inside the sealer tunnel
        ("Carton_PalletInfeed", "carton_sealed", 3.03, -2.45, 90.0),  # palletiser PickPoint
    ],
    # rigid placeholder bags on the pick table top (0.87 m): (name, x, y, yaw_deg)
    "bags": [
        # inside the spawn zone (X -0.665..0.235, Y 0.48..0.98); Bag_03 turned 90 deg at the far side
        ("Bag_01", 0.05, 0.62, 0.0),
        ("Bag_02", -0.25, 0.62, 4.0),
        ("Bag_03", -0.10, 0.90, 90.0),
    ],
    # bags elsewhere in the story: (name, x, y, z_bottom, yaw_deg)
    "line_bags": [
        ("Bag_InCarton", 0.60, -0.90, BELT_TOP + 0.008, 0.0),   # on Carton_Down_01's base
        ("Bag_InBin", 0.62, -2.28, 0.108, 25.0),                # stray bag on the floor-bin floor
    ],
}
TABLE_TOP = 0.870
# The ToteSlot frame puts the tote's front-bottom edge on the stop bar; with the tote's 3 mm
# contact offset PhysX pushed every tote 13.7 mm up-slope in the settle test, so spawn them
# there (along slot -X) to start at rest.
TOTE_SLOT_BACKOFF = 0.014
TOTE_COLOURS = {"L1_1": "red", "L1_2": "blue", "L2_1": "green", "L2_2": "yellow"}   # only L1_1/L2_1 exist now
# right-angle transfer placeholder
RAT_SIZE = 0.70
RAT_LEG_INSET = 0.015       # legs at +-0.335 m: outside the floor bin (+-0.302 m in X)

COLOURS = {"L1_1": "red", "L1_2": "blue", "L2_1": "green", "L2_2": "yellow"}

COLOURS = {
    "blue": (0.02, 0.15, 0.55),
    "green": (0.05, 0.35, 0.10),
    "grey": (0.35, 0.36, 0.37),
    "red": (0.55, 0.04, 0.03),
    "yellow": (0.65, 0.50, 0.02),
}

# ---------------------------------------------------------------- rigid placeholder bag
BAG_SIZE = (0.21, 0.30, 0.07)   # X, Y, Z
BAG_RADIUS = 0.025              # edge rounding
BAG_MASS = 0.12
BAG_FRICTION = 0.5
BAG_COLOUR = (0.32, 0.58, 0.22)  # salad green
CONTACT_OFFSET = 0.003

# asset folder name -> (source folder, root file, defaultPrim)
ASSET_SOURCES = {
    "room": ("room", "room.usda"),
    "props": ("props", "props.usda"),
    "conveyor": ("conveyor", "conveyor.usda"),
    "pick_table": ("pick_table", "pick_table.usda"),
    "sort_rack": ("sort_rack", "sort_rack.usda"),
    "deep_325": ("crates/deep_325", "deep_325.usda"),
    "stacking_224": ("crates/stacking_224", "stacking_224.usda"),
    "case_erector": ("case_erector", "case_erector.usda"),
    "case_sealer": ("case_sealer", "case_sealer.usda"),
    "palletiser": ("palletiser", "palletiser.usda"),
    "floor_bin": ("floor_bin", "floor_bin.usda"),
}
# cartons/ holds several sub-assets (+ usdkit.py); copied as a tree, referenced by erector/palletiser
CARTON_ASSETS = ("carton_open", "carton_sealed", "carton_flat", "carton_flat_stack")


def new_stage(path: Path, default_prim: str | None = None) -> Usd.Stage:
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().comment = "Generated by assets/cell/build.py - do not edit by hand."
    return stage


# ---------------------------------------------------------------- bag asset
def rounded_box_mesh(size, r, n_band=4, n_flat=2):
    """Quad mesh of a box with rounded edges/corners (radius r), centred on the origin."""
    half = [s / 2 for s in size]
    inner = [h - r for h in half]

    def axis_coords(a):
        band = [-(inner[a] + r * math.cos(t)) for t in
                [i * (math.pi / 2) / n_band for i in range(n_band + 1)]]  # -h .. -inner
        flat = [-inner[a] + 2 * inner[a] * k / n_flat for k in range(1, n_flat)]
        return band + flat + [-c for c in reversed(band)]

    coords = [axis_coords(a) for a in range(3)]
    points, faces, index = [], [], {}

    def project(p):
        q = [max(-inner[i], min(inner[i], p[i])) for i in range(3)]
        d = [p[i] - q[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in d)) or 1.0
        return tuple(q[i] + r * d[i] / n for i in range(3))

    def vid(p):
        key = tuple(round(c, 9) for c in p)
        if key not in index:
            index[key] = len(points)
            points.append(project(p))
        return index[key]

    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            cu, cv = coords[u], coords[v]
            for i in range(len(cu) - 1):
                for j in range(len(cv) - 1):
                    quad = []
                    for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis] = sign * half[axis]
                        p[u], p[v] = cu[a], cv[b]
                        quad.append(vid(p))
                    # outward winding (right-handed): flip for negative faces / axis parity
                    flip = (sign < 0) ^ (axis == 1)
                    faces.append(list(reversed(quad)) if flip else quad)
    return points, faces


def write_bag_asset(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    prim_name = "BagRigid"
    sx, sy, sz = BAG_SIZE

    # geo.usda: visual mesh (origin = footprint centre on the table, Z = 0 at the bottom)
    geo = new_stage(folder / "geo.usda")
    root = UsdGeom.Xform.Define(geo, f"/{prim_name}")
    geo.SetDefaultPrim(root.GetPrim())
    pts, faces = rounded_box_mesh(BAG_SIZE, BAG_RADIUS)
    mesh = UsdGeom.Mesh.Define(geo, f"/{prim_name}/Geo/Body")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(p[0], p[1], p[2] + sz / 2) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*BAG_COLOUR)]))
    mat = UsdShade.Material.Define(geo, f"/{prim_name}/Looks/BagFilm")
    sh = UsdShade.Shader.Define(geo, f"/{prim_name}/Looks/BagFilm/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BAG_COLOUR))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    col = UsdGeom.Cube.Define(geo, f"/{prim_name}/Colliders/Box")
    col.CreateSizeAttr(1.0)
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xf = UsdGeom.Xformable(col)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, sz / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    geo.GetRootLayer().Save()

    # physics.usda: overs only + material
    ph = new_stage(folder / "physics.usda")
    over = ph.OverridePrim(f"/{prim_name}")
    ph.SetDefaultPrim(over)
    UsdPhysics.RigidBodyAPI.Apply(over)
    UsdPhysics.MassAPI.Apply(over).CreateMassAttr(BAG_MASS)
    pm = UsdShade.Material.Define(ph, f"/{prim_name}/PhysicsMaterials/BagFilm")
    pmapi = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    pmapi.CreateStaticFrictionAttr(BAG_FRICTION)
    pmapi.CreateDynamicFrictionAttr(BAG_FRICTION)
    pmapi.CreateRestitutionAttr(0.0)
    box = ph.OverridePrim(f"/{prim_name}/Colliders/Box")
    UsdPhysics.CollisionAPI.Apply(box)
    box.AddAppliedSchema("PhysxCollisionAPI")
    box.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=True).Set(CONTACT_OFFSET)
    box.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=True).Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(box).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    ph.GetRootLayer().Save()

    # root layer
    rs = new_stage(folder / "bag_rigid.usda")
    rp = rs.DefinePrim(f"/{prim_name}", "Xform")
    rs.SetDefaultPrim(rp)
    Usd.ModelAPI(rp).SetKind(Kind.Tokens.component)
    rp.GetReferences().AddReference("./geo.usda")
    rp.GetPayloads().AddPayload("./physics.usda")
    rs.GetRootLayer().Save()


# ---------------------------------------------------------------- copying
def copy_assets(dest_assets: Path) -> None:
    for name, (src, _root) in ASSET_SOURCES.items():
        out = dest_assets / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for f in (ASSETS / src).iterdir():
            if f.suffix in (".usda", ".md"):
                shutil.copy2(f, out / f.name)
    out = dest_assets / "cartons"
    if out.exists():
        shutil.rmtree(out)
    for sub in CARTON_ASSETS:
        (out / sub).mkdir(parents=True)
        for f in (ASSETS / "cartons" / sub).iterdir():
            if f.suffix in (".usda", ".md"):
                shutil.copy2(f, out / sub / f.name)
    for local in ("bag_rigid", "transfer_rat"):
        out = dest_assets / local
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(HERE / local, out)


# ---------------------------------------------------------------- right-angle transfer placeholder
def write_rat_asset(folder: Path) -> None:
    """Static right-angle transfer between the conveyor's drop gap and the sealer (placeholder:
    roller bed + frame on 4 legs, open underneath so the floor bin slides half under it).
    Origin = footprint centre on the floor; flow in from +Y, out to +X. Not animated."""
    folder.mkdir(parents=True, exist_ok=True)
    name = "TransferRAT"
    h = RAT_SIZE / 2
    geo = new_stage(folder / "geo.usda")
    root = UsdGeom.Xform.Define(geo, f"/{name}")
    geo.SetDefaultPrim(root.GetPrim())
    steel = UsdShade.Material.Define(geo, f"/{name}/Looks/Stainless")
    sh = UsdShade.Shader.Define(geo, f"/{name}/Looks/Stainless/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.62, 0.63, 0.64))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    steel.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    def box(path, centre, size, visible=True):
        c = UsdGeom.Cube.Define(geo, path)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        if visible:
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(steel)
        else:
            c.CreatePurposeAttr(UsdGeom.Tokens.guide)
        return c

    t = 0.04
    for sx in (-1, 1):   # side channels along Y and X (frame 0.72..0.80)
        box(f"/{name}/Geo/Frame/RailY{'P' if sx > 0 else 'N'}", (sx * (h - t / 2), 0, 0.76), (t, RAT_SIZE, 0.08))
        box(f"/{name}/Geo/Frame/RailX{'P' if sx > 0 else 'N'}", (0, sx * (h - t / 2), 0.76), (RAT_SIZE, t, 0.08))
    k = 0
    for sx in (-1, 1):
        for sy in (-1, 1):
            k += 1
            box(f"/{name}/Geo/Frame/Leg{k}", (sx * (h - RAT_LEG_INSET), sy * (h - RAT_LEG_INSET), 0.36),
                (0.03, 0.03, 0.72))
    # rollers: diameter 50 mm, axis along Y (carry toward +X), 75 mm pitch, top 0.82
    n = int((RAT_SIZE - 2 * t) // 0.075)
    for i in range(n):
        cyl = UsdGeom.Cylinder.Define(geo, f"/{name}/Geo/Rollers/Roller{i:02d}")
        cyl.CreateRadiusAttr(0.025)
        cyl.CreateHeightAttr(RAT_SIZE - 2 * t)
        cyl.CreateAxisAttr(UsdGeom.Tokens.y)
        UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(-h + t + 0.0375 + i * 0.075, 0, BELT_TOP - 0.025))
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(steel)
    box(f"/{name}/Colliders/Bed", (0, 0, BELT_TOP - 0.005), (RAT_SIZE - 2 * t, RAT_SIZE - 2 * t, 0.01), visible=False)
    box(f"/{name}/Colliders/Frame", (0, 0, 0.76), (RAT_SIZE, RAT_SIZE, 0.08), visible=False)
    geo.GetRootLayer().Save()

    ph = new_stage(folder / "physics.usda")
    over = ph.OverridePrim(f"/{name}")
    ph.SetDefaultPrim(over)
    pm = UsdShade.Material.Define(ph, f"/{name}/PhysicsMaterials/Stainless")
    pmapi = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    pmapi.CreateStaticFrictionAttr(0.3)
    pmapi.CreateDynamicFrictionAttr(0.3)
    for c in ("Bed", "Frame"):
        p = ph.OverridePrim(f"/{name}/Colliders/{c}")
        UsdPhysics.CollisionAPI.Apply(p)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    ph.GetRootLayer().Save()

    rs = new_stage(folder / "transfer_rat.usda")
    rp = rs.DefinePrim(f"/{name}", "Xform")
    rs.SetDefaultPrim(rp)
    Usd.ModelAPI(rp).SetKind(Kind.Tokens.component)
    rp.GetReferences().AddReference("./geo.usda")
    rp.GetPayloads().AddPayload("./physics.usda")
    rs.GetRootLayer().Save()


# ---------------------------------------------------------------- composition
def add_ref(stage, path, asset_rel, translate=(0, 0, 0), orient=None, yaw_deg=None):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(asset_rel)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if orient is not None:
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    elif yaw_deg is not None:
        xf.AddRotateZOp().Set(float(yaw_deg))
    return prim


def set_colour(stage, crate_path, geo_child, colour):
    geo = stage.GetPrimAtPath(f"{crate_path}/Geo")
    UsdGeom.PrimvarsAPI(geo).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
    ).Set(Vt.Vec3fArray([Gf.Vec3f(*COLOURS[colour])]))


def tote_slot_poses():
    """World pose of each sort-rack ToteSlot, from the sort_rack asset itself."""
    rack = Usd.Stage.Open(str(ASSETS / "sort_rack" / "sort_rack.usda"))
    rx, ry, ryaw = LAYOUT["sort_rack"]
    t_rack = (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), ryaw))
              * Gf.Matrix4d().SetTranslate(Gf.Vec3d(rx, ry, 0)))
    out = {}
    for p in rack.Traverse():
        if p.GetName() == "ToteSlot":
            lane = p.GetParent().GetName()  # L{level}_{lane}
            slot = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            size = p.GetAttribute("packoff:toteSize").Get()
            # tote asset: origin at base-footprint centre, long side +Y; slot: origin at the
            # volume centre, +X = 600 mm side down-slope. So: rotate Z 90 deg, drop half height.
            local = (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
                     * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-TOTE_SLOT_BACKOFF, 0, -size[2] / 2)))
            out[lane] = local * slot * t_rack
    return dict(sorted(out.items()))


def build_cell(dest: Path) -> Path:
    stage = new_stage(dest / "cell.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    add_ref(stage, "/World/Room", "./assets/room/room.usda")
    add_ref(stage, "/World/Props", "./assets/props/props.usda")
    cx, cy = LAYOUT["conveyor"]
    add_ref(stage, "/World/Conveyor", "./assets/conveyor/conveyor.usda", (cx, cy, 0))
    tx, ty, tyaw = LAYOUT["pick_table"]
    add_ref(stage, "/World/PickTable", "./assets/pick_table/pick_table.usda", (tx, ty, 0), yaw_deg=tyaw)
    # Genie Sim's kit/stage.py _configure_drives() classifies EVERY revolute/prismatic joint on
    # the stage by robot-joint name regexes and aborts on unknown ones ("StopGatePrismatic").
    # Upstream is not edited, so the gate joint is deactivated here and the gate made kinematic
    # at its lowered pose. Stage 3 drives the gate kinematically (or via a renamed/handled joint).
    stage.OverridePrim("/World/Conveyor/Joints/StopGatePrismatic").SetActive(False)
    stage.OverridePrim("/World/Conveyor/StopGate").CreateAttribute(
        "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    rx, ry, ryaw = LAYOUT["sort_rack"]
    add_ref(stage, "/World/SortRack", "./assets/sort_rack/sort_rack.usda", (rx, ry, 0), yaw_deg=ryaw)
    for key, prim, rel in (("case_erector", "CaseErector", "case_erector/case_erector.usda"),
                           ("case_sealer", "CaseSealer", "case_sealer/case_sealer.usda"),
                           ("palletiser", "Palletiser", "palletiser/palletiser.usda"),
                           ("floor_bin", "FloorBin", "floor_bin/floor_bin.usda")):
        x, y, yaw = LAYOUT[key]
        add_ref(stage, f"/World/{prim}", f"./assets/{rel}", (x, y, 0), yaw_deg=yaw)
    x, y = LAYOUT["transfer"]
    add_ref(stage, "/World/TransferRAT", "./assets/transfer_rat/transfer_rat.usda", (x, y, 0))
    for prop, (x, y) in LAYOUT["prop_moves"].items():
        stage.OverridePrim(f"/World/Props/{prop}").CreateAttribute(
            "xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(x, y, 0))

    UsdGeom.Xform.Define(stage, "/World/Totes")
    for lane, m in tote_slot_poses().items():
        path = f"/World/Totes/Tote_{lane}"
        q = m.ExtractRotationQuat()
        add_ref(stage, path, "./assets/deep_325/deep_325.usda", tuple(m.ExtractTranslation()),
                orient=Gf.Quatd(q.GetReal(), q.GetImaginary()))
        set_colour(stage, path, "Geo", TOTE_COLOURS[lane])

    # Layout v2: corrugated cartons replace the stacking crates on the line
    UsdGeom.Xform.Define(stage, "/World/Cartons")
    for name, asset, x, y, yaw in LAYOUT["cartons"]:
        add_ref(stage, f"/World/Cartons/{name}", f"./assets/cartons/{asset}/{asset}.usda",
                (x, y, BELT_TOP), yaw_deg=yaw)

    UsdGeom.Xform.Define(stage, "/World/Bags")
    for name, x, y, yaw in LAYOUT["bags"]:
        add_ref(stage, f"/World/Bags/{name}", "./assets/bag_rigid/bag_rigid.usda",
                (x, y, TABLE_TOP), yaw_deg=yaw)
    UsdGeom.Xform.Define(stage, "/World/LineBags")
    for name, x, y, z, yaw in LAYOUT["line_bags"]:
        add_ref(stage, f"/World/LineBags/{name}", "./assets/bag_rigid/bag_rigid.usda", (x, y, z), yaw_deg=yaw)

    # PhysicsScene mirrors scenes/GDK/demo.usda: root-level, schema tokens only, defaults.
    ps = stage.DefinePrim("/PhysicsScene", "PhysicsScene")
    for schema in ("NewtonSceneAPI", "PhysxSceneAPI"):
        ps.AddAppliedSchema(schema)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)
    stage.SetTimeCodesPerSecond(60)
    stage.GetRootLayer().Save()
    return dest / "cell.usda"


def export_colliders(cell_usda: Path) -> Path:
    """Oriented boxes of every collider in the composed cell (cell frame) -> cell_colliders.json next
    to cell.usda. Used by g2cell_demo.env_check to keep planned arm paths off the fixtures (the G2's
    arm links DO collide with the environment in the mobile scene; a contact kicks the free base).
    Cube colliders are exact OBBs; mesh colliders (conveyor frame, all boxy) use their world AABB.
    The placeholder bags are left out (they are the grasp targets)."""
    import json
    stage = Usd.Stage.Open(str(cell_usda))
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    out = []
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not p.HasAPI(UsdPhysics.CollisionAPI) or path.startswith(("/World/Bags", "/World/LineBags")):
            continue
        if p.IsA(UsdGeom.Cube):
            m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            half = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
            c = m.Transform(Gf.Vec3d(0, 0, 0))
            axes, ext = [], []
            for i in range(3):
                v = m.TransformDir(Gf.Vec3d(*[1.0 if k == i else 0.0 for k in range(3)])) * half
                ext.append(v.GetLength())
                axes.append(list(v.GetNormalized()))
        else:
            r = bc.ComputeWorldBound(p).ComputeAlignedRange()
            if r.IsEmpty():
                continue
            c = (r.GetMin() + r.GetMax()) / 2
            ext = list((r.GetMax() - r.GetMin()) / 2)
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out.append({"name": path, "c": [round(v, 5) for v in c], "axes": [[round(v, 6) for v in a] for a in axes],
                    "h": [round(v, 5) for v in ext]})
    dst = cell_usda.parent / "cell_colliders.json"
    dst.write_text(json.dumps(out))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    write_bag_asset(HERE / "bag_rigid")
    write_rat_asset(HERE / "transfer_rat")
    args.dest.mkdir(parents=True, exist_ok=True)
    copy_assets(args.dest / "assets")
    out = build_cell(args.dest)
    print(f"wrote {out}")
    print(f"wrote {export_colliders(out)}")


if __name__ == "__main__":
    main()
