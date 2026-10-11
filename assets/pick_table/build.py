"""Generator for the pick_table asset (infeed / accumulation table).

Run: /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/pick_table/build.py
Outputs pick_table.usda (root), geo.usda, physics.usda next to this file. Never hand-edit them.

Local frame: origin = footprint centre on the floor, Z up, metres.
The open (lip-free) side faces -Y: in the cell the table centre is at (-0.05, +0.95), on the
robot's left, so -Y local points back at the robot (no rotation needed when placing it).
"""
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
ROOT_PRIM = "/PickTable"

# ---- parameters (metres) --------------------------------------------------------------
SIZE_X = 0.60            # footprint X
SIZE_Y = 1.00            # footprint Y
TOP_Z = 0.87             # working surface height
TOP_THICK = 0.030        # top sheet + stiffener depth
LIP_H = 0.025            # upturned lip height above the top surface
LIP_T = 0.003            # lip sheet thickness (thin box collider)
LEG_W = 0.040            # square tube legs
LEG_INSET = 0.050        # leg centre inset from footprint edge
FOOT_R = 0.030           # adjustable levelling foot (round pad)
FOOT_H = 0.040
STEM_R = 0.008           # threaded foot stem (visual)
STEM_H = 0.030
BRACE_Z = 0.150          # lower cross-brace centre height
BRACE_W = 0.030
SPAWN_SIZE = (0.50, 0.90, 0.15)
SPAWN_GAP = 0.02         # spawn zone starts this far above the top surface
FRICTION = 0.30          # stainless (CONVENTIONS: 0.3; spec 0.25-0.35)


def _new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def _preview_material(stage, path, color, metallic, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _box(stage, path, center, size, material=None, color=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def _cyl(stage, path, center, radius, height, material=None, color=None):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([(-radius, -radius, -height / 2), (radius, radius, height / 2)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*center))
    if color is not None:
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    return cyl


def lip_specs():
    """name -> (center, size). Lips on +X, -X, +Y; -Y is open (faces the robot)."""
    hx, hy = SIZE_X / 2, SIZE_Y / 2
    zc = TOP_Z + LIP_H / 2
    # lips stand on the top surface, flush with the outer edge
    return {
        "LipPosX": ((hx - LIP_T / 2, 0.0, zc), (LIP_T, SIZE_Y, LIP_H)),
        "LipNegX": ((-hx + LIP_T / 2, 0.0, zc), (LIP_T, SIZE_Y, LIP_H)),
        "LipPosY": ((0.0, hy - LIP_T / 2, zc), (SIZE_X - 2 * LIP_T, LIP_T, LIP_H)),
    }


def build_geo(path):
    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Scope.Define(stage, f"{ROOT_PRIM}/Looks")
    brushed = _preview_material(stage, f"{ROOT_PRIM}/Looks/BrushedStainless", (0.78, 0.78, 0.77), 1.0, 0.35)
    tube = _preview_material(stage, f"{ROOT_PRIM}/Looks/StainlessTube", (0.72, 0.72, 0.72), 1.0, 0.25)
    nylon = _preview_material(stage, f"{ROOT_PRIM}/Looks/FootPad", (0.15, 0.15, 0.15), 0.0, 0.8)
    steel_c, pad_c = (0.78, 0.78, 0.77), (0.15, 0.15, 0.15)

    _box(stage, f"{ROOT_PRIM}/Top", (0.0, 0.0, TOP_Z - TOP_THICK / 2), (SIZE_X, SIZE_Y, TOP_THICK), brushed, steel_c)

    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Lips")
    for name, (c, s) in lip_specs().items():
        _box(stage, f"{ROOT_PRIM}/Lips/{name}", c, s, brushed, steel_c)

    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Frame")
    lx, ly = SIZE_X / 2 - LEG_INSET, SIZE_Y / 2 - LEG_INSET
    leg_z0 = FOOT_H + STEM_H
    leg_z1 = TOP_Z - TOP_THICK
    corners = [(lx, ly), (-lx, ly), (-lx, -ly), (lx, -ly)]
    for i, (x, y) in enumerate(corners, start=1):
        _box(stage, f"{ROOT_PRIM}/Frame/Leg{i}", (x, y, (leg_z0 + leg_z1) / 2),
             (LEG_W, LEG_W, leg_z1 - leg_z0), tube, steel_c)
        _cyl(stage, f"{ROOT_PRIM}/Frame/Foot{i}", (x, y, FOOT_H / 2), FOOT_R, FOOT_H, nylon, pad_c)
        _cyl(stage, f"{ROOT_PRIM}/Frame/FootStem{i}", (x, y, FOOT_H + STEM_H / 2), STEM_R, STEM_H, tube, steel_c)
    # H-shaped lower cross-brace: two side rails along Y + one centre rail along X
    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Frame/CrossBrace")
    rail_len_y = 2 * ly - LEG_W
    _box(stage, f"{ROOT_PRIM}/Frame/CrossBrace/RailPosX", (lx, 0.0, BRACE_Z), (BRACE_W, rail_len_y, BRACE_W), tube, steel_c)
    _box(stage, f"{ROOT_PRIM}/Frame/CrossBrace/RailNegX", (-lx, 0.0, BRACE_Z), (BRACE_W, rail_len_y, BRACE_W), tube, steel_c)
    _box(stage, f"{ROOT_PRIM}/Frame/CrossBrace/RailCentre", (0.0, 0.0, BRACE_Z), (2 * lx - BRACE_W, BRACE_W, BRACE_W), tube, steel_c)

    # Spawn zone: drop region above the top. Invisible guide box, no collision.
    sz = UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/SpawnZone")
    sz_z0 = TOP_Z + SPAWN_GAP
    sz.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sz_z0 + SPAWN_SIZE[2] / 2))
    vol = _box(stage, f"{ROOT_PRIM}/SpawnZone/Volume", (0.0, 0.0, 0.0), SPAWN_SIZE, color=(0.1, 0.8, 0.2))
    vol.CreatePurposeAttr(UsdGeom.Tokens.guide)
    vol.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    stage.GetRootLayer().Save()


COLLIDERS = ["Top", "Lips/LipPosX", "Lips/LipNegX", "Lips/LipPosY",
             "Frame/Leg1", "Frame/Leg2", "Frame/Leg3", "Frame/Leg4"]


def build_physics(path):
    stage = _new_stage(path)
    root = stage.OverridePrim(ROOT_PRIM)
    stage.SetDefaultPrim(root)

    UsdGeom.Scope.Define(stage, f"{ROOT_PRIM}/PhysicsMaterials")
    mat = UsdShade.Material.Define(stage, f"{ROOT_PRIM}/PhysicsMaterials/Stainless")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(FRICTION)
    mapi.CreateDynamicFrictionAttr(FRICTION)
    mapi.CreateRestitutionAttr(0.0)

    # Static colliders: CollisionAPI, no RigidBodyAPI. Cubes are analytic boxes in PhysX.
    for rel in COLLIDERS:
        prim = stage.OverridePrim(f"{ROOT_PRIM}/{rel}")
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    root.SetSpecifier(Sdf.SpecifierOver)  # physics layer carries overs only
    stage.GetRootLayer().Save()


def build_root(path):
    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
    root.GetPrim().GetReferences().AddReference("./geo.usda")
    root.GetPrim().GetPayloads().AddPayload("./physics.usda")
    stage.GetRootLayer().Save()


def main():
    for name in ("pick_table.usda", "geo.usda", "physics.usda"):
        (HERE / name).unlink(missing_ok=True)
    build_geo(HERE / "geo.usda")
    build_physics(HERE / "physics.usda")
    build_root(HERE / "pick_table.usda")
    print(f"wrote {HERE}/pick_table.usda, geo.usda, physics.usda")


if __name__ == "__main__":
    main()
