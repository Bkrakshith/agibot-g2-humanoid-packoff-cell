"""Generator for the sort_rack asset (angled gravity-flow / tote-presentation rack).

Run: ../.venv-tools/bin/python assets/sort_rack/build.py
Outputs sort_rack.usda (root), geo.usda, physics.usda next to this file. Never hand-edit them.

Local frame: origin = footprint centre on the floor, Z up, metres. The open, tote-facing front
is +X; in the cell the rack centre is at (-0.78, 0) so +X points at the robot (no rotation).
Each shelf slopes DOWN toward +X by TILT_DEG; totes slide forward against a front stop bar.
Levels are numbered 1..N_LEVELS bottom-up, lanes 1..N_LANES from -Y to +Y.
"""
import math
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
ROOT_PRIM = "/SortRack"

# ---- parameters (metres / degrees) ----------------------------------------------------
DEPTH_X = 0.65
WIDTH_Y = 0.55                 # Layout v2: one lane column (was 1.35 m, 2 lanes)
N_LEVELS = 2
N_LANES = 1                    # Layout v2 (was 2; the -X column sat behind the robot, out of reach)
LIP_HEIGHTS = [0.75, 1.30]     # top of the front stop bar per level (bottom-up). Layout v2 tried L2 1.20; back to 1.30 (see cell README)
TILT_DEG = 20.0                # shelf slope from horizontal, down toward +X
PROFILE = 0.040                # 40x40 aluminium extrusion
LABEL_T = 0.003                # label card thickness (front of stop bar)
STOP_T = 0.030                 # stop bar depth (X)
STOP_ABOVE_DECK = 0.040        # stop bar top above the roller surface at the front edge
STOP_BELOW_DECK = 0.030        # stop bar extends this far below the roller surface
DECK_T = 0.010                 # thin collision box thickness
ROLLER_R = 0.015
ROLLER_PITCH = 0.050
CHANNEL_W, CHANNEL_H = 0.012, 0.035   # roller side channels (visual)
TOP_CLEAR = 0.060              # upright height above the highest rear deck point
BASE_RAIL_Z = 0.060
TOTE = (0.600, 0.400, 0.325)   # L (along slope) x W (across lane) x H
DECK_FRICTION = 0.10
FRAME_FRICTION = 0.30
LABEL_COLOURS = [(0.85, 0.15, 0.15), (0.15, 0.35, 0.85), (0.15, 0.70, 0.25), (0.95, 0.80, 0.10),
                 (0.60, 0.20, 0.70), (0.95, 0.50, 0.10), (0.10, 0.70, 0.70), (0.90, 0.90, 0.90),
                 (0.40, 0.25, 0.10)]

assert len(LIP_HEIGHTS) == N_LEVELS


# ---- derived layout ------------------------------------------------------------------
def layout():
    t = math.radians(TILT_DEG)
    hx, hy = DEPTH_X / 2, WIDTH_Y / 2
    stop_x1 = hx - LABEL_T
    stop_x0 = stop_x1 - STOP_T
    deck_xf = stop_x0                  # roller surface front edge (against stop bar)
    deck_xr = -hx + PROFILE            # rear edge, in front of the rear uprights
    span = deck_xf - deck_xr
    length = span / math.cos(t)
    rise = span * math.tan(t)
    lane_w = (WIDTH_Y - (N_LANES + 1) * PROFILE) / N_LANES
    lane_y = [-hy + PROFILE + lane_w / 2 + i * (lane_w + PROFILE) for i in range(N_LANES)]
    post_y = [-hy + PROFILE / 2 + i * (lane_w + PROFILE) for i in range(N_LANES + 1)]
    levels = []
    for lip in LIP_HEIGHTS:
        zf = lip - STOP_ABOVE_DECK
        levels.append(dict(lip=lip, zf=zf, zr=zf + rise))
    height = max(l["zr"] for l in levels) + TOP_CLEAR
    return dict(t=t, hx=hx, hy=hy, stop_x0=stop_x0, stop_x1=stop_x1, deck_xf=deck_xf, deck_xr=deck_xr,
                span=span, length=length, lane_w=lane_w, lane_y=lane_y, post_y=post_y,
                levels=levels, height=height)


def slope_dirs(t):
    """World directions of the tilted frame (rotateY(+TILT)): +X down-slope, +Z deck normal."""
    return Gf.Vec3d(math.cos(t), 0, -math.sin(t)), Gf.Vec3d(math.sin(t), 0, math.cos(t))


def tote_slot_centre(L, level, lane):
    """Tote volume centre: front-bottom edge against the stop bar, centred in the lane."""
    ex, ez = slope_dirs(L["t"])
    lv = L["levels"][level]
    front = Gf.Vec3d(L["deck_xf"], L["lane_y"][lane], lv["zf"])
    return front - ex * (TOTE[0] / 2) + ez * (TOTE[2] / 2)


# ---- helpers ---------------------------------------------------------------------------
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


def _bind(prim, material, color):
    if color is not None:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box(stage, path, center, size, material=None, color=None, rot_y=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rot_y is not None:
        cube.AddRotateYOp().Set(rot_y)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material, color)
    return cube


def _roller(stage, path, x_local, radius, length, material, color):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(UsdGeom.Tokens.y)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateExtentAttr([(-radius, -length / 2, -radius), (radius, length / 2, radius)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(x_local, 0, -radius))
    _bind(cyl.GetPrim(), material, color)
    return cyl


# ---- layers ----------------------------------------------------------------------------
def build_geo(path):
    L = layout()
    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())
    ex, ez = slope_dirs(L["t"])

    UsdGeom.Scope.Define(stage, f"{ROOT_PRIM}/Looks")
    alu_c, roller_c, zinc_c = (0.80, 0.81, 0.83), (0.62, 0.64, 0.66), (0.55, 0.56, 0.58)
    alu = _preview_material(stage, f"{ROOT_PRIM}/Looks/AnodisedAluminium", alu_c, 0.9, 0.45)
    roller_m = _preview_material(stage, f"{ROOT_PRIM}/Looks/ZincRoller", roller_c, 1.0, 0.3)
    steel = _preview_material(stage, f"{ROOT_PRIM}/Looks/ZincSteel", zinc_c, 1.0, 0.4)

    # Frame: uprights at front and rear of every lane boundary, plus horizontal rails.
    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Frame")
    H = L["height"]
    ux = L["hx"] - PROFILE / 2
    k = 0
    for side, x in (("Front", ux), ("Rear", -ux)):
        for j, y in enumerate(L["post_y"]):
            k += 1
            _box(stage, f"{ROOT_PRIM}/Frame/Upright{side}{j + 1}", (x, y, H / 2), (PROFILE, PROFILE, H), alu, alu_c)
    inner_w = WIDTH_Y - 2 * PROFILE
    for side, x in (("Front", ux), ("Rear", -ux)):
        _box(stage, f"{ROOT_PRIM}/Frame/BaseRail{side}", (x, 0, BASE_RAIL_Z), (PROFILE, inner_w, PROFILE), alu, alu_c)
        _box(stage, f"{ROOT_PRIM}/Frame/TopRail{side}", (x, 0, H - PROFILE / 2), (PROFILE, inner_w, PROFILE), alu, alu_c)
    for j, y in enumerate(L["post_y"]):
        _box(stage, f"{ROOT_PRIM}/Frame/BaseTie{j + 1}", (0, y, BASE_RAIL_Z), (DEPTH_X - 2 * PROFILE, PROFILE, PROFILE), alu, alu_c)

    # Shelves
    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Shelves")
    n_rollers = int((L["length"] - ROLLER_PITCH / 2) // ROLLER_PITCH)
    for li, lv in enumerate(L["levels"], start=1):
        sp = f"{ROOT_PRIM}/Shelves/Level{li}"
        UsdGeom.Xform.Define(stage, sp)
        front = Gf.Vec3d(L["deck_xf"], 0, lv["zf"])          # roller surface, front edge, y = 0
        mid = front - ex * (L["length"] / 2)

        # Collision deck (one thin tilted box per shelf, top face = roller tangent plane)
        dc = mid - ez * (DECK_T / 2)
        deck = _box(stage, f"{sp}/DeckCollider", (dc[0], dc[1], dc[2]),
                    (L["length"], WIDTH_Y - 2 * PROFILE, DECK_T), rot_y=TILT_DEG)
        deck.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        # Front stop bar (vertical, static) — its top is the spec "front lip" height
        z0 = lv["zf"] - STOP_BELOW_DECK
        _box(stage, f"{sp}/StopBar", ((L["stop_x0"] + L["stop_x1"]) / 2, 0, (z0 + lv["lip"]) / 2),
             (STOP_T, WIDTH_Y - 2 * PROFILE, lv["lip"] - z0), steel, zinc_c)

        # Tilted support beams under the deck along each upright line
        bc = mid - ez * (CHANNEL_H + PROFILE / 2)
        for j, y in enumerate(L["post_y"]):
            _box(stage, f"{sp}/Beam{j + 1}", (bc[0], y, bc[2]), (L["length"], PROFILE, PROFILE),
                 alu, alu_c, rot_y=TILT_DEG)

        # Roller tracks per lane (visual only)
        roller_len = L["lane_w"] - 2 * CHANNEL_W - 0.004
        for lane, y in enumerate(L["lane_y"], start=1):
            rp = f"{sp}/Lane{lane}Rollers"
            xf = UsdGeom.Xform.Define(stage, rp)
            xf.AddTranslateOp().Set(Gf.Vec3d(front[0], y, front[2]))
            xf.AddRotateYOp().Set(TILT_DEG)
            for i in range(n_rollers):
                _roller(stage, f"{rp}/Roller{i:02d}", -(i + 0.5) * ROLLER_PITCH,
                        ROLLER_R, roller_len, roller_m, roller_c)
            for s, sy in (("L", -1), ("R", 1)):
                ch = UsdGeom.Cube.Define(stage, f"{rp}/Channel{s}")
                ch.CreateSizeAttr(1.0)
                ch.AddTranslateOp().Set(Gf.Vec3d(-L["length"] / 2, sy * (L["lane_w"] / 2 - CHANNEL_W / 2), -CHANNEL_H / 2 + 0.005))
                ch.AddScaleOp().Set(Gf.Vec3f(L["length"], CHANNEL_W, CHANNEL_H))
                _bind(ch.GetPrim(), steel, zinc_c)

    # Lanes: label card per lane on the stop bar front face + tote slot pose
    UsdGeom.Xform.Define(stage, f"{ROOT_PRIM}/Lanes")
    ci = 0
    for li, lv in enumerate(L["levels"], start=1):
        z0 = lv["zf"] - STOP_BELOW_DECK
        for lane, y in enumerate(L["lane_y"], start=1):
            lp = f"{ROOT_PRIM}/Lanes/L{li}_{lane}"
            UsdGeom.Xform.Define(stage, lp)
            colour = LABEL_COLOURS[ci % len(LABEL_COLOURS)]
            ci += 1
            _box(stage, f"{lp}/Label", (L["stop_x1"] + LABEL_T / 2, y, (z0 + lv["lip"]) / 2),
                 (LABEL_T, 0.12, 0.045), None, colour)
            slot = UsdGeom.Xform.Define(stage, f"{lp}/ToteSlot")
            c = tote_slot_centre(L, li - 1, lane - 1)
            slot.AddTranslateOp().Set(c)
            slot.AddRotateYOp().Set(TILT_DEG)
            p = slot.GetPrim()
            p.SetDocumentation(
                "Tote volume centre. Local +X = down-slope toward the robot (600 mm side), "
                "+Y = across lane (400 mm), +Z = deck normal (325 mm, open top tilted toward the robot).")
            p.CreateAttribute("packoff:toteSize", Sdf.ValueTypeNames.Float3, custom=True).Set(Gf.Vec3f(*TOTE))

    stage.GetRootLayer().Save()


def build_physics(path):
    L = layout()
    stage = _new_stage(path)
    root = stage.OverridePrim(ROOT_PRIM)
    stage.SetDefaultPrim(root)

    UsdGeom.Scope.Define(stage, f"{ROOT_PRIM}/PhysicsMaterials")

    def phys_mat(name, mu):
        m = UsdShade.Material.Define(stage, f"{ROOT_PRIM}/PhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(mu)
        api.CreateDynamicFrictionAttr(mu)
        api.CreateRestitutionAttr(0.0)
        return m

    deck_mat = phys_mat("RollerDeck", DECK_FRICTION)
    frame_mat = phys_mat("Aluminium", FRAME_FRICTION)

    def collide(rel, mat):
        prim = stage.OverridePrim(f"{ROOT_PRIM}/{rel}")
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    for li in range(1, N_LEVELS + 1):
        collide(f"Shelves/Level{li}/DeckCollider", deck_mat)
        collide(f"Shelves/Level{li}/StopBar", frame_mat)
    for side in ("Front", "Rear"):
        for j in range(N_LANES + 1):
            collide(f"Frame/Upright{side}{j + 1}", frame_mat)

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
    for name in ("sort_rack.usda", "geo.usda", "physics.usda"):
        (HERE / name).unlink(missing_ok=True)
    build_geo(HERE / "geo.usda")
    build_physics(HERE / "physics.usda")
    build_root(HERE / "sort_rack.usda")
    L = layout()
    print(f"wrote {HERE}/sort_rack.usda; height {L['height']:.3f} m, deck length {L['length']:.3f} m")


if __name__ == "__main__":
    main()
