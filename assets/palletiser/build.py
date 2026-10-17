"""Generator for the `palletiser` asset: generic compact cobot palletising cell (static prop).

Run:  ../.venv-tools/bin/python assets/palletiser/build.py
(build assets/cartons first: the stack references cartons/carton_sealed/geo.usda).
Writes palletiser.usda, geo.usda, physics.usda, README.md next to this script.

Asset frame (assets/CONVENTIONS.md): metres, Z up, origin = centre of the 3.2 (X) x 2.2 (Y)
footprint on the floor. Sealed cartons arrive travelling toward -Y on a short roller infeed
from the +Y edge to a pick position in front of the robot (robot base at the origin, on a
0.6 m lift column). Pallet station A is on +X, station B on -X. Unbranded and not animated:
the arm is posed statically mid-pick.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cartons"))

from pxr import Kind, Sdf, Usd, UsdGeom  # noqa: E402

import usdkit as K  # noqa: E402

GEN = "assets/palletiser/build.py"
NAME = "Palletiser"
R = Sdf.Path(f"/{NAME}")

HALF_X, HALF_Y = 1.60, 1.10
BED = 0.82
BED_HALF = 0.225
GUIDE_T, GUIDE_TOP = 0.012, 0.86
IN_Y0, IN_Y1 = 0.30, HALF_Y
PICK_Y = 0.62                 # carton centre at the pick position (carton spans 0.32..0.92)
CARTON = (0.400, 0.600, 0.250)
# EUR pallet 1200 x 800 x 144, long side along Y
PAL_X, PAL_Y, PAL_H = 0.800, 1.200, 0.144
STATION_X = 0.80              # A at +0.8, B at -0.8 (pallet X 0.4..1.2)
LAYERS = 3
# robot (generic 6-axis cobot, ~20 kg payload class, 1.3 m reach)
PED_H = 0.60
REACH = 1.30
L_SHOULDER = 0.18             # base flange -> shoulder axis
L_UPPER, L_FORE = 0.61, 0.57
L_W1, L_W2 = 0.12, 0.10       # wrist offsets (pointing down when picking)
TOOL_BOTTOM = BED + CARTON[2] + 0.03      # vacuum cups 30 mm above the carton top

WHITE = (0.86, 0.87, 0.88)
JOINT = (0.08, 0.09, 0.10)
GREY = (0.55, 0.57, 0.60)
STEEL = (0.62, 0.62, 0.62)
YELLOW = (0.90, 0.62, 0.02)
BLACK = (0.03, 0.03, 0.03)
WOOD = (0.55, 0.41, 0.25)


# ---------------------------------------------------------------- pieces
def eur_pallet():
    """EUR pallet: 3 bottom boards, 9 blocks (78), 3 stringer boards, 5 top boards; 22 mm boards."""
    mb = K.MeshBuilder()
    hx, hy = PAL_X / 2, PAL_Y / 2
    t, blk = 0.022, 0.078
    xs = (-hx + 0.0725, 0.0, hx - 0.0725)            # block / bottom board centres
    ys = (-hy + 0.0725, 0.0, hy - 0.0725)
    for x, w in zip(xs, (0.100, 0.145, 0.100)):
        mb.box((x - w / 2, -hy, 0.0), (x + w / 2, hy, t))
    for x in xs:
        for y in ys:
            mb.box((x - 0.0725, y - 0.0725, t), (x + 0.0725, y + 0.0725, t + blk))
    for y in ys:
        mb.box((-hx, y - 0.0725, t + blk), (hx, y + 0.0725, 2 * t + blk))
    z0, z1 = 2 * t + blk, PAL_H
    for x0, x1 in ((-0.400, -0.255), (-0.21375, -0.11375), (-0.0725, 0.0725),
                   (0.11375, 0.21375), (0.255, 0.400)):
        mb.box((x0, -hy, z0), (x1, hy, z1))
    return mb


def stack_layout():
    """4 cartons per layer x 3 layers on 800 x 1200: a 2 x 2 column stack (600 x 400 tiles
    1200 x 800 exactly, so no interlocking rotation is possible without overhang)."""
    out = []
    for layer in range(LAYERS):
        for ix, x in enumerate((-0.2, 0.2)):
            for iy, y in enumerate((-0.3, 0.3)):
                out.append((x, y, PAL_H + layer * CARTON[2]))
    return out


def arm_points():
    """Planar (YZ) elbow-up IK for a vertical-down tool over the pick point."""
    s = (0.0, PED_H + L_SHOULDER)                                     # (y, z)
    flange_z = TOOL_BOTTOM + 0.12                                      # tool stack height 0.12
    w2 = (PICK_Y, flange_z + L_W2)
    w1 = (PICK_Y, w2[1] + L_W1)
    dy, dz = w1[0] - s[0], w1[1] - s[1]
    d = math.hypot(dy, dz)
    assert d < L_UPPER + L_FORE, "pick point out of reach"
    th = math.atan2(dz, dy)
    al = math.acos((L_UPPER ** 2 + d ** 2 - L_FORE ** 2) / (2 * L_UPPER * d))
    e = (s[0] + L_UPPER * math.cos(th + al), s[1] + L_UPPER * math.sin(th + al))
    return s, e, w1, w2, flange_z


def build_geo(path: Path):
    st = K.new_stage(path, GEN)
    K.root_xform(st, NAME)
    lp = R.AppendChild("Looks")
    UsdGeom.Scope.Define(st, lp)
    m = dict(
        white=K.preview_material(st, lp.AppendChild("RobotWhite"), WHITE, 0.4),
        joint=K.preview_material(st, lp.AppendChild("RobotJoint"), JOINT, 0.5),
        grey=K.preview_material(st, lp.AppendChild("PaintedSteelGrey"), GREY, 0.5),
        steel=K.preview_material(st, lp.AppendChild("Stainless"), STEEL, 0.35, 1.0),
        yellow=K.preview_material(st, lp.AppendChild("SafetyYellow"), YELLOW, 0.45),
        black=K.preview_material(st, lp.AppendChild("BlackPolymer"), BLACK, 0.6),
        wood=K.preview_material(st, lp.AppendChild("Wood"), WOOD, 0.8, use_primvar=True),
        screen=K.preview_material(st, lp.AppendChild("Screen"), (0.01, 0.015, 0.02), 0.1,
                                  emissive=(0.03, 0.08, 0.15)),
    )

    # ---------------- infeed (short roller conveyor, flow -Y, end stop at the pick position)
    UsdGeom.Xform.Define(st, R.AppendChild("Infeed"))
    K.rollers(IN_Y0 + 0.02, IN_Y1, BED, BED_HALF).write(st, R.AppendPath("Infeed/Rollers"),
                                                         m["steel"], STEEL)
    fr = K.MeshBuilder()
    for s in (-1, 1):
        a, b = sorted((s * (BED_HALF + 0.002), s * (BED_HALF + 0.022)))
        fr.box((a, IN_Y0, BED - 0.11), (b, IN_Y1, BED - 0.01))
        g0, g1 = sorted((s * BED_HALF, s * (BED_HALF + GUIDE_T)))
        fr.box((g0, IN_Y0, GUIDE_TOP - 0.025), (g1, IN_Y1, GUIDE_TOP))
        for y in (IN_Y0 + 0.05, IN_Y1 - 0.05):
            fr.box((g0, y - 0.006, BED - 0.01), (g1, y + 0.006, GUIDE_TOP - 0.025))
            fr.box((a, y - 0.02, 0.0), (b, y + 0.02, BED - 0.11))
            fr.box((a - 0.02, y - 0.03, 0.0), (b + 0.02, y + 0.03, 0.01))
    fr.box((-BED_HALF, IN_Y0, BED), (BED_HALF, IN_Y0 + 0.02, BED + 0.08))         # end stop
    fr.write(st, R.AppendPath("Infeed/Frame"), m["steel"], STEEL)
    pe = K.MeshBuilder()                                                             # presence sensor
    pe.box((BED_HALF + GUIDE_T, PICK_Y - 0.02, BED + 0.02), (BED_HALF + 0.05, PICK_Y + 0.02, BED + 0.06))
    pe.write(st, R.AppendPath("Infeed/PresenceSensor"), m["black"], BLACK)
    UsdGeom.Scope.Define(st, R.AppendPath("Infeed/Colliders"))
    K.collider_box(st, R.AppendPath("Infeed/Colliders/Bed"),
                   (-BED_HALF, IN_Y0, BED - 0.01), (BED_HALF, IN_Y1, BED))
    K.collider_box(st, R.AppendPath("Infeed/Colliders/GuideNegX"),
                   (-BED_HALF - GUIDE_T, IN_Y0, BED), (-BED_HALF, IN_Y1, GUIDE_TOP))
    K.collider_box(st, R.AppendPath("Infeed/Colliders/GuidePosX"),
                   (BED_HALF, IN_Y0, BED), (BED_HALF + GUIDE_T, IN_Y1, GUIDE_TOP))
    K.collider_box(st, R.AppendPath("Infeed/Colliders/EndStop"),
                   (-BED_HALF, IN_Y0, BED), (BED_HALF, IN_Y0 + 0.02, BED + 0.08))
    K.guide_point(st, R.AppendPath("Infeed/EntryPoint"), (0.0, IN_Y1, BED),
                  "Sealed cartons enter here (+Y edge, bed top 0.82 m) travelling toward -Y.")
    K.guide_point(st, R.AppendPath("Infeed/PickPoint"), (0.0, PICK_Y, BED),
                  "Carton bottom-centre at the pick position (against the end stop).")

    # ---------------- pallet stations
    for stn, sx in (("StationA", 1), ("StationB", -1)):
        cx = sx * STATION_X
        UsdGeom.Xform.Define(st, R.AppendChild(stn))
        pal = K.xform(st, R.AppendPath(f"{stn}/Pallet"), translate=(cx, 0.0, 0.0))
        eur_pallet().write(st, pal.GetPath().AppendChild("Boards"), m["wood"], WOOD,
                           jitter=0.10, seed=7 if sx > 0 else 8)
        UsdGeom.Scope.Define(st, pal.GetPath().AppendChild("Colliders"))
        K.collider_box(st, pal.GetPath().AppendPath("Colliders/Box"),
                       (-PAL_X / 2, -PAL_Y / 2, 0.0), (PAL_X / 2, PAL_Y / 2, PAL_H))
        K.guide_point(st, R.AppendPath(f"{stn}/StackOrigin"), (cx, 0.0, PAL_H),
                      "Pallet top-surface centre: origin for the stacking pattern (layer 1 "
                      "carton bottoms at this Z; X across 800, Y along 1200).")
        mk = K.MeshBuilder()                                          # yellow floor marking
        x0, x1, y0, y1, w = cx - 0.45, cx + 0.45, -0.65, 0.65, 0.05
        for lo, hi in (((x0, y0, 0), (x1, y0 + w, 0.001)), ((x0, y1 - w, 0), (x1, y1, 0.001)),
                       ((x0, y0, 0), (x0 + w, y1, 0.001)), ((x1 - w, y0, 0), (x1, y1, 0.001))):
            mk.box(lo, hi)
        mk.write(st, R.AppendPath(f"{stn}/FloorMarking"), m["yellow"], YELLOW)
    UsdGeom.Xform.Define(st, R.AppendPath("StationA/Stack"))
    for i, (x, y, z) in enumerate(stack_layout()):
        c = K.xform(st, R.AppendPath(f"StationA/Stack/Carton_{i:02d}"),
                    translate=(STATION_X + x, y, z))
        c.GetPrim().GetReferences().AddReference("../cartons/carton_sealed/geo.usda")
        Usd.ModelAPI(c).SetKind(Kind.Tokens.subcomponent)

    # ---------------- robot: lift column + 6-axis cobot posed mid-pick
    rob = UsdGeom.Xform.Define(st, R.AppendChild("Robot"))
    a = rob.GetPrim().CreateAttribute("packoff:reach", Sdf.ValueTypeNames.Float)
    a.Set(REACH)
    a.SetDocumentation("Nominal cobot reach (m), generic ~20 kg / 1.3 m class.")
    ped = UsdGeom.Xform.Define(st, R.AppendPath("Robot/Pedestal"))
    pm = K.MeshBuilder()
    pm.box((-0.30, -0.30, 0.0), (0.30, 0.30, 0.02))                    # floor plate
    pm.box((-0.16, -0.16, 0.02), (0.16, 0.16, 0.35))                   # lift column outer
    pm.box((-0.14, -0.14, 0.35), (0.14, 0.14, PED_H - 0.02))           # telescopic stage
    pm.box((-0.15, -0.15, PED_H - 0.02), (0.15, 0.15, PED_H))          # top plate
    pm.write(st, ped.GetPath().AppendChild("Column"), m["grey"], GREY)
    UsdGeom.Scope.Define(st, ped.GetPath().AppendChild("Colliders"))
    K.collider_box(st, ped.GetPath().AppendPath("Colliders/Box"), (-0.16, -0.16, 0.0), (0.16, 0.16, PED_H))
    K.collider_box(st, ped.GetPath().AppendPath("Colliders/FloorPlate"), (-0.30, -0.30, 0.0), (0.30, 0.30, 0.02))

    s, e, w1, w2, fz = arm_points()
    P = lambda yz, x=0.0: (x, yz[0], yz[1])                            # noqa: E731
    arm = R.AppendPath("Robot/Arm")
    UsdGeom.Xform.Define(st, arm)
    parts = {}
    b = K.MeshBuilder()                                                # J1 base
    b.cylinder((0, 0, PED_H), (0, 0, PED_H + 0.10), 0.095, 24)
    parts["Base"] = (b, "joint")
    sh = K.MeshBuilder()                                               # J1 rotor + J2 housing
    sh.cylinder((0, 0, PED_H + 0.10), (0, 0, s[1]), 0.08, 24)
    sh.cylinder((-0.10, s[0], s[1]), (0.10, s[0], s[1]), 0.08, 24)
    parts["Shoulder"] = (sh, "joint")
    ua = K.MeshBuilder()
    ua.capsule(P(s, 0.10), P(e, 0.10), 0.06, 20)
    parts["UpperArm"] = (ua, "white")
    el = K.MeshBuilder()                                               # J3 housing
    el.cylinder((-0.03, e[0], e[1]), (0.17, e[0], e[1]), 0.07, 24)
    parts["Elbow"] = (el, "joint")
    fa = K.MeshBuilder()
    fa.capsule(P(e, 0.02), P(w1, 0.02), 0.048, 20)
    parts["Forearm"] = (fa, "white")
    j4 = K.MeshBuilder()                                               # J4 (axis X)
    j4.cylinder((-0.04, w1[0], w1[1]), (0.08, w1[0], w1[1]), 0.055, 20)
    parts["Wrist1"] = (j4, "joint")
    j5 = K.MeshBuilder()                                               # link down + J5 (axis Y)
    j5.cylinder((0.0, w1[0], w1[1]), (0.0, w2[0], w2[1]), 0.045, 20)
    j5.cylinder((0.0, w2[0] - 0.055, w2[1]), (0.0, w2[0] + 0.055, w2[1]), 0.05, 20)
    parts["Wrist2"] = (j5, "joint")
    j6 = K.MeshBuilder()                                               # J6 + tool flange (axis Z)
    j6.cylinder((0.0, w2[0], w2[1] - 0.04), (0.0, w2[0], fz), 0.045, 20)
    parts["Wrist3"] = (j6, "white")
    tl = K.MeshBuilder()                                               # vacuum gripper
    tl.cylinder((0.0, PICK_Y, fz), (0.0, PICK_Y, fz - 0.05), 0.04, 16)
    tl.box((-0.16, PICK_Y - 0.26, fz - 0.08), (0.16, PICK_Y + 0.26, fz - 0.05))
    tl.box((-0.03, PICK_Y - 0.25, fz - 0.05), (0.03, PICK_Y + 0.25, fz - 0.035))   # vacuum manifold
    for cx in (-0.10, 0.10):
        for cy in (-0.18, 0.0, 0.18):
            tl.cylinder((cx, PICK_Y + cy, fz - 0.08), (cx, PICK_Y + cy, TOOL_BOTTOM + 0.015), 0.01, 10)
            tl.cylinder((cx, PICK_Y + cy, TOOL_BOTTOM + 0.015), (cx, PICK_Y + cy, TOOL_BOTTOM),
                        0.012, 12, r1=0.025)
    parts["Tool"] = (tl, "black")
    for n, (mb, look) in parts.items():
        mb.write(st, arm.AppendChild(n), m[look], {"white": WHITE, "joint": JOINT, "black": BLACK}[look])
    cbl = K.MeshBuilder()                                              # vacuum hose
    cbl.cylinder((0.12, 0.0, PED_H + 0.05), (0.12, e[0], e[1] - 0.05), 0.012, 8)
    cbl.cylinder((0.12, e[0], e[1] - 0.05), (0.08, w1[0] - 0.05, w1[1]), 0.012, 8)
    cbl.write(st, R.AppendPath("Robot/Hose"), m["black"], BLACK)

    # ---------------- control cabinet (behind the robot, -Y)
    UsdGeom.Xform.Define(st, R.AppendChild("ControlCabinet"))
    cc = K.MeshBuilder()
    cc.box((-0.30, -1.02, 0.10), (0.30, -0.72, 1.30))
    cc.box((-0.28, -1.00, 0.0), (-0.22, -0.94, 0.10))
    cc.box((0.22, -1.00, 0.0), (0.28, -0.94, 0.10))
    cc.box((-0.28, -0.80, 0.0), (-0.22, -0.74, 0.10))
    cc.box((0.22, -0.80, 0.0), (0.28, -0.74, 0.10))
    cc.write(st, R.AppendPath("ControlCabinet/Body"), m["grey"], GREY)
    hm = K.MeshBuilder()
    hm.box((-0.15, -0.72, 0.95), (0.15, -0.70, 1.15))
    hm.write(st, R.AppendPath("ControlCabinet/HMI"), m["screen"], (0.01, 0.015, 0.02))
    dt = K.MeshBuilder()
    dt.box((0.20, -0.72, 0.55), (0.22, -0.69, 0.70))                   # handle
    dt.box((-0.002, -0.721, 0.15), (0.002, -0.719, 1.25))              # door seam
    dt.write(st, R.AppendPath("ControlCabinet/Details"), m["black"], BLACK)
    UsdGeom.Scope.Define(st, R.AppendPath("ControlCabinet/Colliders"))
    K.collider_box(st, R.AppendPath("ControlCabinet/Colliders/Box"), (-0.30, -1.02, 0.0), (0.30, -0.72, 1.30))
    K.stack_light(st, R.AppendPath("ControlCabinet/StackLight"), (0.20, -0.87, 1.30), lp,
                  pole_h=0.05, seg_h=0.05, r=0.03)

    # ---------------- perimeter: yellow bollards + light-curtain posts at the open +-X sides
    UsdGeom.Xform.Define(st, R.AppendChild("Perimeter"))
    UsdGeom.Scope.Define(st, R.AppendPath("Perimeter/Colliders"))
    br, fl = 0.05, 0.07                              # post radius, base flange radius
    pts = [(sx * (HALF_X - fl), sy * (HALF_Y - fl)) for sx in (-1, 1) for sy in (-1, 1)]
    pts += [(sx * 0.80, sy * (HALF_Y - fl)) for sx in (-1, 1) for sy in (-1, 1)]
    for i, (x, y) in enumerate(pts):
        bm = K.MeshBuilder()
        bm.cylinder((x, y, 0.0), (x, y, 0.95), br, 16)
        bm.cylinder((x, y, 0.0), (x, y, 0.01), fl, 16)
        bm.write(st, R.AppendPath(f"Perimeter/Bollard_{i:02d}"), m["yellow"], YELLOW)
        sm = K.MeshBuilder()
        for z in (0.70, 0.82):
            sm.cylinder((x, y, z), (x, y, z + 0.05), br + 0.001, 16)
        sm.write(st, R.AppendPath(f"Perimeter/Bollard_{i:02d}_Stripes"), m["black"], BLACK)
        K.collider_box(st, R.AppendPath(f"Perimeter/Colliders/Bollard_{i:02d}"),
                       (x - br, y - br, 0.0), (x + br, y + br, 0.95))
    k = 0
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (HALF_X - 0.10), sy * 0.70
            pm = K.MeshBuilder()
            pm.box((x - 0.03, y - 0.03, 0.0), (x + 0.03, y + 0.03, 1.40))
            pm.box((x - 0.10, y - 0.10, 0.0), (x + 0.10, y + 0.10, 0.01))
            pm.write(st, R.AppendPath(f"Perimeter/ScannerPost_{k:02d}"), m["yellow"], YELLOW)
            sm = K.MeshBuilder()                                       # sensor strip facing the gap
            sm.box((x - 0.02, y - sy * 0.033, 0.15), (x + 0.02, y - sy * 0.030, 1.35))
            sm.write(st, R.AppendPath(f"Perimeter/ScannerPost_{k:02d}_Strip"), m["black"], BLACK)
            K.collider_box(st, R.AppendPath(f"Perimeter/Colliders/ScannerPost_{k:02d}"),
                           (x - 0.03, y - 0.03, 0.0), (x + 0.03, y + 0.03, 1.40))
            k += 1
    st.GetRootLayer().Save()
    return st


def collider_map(st):
    out = {}
    for p in st.Traverse():
        sp = str(p.GetPath())
        if p.IsA(UsdGeom.Cube) and "/Colliders/" in sp:
            out[sp] = ("Cardboard" if "/Stack/" in sp else "Wood" if "/Pallet/" in sp else
                       "Stainless" if "/Infeed/" in sp else "PaintedSteel")
    return out


def write_readme(tris):
    s, e, w1, w2, fz = arm_points()
    rows = [
        ("Type", "compact cobot palletiser, generic", "static prop, not animated, unbranded; "
         "inspired by small-footprint cobot palletisers (~2.2 x 3.2 m, ~20 kg / 1.3 m-reach cobot "
         "on a lift column)", "task spec"),
        ("Footprint", "~2.2 x 3.2 m", "3.20 (X) x 2.20 (Y) m; height 1.52 m (arm / cabinet stack light)",
         "task spec; X = pallet-to-pallet axis ASSUMPTION"),
        ("Robot", "6-axis cobot, white + dark joints, ~1.3 m reach",
         f"capsule/cylinder links: shoulder {L_SHOULDER} m, upper arm {L_UPPER} m, forearm "
         f"{L_FORE} m, wrist {L_W1}/{L_W2} m; `packoff:reach` = {REACH} m on `/Palletiser/Robot`",
         "task spec; link lengths ASSUMPTION (typical 1.3 m-class cobot)"),
        ("Pedestal", "~0.6 m lift column", f"two-stage column 0.32 m square on a 0.6 m floor plate, "
         f"top {PED_H} m, at the origin", "task spec"),
        ("Pose", "mid-pick over the infeed", f"vertical vacuum tool over the pick point, cups "
         f"{TOOL_BOTTOM:.2f} m (30 mm above the carton top); elbow at (y {e[0]:.3f}, z {e[1]:.3f})",
         "planar IK in build.py"),
        ("Tool", "gripper/vacuum", "320 x 520 mm plate with 6 suction cups", "ASSUMPTION"),
        ("Infeed", "short gravity/roller conveyor, top 0.82 m",
         f"Ø50 rollers, 450 mm between guides (guide top 0.86 m), Y {IN_Y0}..{IN_Y1}, end stop at "
         f"Y {IN_Y0}; pick point carton centre at Y {PICK_Y}", "task spec / conveyor asset"),
        ("Pallets", "2 x EUR 1200 x 800 x 144, wood",
         "boards 22 mm, blocks 78 mm (3 bottom boards, 9 blocks, 3 stringers, 5 top boards); "
         f"station A centre (+{STATION_X}, 0), station B (-{STATION_X}, 0), 1200 along Y",
         "EN 13698-1"),
        ("Stack A", "4 per layer x 3 layers of carton_sealed",
         "2 x 2 column stack (600 x 400 tiles 1200 x 800 exactly; an interlocked/rotated layer "
         "would overhang, so layers are column-stacked), 12 cartons referencing "
         "`../cartons/carton_sealed/geo.usda`, top at 0.894 m", "task spec; layout computed"),
        ("Station B", "empty pallet", "pallet only, no Stack prim", "task spec"),
        ("Perimeter", "bollards + light-curtain posts, no fence",
         "8 yellow Ø100 x 950 mm bollards (black bands) on the +-Y edges; 4 yellow 1.4 m "
         "light-curtain posts with dark sensor strips framing the open +-X sides; floor markings",
         "task spec"),
        ("Control cabinet", "-", "0.6 x 0.3 x 1.3 m grey cabinet with HMI and stack light behind "
         "the robot (-Y)", "ASSUMPTION"),
        ("Colliders", "infeed, pallets, pedestal, carton stack",
         "static guide-purpose boxes: Infeed/Colliders/{Bed,GuideNegX,GuidePosX,EndStop} (stainless), "
         "Station*/Pallet/Colliders/Box (wood 0.5), Robot/Pedestal/Colliders/{Box,FloorPlate}, one box "
         "per stacked carton (cardboard), cabinet, bollards and posts (painted 0.4). No rigid bodies.",
         "task spec"),
        ("Triangles", "< 60k", str(tris), "build.py (incl. referenced cartons)"),
    ]
    lines = ["# palletiser — generic compact cobot palletising cell", "",
             f"Generated by `{GEN}`; do not hand-edit the .usda files. Unbranded, no logos.",
             "Placement: put `Infeed/EntryPoint` (asset (0, +1.10, 0.82)) at the case sealer's "
             "`Outfeed/ExitPoint`; cartons travel toward -Y.", ""]
    lines += K.readme_table(rows)
    lines += ["", "## Prim paths", "",
              "- `/Palletiser/Infeed/EntryPoint` (Xform, (0, 1.10, 0.82)), `/Palletiser/Infeed/PickPoint` "
              f"(0, {PICK_Y}, 0.82)",
              f"- `/Palletiser/StationA/StackOrigin` (+{STATION_X}, 0, 0.144), "
              f"`/Palletiser/StationB/StackOrigin` (-{STATION_X}, 0, 0.144): pallet top centres",
              "- `/Palletiser/StationA/Stack/Carton_00..11` (references), `/Palletiser/Station{A,B}/Pallet`",
              "- `/Palletiser/Robot/Pedestal`, `/Palletiser/Robot/Arm/{Base,Shoulder,UpperArm,Elbow,"
              "Forearm,Wrist1,Wrist2,Wrist3,Tool}`, `/Palletiser/Robot/Hose`",
              "- `/Palletiser/ControlCabinet/{Body,HMI,Details,StackLight}`",
              "- `/Palletiser/Perimeter/{Bollard_NN,ScannerPost_NN}`", ""]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


def main():
    st = build_geo(HERE / "geo.usda")
    K.write_physics(HERE, NAME, GEN, collider_map(st))
    K.write_root(HERE, "palletiser", NAME, GEN)
    tris = K.count_triangles(Usd.Stage.Open(str(HERE / "palletiser.usda"), Usd.Stage.LoadAll))
    write_readme(tris)
    print(f"palletiser: {tris} triangles")


if __name__ == "__main__":
    main()
