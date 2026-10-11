"""Generator for the `case_erector` asset: generic automatic carton erector (static prop).

Run:  /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/case_erector/build.py
(build assets/cartons first: the magazine references cartons/carton_flat_stack).
Writes case_erector.usda, geo.usda, physics.usda, README.md next to this script.

Asset frame (assets/CONVENTIONS.md): metres, Z up, origin = centre of the 1.4 (X) x 2.2 (Y)
footprint on the floor. Flow is toward -Y: the outfeed bed (top 0.82 m, centred on X = 0) ends
at the asset's -Y edge so it butts against the conveyor's +Y end. Not animated: an erected
carton_open is simply spawned at /CaseErector/Outfeed/SpawnPoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cartons"))

from pxr import Kind, Sdf, Usd, UsdGeom  # noqa: E402

import usdkit as K  # noqa: E402

GEN = "assets/case_erector/build.py"
NAME = "CaseErector"
R = Sdf.Path(f"/{NAME}")

X0, X1, Y0, Y1, HEIGHT = -0.70, 0.70, -1.10, 1.10, 1.90
# main frame (forming section)
FX0, FX1, FY0, FY1 = -0.45, 0.55, -0.35, 1.10
POST = 0.060
CAB_TOP = 0.70                 # sheet-steel base cabinet top
GUARD_Z0, GUARD_Z1 = 0.75, HEIGHT - POST
PANEL_T = 0.006
# outfeed
BED = 0.82
BED_HALF = 0.225               # 450 mm between guides (same as the conveyor)
OUT_Y0, OUT_Y1 = Y0, -0.05
GUIDE_T, GUIDE_TOP = 0.012, 0.86
OPEN_HALF_X, OPEN_TOP = 0.25, 1.15
SPAWN_Y = -0.70
# magazine (blanks standing on edge, stacked along +X)
MAG_Z = 0.70
MAG_Y0, MAG_Y1 = 0.0, 1.0
STACK_X0 = -0.66
# control cabinet on the +X side
CC_X0, CC_X1, CC_Y0, CC_Y1, CC_TOP = 0.55, 0.66, 0.30, 0.90, 1.50

GREY = (0.60, 0.62, 0.64)
DARK_GREY = (0.18, 0.19, 0.20)
STEEL = (0.62, 0.62, 0.62)
BLACK = (0.03, 0.03, 0.03)
PC = (0.85, 0.90, 0.95)
TAPE = (0.36, 0.20, 0.07)


def looks(st):
    lp = R.AppendChild("Looks")
    UsdGeom.Scope.Define(st, lp)
    return dict(
        paint=K.preview_material(st, lp.AppendChild("PaintedSteelLightGrey"), GREY, 0.5),
        dark=K.preview_material(st, lp.AppendChild("PaintedSteelDark"), DARK_GREY, 0.55),
        steel=K.preview_material(st, lp.AppendChild("Stainless"), STEEL, 0.35, 1.0),
        black=K.preview_material(st, lp.AppendChild("BlackPolymer"), BLACK, 0.6),
        pc=K.preview_material(st, lp.AppendChild("Polycarbonate"), PC, 0.05, 0.0,
                              opacity=0.2, ior=1.58),
        screen=K.preview_material(st, lp.AppendChild("Screen"), (0.01, 0.015, 0.02), 0.1,
                                  emissive=(0.03, 0.08, 0.15)),
        red=K.preview_material(st, lp.AppendChild("EStopRed"), (0.7, 0.02, 0.02), 0.3),
        yellow=K.preview_material(st, lp.AppendChild("EStopYellow"), (0.9, 0.7, 0.02), 0.4),
        tape=K.preview_material(st, lp.AppendChild("Tape"), TAPE, 0.3),
    ), lp


def guard_panels():
    """name -> (lo, hi) of every polycarbonate panel (thin boxes in the frame faces)."""
    t, z0, z1 = PANEL_T, GUARD_Z0, GUARD_Z1
    xi0, xi1 = FX0 + POST / 2, FX1 - POST / 2         # panel planes on the post centre lines
    yi0, yi1 = FY0 + POST / 2, FY1 - POST / 2
    return {
        "PanelPosX": ((xi1 - t / 2, yi0, z0), (xi1 + t / 2, yi1, z1)),
        "PanelPosY": ((xi0, yi1 - t / 2, z0), (xi1, yi1 + t / 2, z1)),
        # -X side: magazine opening under the upper panel between Y = MAG_Y0 .. MAG_Y1
        "PanelNegXFront": ((xi0 - t / 2, yi0, z0), (xi0 + t / 2, MAG_Y0, z1)),
        "PanelNegXBack": ((xi0 - t / 2, MAG_Y1, z0), (xi0 + t / 2, yi1, z1)),
        "PanelNegXTop": ((xi0 - t / 2, MAG_Y0, MAG_Z + 0.70), (xi0 + t / 2, MAG_Y1, z1)),
        # -Y end: outfeed tunnel opening X +-0.25, Z up to 1.15
        "PanelNegYLeft": ((xi0, yi0 - t / 2, z0), (-OPEN_HALF_X, yi0 + t / 2, z1)),
        "PanelNegYRight": ((OPEN_HALF_X, yi0 - t / 2, z0), (xi1, yi0 + t / 2, z1)),
        "PanelNegYTop": ((-OPEN_HALF_X, yi0 - t / 2, OPEN_TOP), (OPEN_HALF_X, yi0 + t / 2, z1)),
    }


def build_geo(path: Path):
    st = K.new_stage(path, GEN)
    K.root_xform(st, NAME)
    m, lp = looks(st)

    # ---------------- frame + base cabinet
    UsdGeom.Xform.Define(st, R.AppendChild("Frame"))
    posts = K.MeshBuilder()
    h = POST / 2
    for x in (FX0 + h, FX1 - h):
        for y in (FY0 + h, FY1 - h, MAG_Y0, MAG_Y1 if x < 0 else 0.5):
            posts.box((x - h, y - h, 0.05), (x + h, y + h, HEIGHT))
    for x in (-OPEN_HALF_X - 0.02, OPEN_HALF_X + 0.02):          # outfeed opening jambs
        posts.box((x - 0.02, FY0, CAB_TOP), (x + 0.02, FY0 + POST, HEIGHT))
    for z in (CAB_TOP + 0.02, HEIGHT - h):                          # ring beams
        for x in (FX0 + h, FX1 - h):
            posts.box((x - h, FY0, z - h), (x + h, FY1, z + h))
        for y in (FY0 + h, FY1 - h):
            posts.box((FX0, y - h, z - h), (FX1, y + h, z + h))
    posts.box((-OPEN_HALF_X - 0.04, FY0, OPEN_TOP), (OPEN_HALF_X + 0.04, FY0 + POST, OPEN_TOP + 0.04))
    posts.write(st, R.AppendPath("Frame/Posts"), m["paint"], GREY)

    cab = K.MeshBuilder()
    cab.box((FX0 + 0.01, FY0 + 0.01, 0.06), (FX1 - 0.01, FY1 - 0.01, CAB_TOP))
    cab.write(st, R.AppendPath("Frame/BaseCabinet"), m["paint"], GREY)
    doors = K.MeshBuilder()                                         # door seams + handles
    for y in (0.0, 0.55):
        doors.box((FX1 - 0.012, y - 0.004, 0.10), (FX1 - 0.005, y + 0.004, CAB_TOP - 0.05))
        doors.box((FX1 - 0.03, y + 0.20, 0.35), (FX1 + 0.0, y + 0.22, 0.45))
    doors.write(st, R.AppendPath("Frame/CabinetDetails"), m["dark"], DARK_GREY)
    feet = K.MeshBuilder()
    for x in (FX0 + 0.05, FX1 - 0.05):
        for y in (FY0 + 0.05, FY1 - 0.05):
            feet.cylinder((x, y, 0.0), (x, y, 0.012), 0.04, 12)
            feet.cylinder((x, y, 0.012), (x, y, 0.06), 0.012, 8)
    feet.write(st, R.AppendPath("Frame/Feet"), m["black"], BLACK)
    UsdGeom.Scope.Define(st, R.AppendPath("Frame/Colliders"))
    K.collider_box(st, R.AppendPath("Frame/Colliders/BaseCabinet"),
                   (FX0, FY0, 0.0), (FX1, FY1, CAB_TOP))

    # ---------------- polycarbonate guards (+ static colliders)
    UsdGeom.Xform.Define(st, R.AppendChild("Guards"))
    pan = K.MeshBuilder()
    for lo, hi in guard_panels().values():
        pan.box(lo, hi)
    pan.write(st, R.AppendPath("Guards/Panels"), m["pc"], PC)
    UsdGeom.Scope.Define(st, R.AppendPath("Guards/Colliders"))
    for n, (lo, hi) in guard_panels().items():
        K.collider_box(st, R.AppendPath(f"Guards/Colliders/{n}"), lo, hi)
    hinges = K.MeshBuilder()                                        # door hinges/handles
    xi1 = FX1 - POST / 2
    for y in (0.0, 0.9):
        for z in (0.95, 1.65):
            hinges.box((xi1 + 0.003, y - 0.02, z - 0.03), (xi1 + 0.02, y + 0.02, z + 0.03))
    hinges.box((xi1 + 0.003, 0.48, 1.20), (xi1 + 0.04, 0.50, 1.40))
    hinges.write(st, R.AppendPath("Guards/Hardware"), m["black"], BLACK)

    # ---------------- forming mechanism (visual only, seen through the guards)
    UsdGeom.Xform.Define(st, R.AppendChild("Mechanism"))
    fy = 0.35                                    # forming station centre (Y)
    fb = K.MeshBuilder()                         # forming box: 4 plates around a 400x600 box
    for s in (-1, 1):
        fb.box((s * 0.215 - 0.005, fy - 0.31, BED + 0.02), (s * 0.215 + 0.005, fy + 0.31, BED + 0.24))
        fb.box((-0.22, fy + s * 0.315 - 0.005, BED + 0.15), (0.22, fy + s * 0.315 + 0.005, BED + 0.24))
    fb.box((-0.22, fy - 0.35, BED - 0.01), (0.22, fy + 0.35, BED))  # dead plate, top 0.82
    for s in (-1, 1):                            # flap folder ploughs
        fb.box((s * 0.24 - 0.004, fy - 0.30, BED - 0.15), (s * 0.24 + 0.004, fy + 0.30, BED - 0.03))
    fb.write(st, R.AppendPath("Mechanism/FormingBox"), m["steel"], STEEL)
    va = K.MeshBuilder()                         # vacuum pick arm: guided cylinder + plate
    va.box((-0.30, fy - 0.03, HEIGHT - 0.14), (0.30, fy + 0.03, HEIGHT - 0.08))      # cross beam
    va.cylinder((0.0, fy, HEIGHT - 0.08), (0.0, fy, 1.30), 0.035, 16)
    va.cylinder((0.0, fy, 1.30), (0.0, fy, 1.16), 0.012, 10)
    for gx in (-0.08, 0.08):
        va.cylinder((gx, fy, HEIGHT - 0.08), (gx, fy, 1.20), 0.008, 8)
    va.box((-0.14, fy - 0.20, 1.14), (0.14, fy + 0.20, 1.16))
    va.write(st, R.AppendPath("Mechanism/VacuumArm"), m["steel"], STEEL)
    cups = K.MeshBuilder()
    for cx in (-0.10, 0.10):
        for cy in (-0.15, 0.15):
            cups.cylinder((cx, fy + cy, 1.14), (cx, fy + cy, 1.11), 0.012, 12, r1=0.022)
    cups.write(st, R.AppendPath("Mechanism/SuctionCups"), m["black"], BLACK)
    th = K.MeshBuilder()                         # bottom taping unit under the dead plate
    th.box((-0.06, fy - 0.30, 0.40), (0.06, fy + 0.25, BED - 0.02))
    th.write(st, R.AppendPath("Mechanism/TapeHead"), m["steel"], STEEL)
    tr = K.MeshBuilder()
    tr.cylinder((-0.03, fy + 0.15, 0.55), (0.03, fy + 0.15, 0.55), 0.10, 24)
    tr.write(st, R.AppendPath("Mechanism/TapeRoll"), m["tape"], TAPE)

    # ---------------- blank magazine on the -X side
    UsdGeom.Xform.Define(st, R.AppendChild("Magazine"))
    mg = K.MeshBuilder()
    for y in (MAG_Y0 + 0.10, MAG_Y1 - 0.10):                       # bottom rails along X
        mg.box((X0, y - 0.02, MAG_Z - 0.04), (FX0, y + 0.02, MAG_Z))
        mg.box((X0 + 0.02, y - 0.02, 0.0), (X0 + 0.06, y + 0.02, MAG_Z - 0.04))   # legs
        mg.box((X0, y - 0.04, 0.0), (X0 + 0.08, y + 0.04, 0.008))                 # foot plates
    for y in (MAG_Y0 - 0.025, MAG_Y1 + 0.005):                      # side guides
        mg.box((X0 + 0.02, y, MAG_Z + 0.10), (FX0, y + 0.02, MAG_Z + 0.12))
        mg.box((X0 + 0.02, y, MAG_Z + 0.50), (FX0, y + 0.02, MAG_Z + 0.52))
    mg.write(st, R.AppendPath("Magazine/Rails"), m["steel"], STEEL)
    pu = K.MeshBuilder()                                            # pusher paddle + cylinder
    pu.box((STACK_X0 - 0.02, MAG_Y0 + 0.08, MAG_Z + 0.05), (STACK_X0 - 0.005, MAG_Y1 - 0.08, MAG_Z + 0.60))
    pu.cylinder((X0, 0.5, MAG_Z + 0.32), (STACK_X0 - 0.02, 0.5, MAG_Z + 0.32), 0.02, 12)
    pu.write(st, R.AppendPath("Magazine/Pusher"), m["dark"], DARK_GREY)
    bs = K.xform(st, R.AppendPath("Magazine/BlankStack"),
                 translate=(STACK_X0, (MAG_Y0 + MAG_Y1) / 2, MAG_Z + 0.325), rotate=(0, 90, 0))
    bs.GetPrim().GetReferences().AddReference("../cartons/carton_flat_stack/carton_flat_stack.usda")
    Usd.ModelAPI(bs).SetKind(Kind.Tokens.subcomponent)
    bs.GetPrim().SetDocumentation("20 knocked-down blanks standing on edge (carton_flat_stack "
                                  "rotated +90 deg about Y: the stack grows along +X).")

    # ---------------- control panel (+X side)
    UsdGeom.Xform.Define(st, R.AppendChild("ControlPanel"))
    cc = K.MeshBuilder()
    cc.box((CC_X0, CC_Y0, CAB_TOP), (CC_X1, CC_Y1, CC_TOP))
    cc.write(st, R.AppendPath("ControlPanel/Cabinet"), m["paint"], GREY)
    UsdGeom.Xform.Define(st, R.AppendPath("ControlPanel/HMI"))
    hb = K.MeshBuilder()
    hb.box((CC_X1, 0.45, 1.15), (X1 - 0.004, 0.75, 1.40))
    hb.write(st, R.AppendPath("ControlPanel/HMI/Housing"), m["dark"], DARK_GREY)
    sc = K.MeshBuilder()
    sc.box((X1 - 0.004, 0.47, 1.18), (X1, 0.73, 1.37))
    sc.write(st, R.AppendPath("ControlPanel/HMI/Screen"), m["screen"], (0.01, 0.015, 0.02))
    es = K.MeshBuilder()
    es.cylinder((CC_X1, 0.82, 1.05), (CC_X1 + 0.03, 0.82, 1.05), 0.022, 16)
    es.cylinder((CC_X1 + 0.03, 0.82, 1.05), (CC_X1 + 0.04, 0.82, 1.05), 0.03, 16)
    es.write(st, R.AppendPath("ControlPanel/EStop"), m["red"], (0.7, 0.02, 0.02))
    ep = K.MeshBuilder()
    ep.box((CC_X1, 0.77, 1.00), (CC_X1 + 0.004, 0.87, 1.10))
    ep.write(st, R.AppendPath("ControlPanel/EStopPlate"), m["yellow"], (0.9, 0.7, 0.02))
    K.stack_light(st, R.AppendChild("StackLight"), ((CC_X0 + CC_X1) / 2, 0.40, CC_TOP), lp)

    # ---------------- outfeed bed (rollers top 0.82, ends at the -Y edge)
    UsdGeom.Xform.Define(st, R.AppendChild("Outfeed"))
    K.rollers(OUT_Y0, OUT_Y1, BED, BED_HALF).write(st, R.AppendPath("Outfeed/Rollers"),
                                                    m["steel"], STEEL)
    of = K.MeshBuilder()
    for s in (-1, 1):
        x_in, x_out = s * (BED_HALF + 0.002), s * (BED_HALF + 0.022)
        lo_x, hi_x = min(x_in, x_out), max(x_in, x_out)
        of.box((lo_x, OUT_Y0, BED - 0.11), (hi_x, OUT_Y1, BED - 0.01))           # side frames
        g0, g1 = (BED_HALF, BED_HALF + GUIDE_T) if s > 0 else (-BED_HALF - GUIDE_T, -BED_HALF)
        of.box((g0, OUT_Y0, GUIDE_TOP - 0.025), (g1, FY0, GUIDE_TOP))              # guide rails
        for y in (OUT_Y0 + 0.05, -0.60):
            of.box((g0, y - 0.006, BED - 0.01), (g1, y + 0.006, GUIDE_TOP - 0.025))  # guide posts
        of.box((lo_x, OUT_Y0 + 0.02, 0.0), (hi_x, OUT_Y0 + 0.06, BED - 0.11))     # leg
        of.box((lo_x - 0.02, OUT_Y0 + 0.01, 0.0), (hi_x + 0.02, OUT_Y0 + 0.07, 0.01))
    of.box((-BED_HALF, OUT_Y0 + 0.02, 0.15), (BED_HALF, OUT_Y0 + 0.05, 0.18))      # leg brace
    of.write(st, R.AppendPath("Outfeed/Frame"), m["steel"], STEEL)
    UsdGeom.Scope.Define(st, R.AppendPath("Outfeed/Colliders"))
    K.collider_box(st, R.AppendPath("Outfeed/Colliders/Bed"),
                   (-BED_HALF, OUT_Y0, BED - 0.01), (BED_HALF, OUT_Y1, BED))
    K.collider_box(st, R.AppendPath("Outfeed/Colliders/GuideNegX"),
                   (-BED_HALF - GUIDE_T, OUT_Y0, BED), (-BED_HALF, FY0, GUIDE_TOP))
    K.collider_box(st, R.AppendPath("Outfeed/Colliders/GuidePosX"),
                   (BED_HALF, OUT_Y0, BED), (BED_HALF + GUIDE_T, FY0, GUIDE_TOP))
    K.guide_point(st, R.AppendPath("Outfeed/SpawnPoint"), (0.0, SPAWN_Y, BED),
                  "Where an erected carton_open appears: carton origin (bottom centre) on the "
                  "outfeed bed top (Z 0.82), long side along Y, flow toward -Y.")
    st.GetRootLayer().Save()
    return st


def collider_map(st):
    out = {}
    for p in st.Traverse():
        if p.IsA(UsdGeom.Cube) and "/Colliders/" in str(p.GetPath()) \
                and "Magazine/BlankStack" not in str(p.GetPath()):
            s = str(p.GetPath())
            out[s] = ("Stainless" if "/Outfeed/" in s else
                      "Polycarbonate" if "/Guards/" in s else "PaintedSteel")
    return out


def write_readme(tris):
    rows = [
        ("Type", "automatic carton erector, generic", "static prop, not animated; cartons spawn "
         "at `Outfeed/SpawnPoint`", "task spec"),
        ("Footprint X x Y x H", "~1.4 x 2.2 x 1.9 m", "1.40 x 2.20 x 1.90 m (visual bbox)", "task spec"),
        ("Frame", "painted steel, light grey", f"60 mm posts/beams, base cabinet to 0.70 m, {GREY}",
         "task spec"),
        ("Guards", "clear polycarbonate", "6 mm panels, opacity 0.2, IOR 1.58, roughness 0.05; "
         "openings for the magazine (-X) and outfeed (-Y, 0.5 x 0.33 m)", "task spec"),
        ("Magazine", "side magazine with carton_flat_stack",
         "-X side, references `../cartons/carton_flat_stack` (20 blanks on edge, rotated 90 deg "
         "about Y), rails + pusher paddle", "task spec; on-edge horizontal magazine ASSUMPTION"),
        ("Control panel", "HMI box + stack light",
         "+X cabinet with HMI housing/screen, E-stop; stack light green/amber/red (green lit, "
         "emissive) on the cabinet top, top 1.84 m", "task spec"),
        ("Outfeed bed", "top 0.82 m, aligned with conveyor",
         "Ø50 rollers at 75 mm pitch, 450 mm between guides (guide top 0.86 m) like the conveyor; "
         "Y -1.10 .. -0.05, centred X = 0, ends at the -Y edge", "task spec / conveyor asset"),
        ("SpawnPoint", "carton bottom at 0.82 m", "`/CaseErector/Outfeed/SpawnPoint` at "
         f"(0, {SPAWN_Y}, 0.82); a carton there spans Y -1.00 .. -0.40", "task spec"),
        ("Colliders", "outfeed bed + outer guards", "static guide-purpose boxes: Outfeed/Colliders/"
         "{Bed,GuideNegX,GuidePosX} (stainless 0.3), Guards/Colliders/Panel* (8, polycarbonate 0.4/0.35), "
         "Frame/Colliders/BaseCabinet (painted steel 0.4); no rigid bodies", "task spec; friction ASSUMPTION"),
        ("Mechanism", "-", "forming box, vacuum pick arm with 4 cups, bottom tape head (visual only)",
         "ASSUMPTION: generic layout"),
        ("Triangles", "< 60k", str(tris), "build.py (incl. referenced blanks)"),
    ]
    lines = ["# case_erector — generic automatic carton erector", "",
             f"Generated by `{GEN}`; do not hand-edit the .usda files. Unbranded, no logos.",
             "Placement: translate so the outfeed's -Y edge (asset Y = -1.10, X = 0) meets the "
             "conveyor's +Y end centreline.", ""]
    lines += K.readme_table(rows)
    lines += ["", "## Prim paths", "",
              "- `/CaseErector/Outfeed/SpawnPoint` (Xform): erected carton_open spawn pose",
              "- `/CaseErector/Outfeed/{Rollers,Frame,Colliders}`",
              "- `/CaseErector/Guards/{Panels,Hardware,Colliders}`",
              "- `/CaseErector/Frame/{Posts,BaseCabinet,CabinetDetails,Feet,Colliders}`",
              "- `/CaseErector/Magazine/{Rails,Pusher,BlankStack}` (BlankStack = reference)",
              "- `/CaseErector/ControlPanel/{Cabinet,HMI/{Housing,Screen},EStop,EStopPlate}`",
              "- `/CaseErector/StackLight/{Pole,Green,Amber,Red,Cap}`",
              "- `/CaseErector/Mechanism/*` (visual only)", ""]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


def main():
    st = build_geo(HERE / "geo.usda")
    K.write_physics(HERE, NAME, GEN, collider_map(st))
    K.write_root(HERE, "case_erector", NAME, GEN)
    full = Usd.Stage.Open(str(HERE / "case_erector.usda"), Usd.Stage.LoadAll)
    tris = K.count_triangles(full)
    write_readme(tris)
    print(f"case_erector: {tris} triangles")


if __name__ == "__main__":
    main()
