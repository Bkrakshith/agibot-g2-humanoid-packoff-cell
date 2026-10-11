"""Generator for the `case_sealer` asset: generic top + bottom tape case sealer (static prop).

Run:  /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/case_sealer/build.py
Writes case_sealer.usda, geo.usda, physics.usda, README.md next to this script.

Asset frame (assets/CONVENTIONS.md): metres, Z up, origin = centre of the 1.0 (X) x 1.6 (Y)
footprint on the floor. Flow toward -Y: infeed at +Y, outfeed at -Y, beds at 0.82 m. The
tunnel (side drive belts + top head) is set for a 400 wide x 250 high carton. Not animated.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cartons"))

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

import usdkit as K  # noqa: E402

GEN = "assets/case_sealer/build.py"
NAME = "CaseSealer"
R = Sdf.Path(f"/{NAME}")

HALF_X, HALF_Y, HEIGHT = 0.50, 0.80, 1.70
BED = 0.82
BED_HALF = 0.25               # roller length 500 mm
SEC = 0.45                    # infeed: Y > +SEC, outfeed: Y < -SEC, tunnel between
CARTON_W, CARTON_H = 0.400, 0.250
BELT_IN = 0.210               # side belt faces at X = +-0.21 -> 420 mm opening
BELT_OUT = 0.330
BELT_Z0, BELT_Z1 = 0.835, 1.000
HEAD_Z0 = BED + CARTON_H + 0.005          # 1.075: top head skid plate bottom
COL_X, COL = 0.42, 0.060
COL_TOP = 1.535                            # + 0.165 stack light = 1.70
BEAM_Z0, BEAM_Z1 = 1.30, 1.36
FRAME_Z0, FRAME_Z1 = 0.70, 0.80
LEG_XY = (0.30, 0.70)

BLUE_GREY = (0.22, 0.30, 0.40)
STEEL = (0.62, 0.62, 0.62)
BLACK = (0.03, 0.03, 0.03)
TAPE = (0.36, 0.20, 0.07)


def build_geo(path: Path):
    st = K.new_stage(path, GEN)
    K.root_xform(st, NAME)
    lp = R.AppendChild("Looks")
    UsdGeom.Scope.Define(st, lp)
    paint = K.preview_material(st, lp.AppendChild("PaintedSteelBlueGrey"), BLUE_GREY, 0.45)
    steel = K.preview_material(st, lp.AppendChild("Stainless"), STEEL, 0.35, 1.0)
    black = K.preview_material(st, lp.AppendChild("BlackRubber"), BLACK, 0.7)
    tape = K.preview_material(st, lp.AppendChild("Tape"), TAPE, 0.3)
    core = K.preview_material(st, lp.AppendChild("CardboardCore"), (0.50, 0.33, 0.17), 0.85)

    # ---------------- base frame
    UsdGeom.Xform.Define(st, R.AppendChild("Frame"))
    fr = K.MeshBuilder()
    lx, ly = LEG_XY
    for sx in (-1, 1):
        fr.box((sx * 0.255 if sx > 0 else -0.295, -HALF_Y, FRAME_Z0),
               (0.295 if sx > 0 else -0.255, HALF_Y, FRAME_Z1))                   # side rails
        for sy in (-1, 1):
            fr.box((sx * lx - 0.025, sy * ly - 0.025, 0.04), (sx * lx + 0.025, sy * ly + 0.025, FRAME_Z0))
    for y in (-ly, 0.0, ly):
        fr.box((-lx, y - 0.02, 0.20), (lx, y + 0.02, 0.24))                         # low cross bars
        fr.box((-0.255, y - 0.02, FRAME_Z0 - 0.04), (0.255, y + 0.02, FRAME_Z0))    # bed cross bars
    fr.box((-lx, -ly, 0.20), (-lx + 0.04, ly, 0.24))
    fr.box((lx - 0.04, -ly, 0.20), (lx, ly, 0.24))
    for sx in (-1, 1):                                                                # column feet
        fr.box((sx * COL_X - 0.04, -0.06, FRAME_Z0 - 0.02), (sx * COL_X + 0.04, 0.06, FRAME_Z0 + 0.02))
        fr.box((sx * 0.295 if sx > 0 else sx * COL_X, -0.03, FRAME_Z0),
               (sx * COL_X if sx > 0 else -0.295, 0.03, FRAME_Z0 + 0.02))
    fr.write(st, R.AppendPath("Frame/Base"), paint, BLUE_GREY)
    ft = K.MeshBuilder()
    for sx in (-1, 1):
        for sy in (-1, 1):
            ft.cylinder((sx * lx, sy * ly, 0.0), (sx * lx, sy * ly, 0.012), 0.035, 12)
            ft.cylinder((sx * lx, sy * ly, 0.012), (sx * lx, sy * ly, 0.04), 0.010, 8)
    ft.write(st, R.AppendPath("Frame/Feet"), black, BLACK)

    # ---------------- infeed / outfeed beds
    for grp, y0, y1 in (("Infeed", SEC, HALF_Y), ("Outfeed", -HALF_Y, -SEC)):
        UsdGeom.Xform.Define(st, R.AppendChild(grp))
        K.rollers(y0, y1, BED, BED_HALF).write(st, R.AppendPath(f"{grp}/Rollers"), steel, STEEL)
        UsdGeom.Scope.Define(st, R.AppendPath(f"{grp}/Colliders"))
        K.collider_box(st, R.AppendPath(f"{grp}/Colliders/Bed"),
                       (-BED_HALF, y0, BED - 0.01), (BED_HALF, y1, BED))
    K.guide_point(st, R.AppendPath("Infeed/EntryPoint"), (0.0, HALF_Y, BED),
                  "Carton entry: +Y edge of the infeed bed, on the bed top (0.82 m).")
    K.guide_point(st, R.AppendPath("Outfeed/ExitPoint"), (0.0, -HALF_Y, BED),
                  "Carton exit: -Y edge of the outfeed bed, on the bed top (0.82 m); a sealed "
                  "carton leaves toward -Y.")

    # ---------------- tunnel: centre bed, side drive belts, columns, top head
    UsdGeom.Xform.Define(st, R.AppendChild("Tunnel"))
    cr = K.MeshBuilder()                      # centre rollers split around the bottom head slot
    for x0, x1 in ((-BED_HALF, -0.06), (0.06, BED_HALF)):
        part = K.rollers(-SEC + 0.01, SEC - 0.01, BED, (x1 - x0) / 2, x0=(x0 + x1) / 2)
        base = len(cr.points)
        cr.points += part.points
        cr.counts += part.counts
        cr.indices += [i + base for i in part.indices]
    cr.write(st, R.AppendPath("Tunnel/Rollers"), steel, STEEL)

    bh = K.MeshBuilder()                      # side belt housings (+ arms to the columns)
    belt = K.MeshBuilder()
    for s in (-1, 1):
        x_in, x_out = s * (BELT_IN + 0.005), s * BELT_OUT
        bh.box((min(x_in, x_out), -SEC, BELT_Z0), (max(x_in, x_out), SEC, BELT_Z1))
        bh.box((min(s * BELT_OUT, s * (COL_X - COL / 2)), -0.03, 0.93),
               (max(s * BELT_OUT, s * (COL_X - COL / 2)), 0.03, 0.97))
        for y in (-SEC + 0.04, SEC - 0.04):                                  # belt pulleys / motors
            bh.cylinder((s * 0.27, y, BELT_Z1), (s * 0.27, y, BELT_Z1 + 0.10), 0.045, 16)
        xb0, xb1 = sorted((s * BELT_IN, s * (BELT_IN + 0.005)))
        belt.box((xb0, -SEC + 0.02, BELT_Z0 + 0.02), (xb1, SEC - 0.02, BELT_Z1 - 0.02))
    bh.write(st, R.AppendPath("Tunnel/BeltHousings"), paint, BLUE_GREY)
    belt.write(st, R.AppendPath("Tunnel/Belts"), black, BLACK)

    co = K.MeshBuilder()                      # columns, crossbeam, height screw, handwheels
    for s in (-1, 1):
        co.box((s * COL_X - COL / 2, -COL / 2, FRAME_Z0), (s * COL_X + COL / 2, COL / 2, COL_TOP))
    co.box((-COL_X - COL / 2, -0.04, BEAM_Z0), (COL_X + COL / 2, 0.04, BEAM_Z1))
    co.write(st, R.AppendPath("Tunnel/Columns"), paint, BLUE_GREY)
    hw = K.MeshBuilder()
    for s in (-1, 1):
        x0 = s * (COL_X + COL / 2)
        hw.cylinder((x0, 0.0, 0.95), (s * (HALF_X - 0.02), 0.0, 0.95), 0.012, 10)
        hw.cylinder((s * (HALF_X - 0.02), 0.0, 0.95), (s * HALF_X, 0.0, 0.95), 0.08, 20)
    hw.cylinder((COL_X, 0.0, COL_TOP), (COL_X, 0.0, COL_TOP + 0.015), 0.07, 20)     # height crank
    hw.cylinder((COL_X + 0.05, 0.0, COL_TOP + 0.015), (COL_X + 0.05, 0.0, COL_TOP + 0.08), 0.01, 8)
    hw.write(st, R.AppendPath("Tunnel/Handwheels"), steel, STEEL)

    UsdGeom.Scope.Define(st, R.AppendPath("Tunnel/Colliders"))
    K.collider_box(st, R.AppendPath("Tunnel/Colliders/BedCentre"),
                   (-BED_HALF, -SEC, BED - 0.01), (BED_HALF, SEC, BED))
    for s, n in ((-1, "NegX"), (1, "PosX")):
        lo, hi = sorted((s * BELT_IN, s * BELT_OUT))
        K.collider_box(st, R.AppendPath(f"Tunnel/Colliders/Belt{n}"),
                       (lo, -SEC, BELT_Z0), (hi, SEC, BELT_Z1))
        K.collider_box(st, R.AppendPath(f"Tunnel/Colliders/Column{n}"),
                       (s * COL_X - COL / 2, -COL / 2, FRAME_Z0), (s * COL_X + COL / 2, COL / 2, COL_TOP))
    K.collider_box(st, R.AppendPath("Tunnel/Colliders/HeadPlate"),
                   (-0.15, -0.38, HEAD_Z0), (0.15, 0.38, HEAD_Z0 + 0.01))
    K.collider_box(st, R.AppendPath("Tunnel/Colliders/Crossbeam"),
                   (-COL_X - COL / 2, -0.04, BEAM_Z0), (COL_X + COL / 2, 0.04, BEAM_Z1))

    # ---------------- top head (adjustable, hangs from the crossbeam) with tape roll
    UsdGeom.Xform.Define(st, R.AppendChild("TopHead"))
    th = K.MeshBuilder()
    th.box((-0.15, -0.38, HEAD_Z0), (0.15, 0.38, HEAD_Z0 + 0.01))           # skid plate
    th.box((-0.08, -0.33, HEAD_Z0 + 0.01), (0.08, 0.33, 1.25))              # head body
    for x in (-0.05, 0.05):
        th.box((x - 0.015, -0.015, 1.25), (x + 0.015, 0.015, BEAM_Z0))     # hangers
    th.cylinder((-0.05, 0.20, 1.40), (0.05, 0.20, 1.40), 0.012, 10)        # roll spindle
    th.box((0.045, 0.18, 1.25), (0.055, 0.22, 1.40))                        # spindle bracket
    th.write(st, R.AppendPath("TopHead/Body"), steel, STEEL)
    wr = K.MeshBuilder()
    for y in (-0.36, 0.36):                                                 # wipe / applicator rollers
        wr.cylinder((-0.04, y, HEAD_Z0 + 0.035), (0.04, y, HEAD_Z0 + 0.035), 0.03, 16)
    wr.write(st, R.AppendPath("TopHead/WipeRollers"), black, BLACK)
    roll = K.MeshBuilder()
    roll.cylinder((-0.024, 0.20, 1.40), (0.024, 0.20, 1.40), 0.13, 32)
    roll.write(st, R.AppendPath("TopHead/TapeRoll"), tape, TAPE)
    cm = K.MeshBuilder()
    cm.cylinder((-0.026, 0.20, 1.40), (0.026, 0.20, 1.40), 0.038, 16)
    cm.write(st, R.AppendPath("TopHead/TapeCore"), core, (0.50, 0.33, 0.17))
    web = K.MeshBuilder()                                                   # tape web to the head
    web.box((-0.024, 0.30, HEAD_Z0 + 0.06), (0.024, 0.302, 1.33))
    web.write(st, R.AppendPath("TopHead/TapeWeb"), tape, TAPE)

    # ---------------- bottom head (under the bed, in the centre slot)
    UsdGeom.Xform.Define(st, R.AppendChild("BottomHead"))
    bb = K.MeshBuilder()
    bb.box((-0.05, -0.33, 0.50), (0.05, 0.33, BED - 0.005))
    bb.write(st, R.AppendPath("BottomHead/Body"), steel, STEEL)
    br = K.MeshBuilder()
    br.cylinder((-0.024, 0.15, 0.40), (0.024, 0.15, 0.40), 0.13, 32)
    br.write(st, R.AppendPath("BottomHead/TapeRoll"), tape, TAPE)

    # ---------------- small stack light on the -X column
    K.stack_light(st, R.AppendChild("StackLight"), (-COL_X, 0.0, COL_TOP), lp,
                  pole_h=0.03, seg_h=0.04, r=0.025)
    st.GetRootLayer().Save()
    return st


def collider_map(st):
    return {str(p.GetPath()): ("PaintedSteel" if "Column" in p.GetName() or "Crossbeam" in p.GetName()
                               else "Rubber" if "Belt" in p.GetName() else "Stainless")
            for p in st.Traverse() if p.IsA(UsdGeom.Cube) and "/Colliders/" in str(p.GetPath())}


def write_readme(tris):
    rows = [
        ("Type", "top + bottom tape case sealer, generic", "static prop, not animated", "task spec"),
        ("Footprint X x Y x H", "~1.0 x 1.6 x 1.7 m", "1.00 x 1.60 x 1.70 m (visual bbox)", "task spec"),
        ("Beds", "infeed/outfeed roller beds at 0.82 m",
         "Ø50 stainless rollers, 500 mm long, 75 mm pitch: infeed Y 0.45..0.80, centre (split "
         "around the bottom head), outfeed Y -0.80..-0.45; top 0.82 m", "task spec"),
        ("Tunnel", "400 W x 250 H cartons", "side belts at X = +-0.21 (420 mm clear), top head "
         f"skid plate at {HEAD_Z0:.3f} m (255 mm above the bed)", "task spec; clearances ASSUMPTION"),
        ("Side drives", "belt housings", "120 mm housings Z 0.835..1.00 with black belts, drive "
         "pulleys/motors on top, width handwheels outside the columns", "task spec"),
        ("Top head", "adjustable, two columns, tape roll visible",
         "60 mm columns at X = +-0.42 to 1.535 m, crossbeam at 1.30..1.36 m, stainless head with "
         "wipe rollers, Ø260 x 48 mm tape roll on a Ø76 core, crank on +X column",
         "task spec; dims typical"),
        ("Bottom head", "-", "stainless head + Ø260 tape roll under the centre bed", "ASSUMPTION"),
        ("Finish", "painted frame + stainless", f"blue-grey {BLUE_GREY} roughness 0.45; stainless "
         "metallic 1 roughness 0.35", "task spec"),
        ("Stack light", "small", "3 x 40 mm segments (green lit) on the -X column, top 1.70 m",
         "task spec"),
        ("Colliders", "beds + tunnel frame", "static boxes: Infeed/Outfeed/Colliders/Bed, "
         "Tunnel/Colliders/{BedCentre,BeltNegX,BeltPosX,ColumnNegX,ColumnPosX,HeadPlate,Crossbeam}; "
         "stainless 0.3, belts rubber 1.0, columns painted 0.4; no rigid bodies",
         "task spec; CONVENTIONS frictions"),
        ("Triangles", "< 60k", str(tris), "build.py"),
    ]
    lines = ["# case_sealer — generic top + bottom tape case sealer", "",
             f"Generated by `{GEN}`; do not hand-edit the .usda files. Unbranded, no logos.",
             "Flow toward -Y. Placement: put `Infeed/EntryPoint` (asset (0, +0.80, 0.82)) at the "
             "conveyor's -Y end.", ""]
    lines += K.readme_table(rows)
    lines += ["", "## Prim paths", "",
              "- `/CaseSealer/Infeed` (Xform group: Rollers, Colliders/Bed, EntryPoint at (0, 0.80, 0.82))",
              "- `/CaseSealer/Outfeed/ExitPoint` (Xform at (0, -0.80, 0.82)); `/CaseSealer/Outfeed/{Rollers,Colliders/Bed}`",
              "- `/CaseSealer/Tunnel/{Rollers,BeltHousings,Belts,Columns,Handwheels,Colliders}`",
              "- `/CaseSealer/TopHead/{Body,WipeRollers,TapeRoll,TapeCore,TapeWeb}`",
              "- `/CaseSealer/BottomHead/{Body,TapeRoll}`",
              "- `/CaseSealer/StackLight/{Pole,Green,Amber,Red,Cap}`", ""]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


def main():
    st = build_geo(HERE / "geo.usda")
    K.write_physics(HERE, NAME, GEN, collider_map(st))
    K.write_root(HERE, "case_sealer", NAME, GEN)
    tris = K.count_triangles(Usd.Stage.Open(str(HERE / "case_sealer.usda"), Usd.Stage.LoadAll))
    write_readme(tris)
    print(f"case_sealer: {tris} triangles")


if __name__ == "__main__":
    main()
