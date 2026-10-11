"""Generate the corrugated carton assets (pxr only).

Run:
    /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/cartons/build.py [--contents-mass KG]

Writes, for each variant (carton_open, carton_sealed, carton_flat, carton_flat_stack):
    assets/cartons/<name>/<name>.usda   root: defaultPrim (Xform, kind=component),
                                         references ./geo.usda, payloads ./physics.usda
    assets/cartons/<name>/geo.usda      visual meshes + UsdPreviewSurface looks, collider boxes
                                         (UsdGeom.Cube, purpose=guide), InteriorVolume (open only)
    assets/cartons/<name>/physics.usda  overs only (+ PhysicsMaterials)
    assets/cartons/<name>/README.md     spec vs modelled table

Frame: metres, Z up, origin at the centre of the footprint on the floor (Z = 0), long side
along Y. The 600 x 400 x 250 mm regular slotted carton (RSC) replaces the 600 x 400 crates.
The .usda/README files are outputs; edit this script, not them.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pxr import Sdf, UsdGeom  # noqa: E402

import usdkit as K  # noqa: E402

GEN = "assets/cartons/build.py"
MM = 0.001

W, L, H = 0.400, 0.600, 0.250     # outer X, Y, Z
T = 4 * MM                        # single-wall corrugate (C/B flute ~4 mm)
BASE_T = 2 * T                    # bottom = inner + outer flap layers
FLAP = W / 2                      # RSC flap depth = half the carton width
FLAP_FLARE_DEG = 12.0             # erected-for-packing: flaps folded out and down, flared 12 deg
FLAP_SLOT = 3 * MM                # half-slot at each flap end
TAPE_W = 48 * MM
TAPE_T = 0.5 * MM
TAPE_LEG = 60 * MM                # tape tab down each end face (typical 50-70 mm)
MASS = 0.55
BLANK = (0.650, 1.000, 8 * MM)    # knocked-down flat (X, Y, Z): two board layers
N_STACK = 20

KRAFT = (0.54, 0.33, 0.16)        # linear RGB of brown kraft liner
KRAFT_JITTER = 0.06               # +-6 % per-face brightness variation (uniform primvar)
TAPE = (0.36, 0.20, 0.07)         # buff polypropylene packing tape: darker, glossier
SCORE = (0.40, 0.24, 0.11)


def looks(stage, rp):
    UsdGeom.Scope.Define(stage, rp.AppendChild("Looks"))
    kraft = K.preview_material(stage, rp.AppendPath("Looks/Kraft"), KRAFT, roughness=0.85,
                               use_primvar=True)
    tape = K.preview_material(stage, rp.AppendPath("Looks/Tape"), TAPE, roughness=0.3)
    score = K.preview_material(stage, rp.AppendPath("Looks/ScoreLine"), SCORE, roughness=0.9)
    return kraft, tape, score


def shell(mb, top_closed):
    """Walls + base (+ closed top) of the RSC body."""
    xo, yo, xi, yi = W / 2, L / 2, W / 2 - T, L / 2 - T
    z0 = TAPE_T if top_closed else 0.0          # sealed: bottom tape sits under the base
    mb.box((-xo, -yo, z0), (-xi, yo, H))
    mb.box((xi, -yo, z0), (xo, yo, H))
    mb.box((-xi, -yo, z0), (xi, -yi, H))
    mb.box((-xi, yi, z0), (xi, yo, H))
    mb.box((-xi, -yi, z0), (xi, yi, BASE_T))
    if top_closed:
        # inner (short) flaps under the outer (long) flaps meeting at the X = 0 seam
        mb.box((-xi, -yi, H - 2 * T), (xi, -yi + FLAP * 0.9, H - T))
        mb.box((-xi, yi - FLAP * 0.9, H - 2 * T), (xi, yi, H - T))
        gap = 1 * MM
        mb.box((-xo, -yo, H - T), (-gap, yo, H))
        mb.box((gap, -yo, H - T), (xo, yo, H))


def flap_meshes():
    a = math.radians(FLAP_FLARE_DEG)
    out = {}
    for name, n, along, width in (("FlapPosX", (1, 0, 0), (0, 1, 0), L),
                                  ("FlapNegX", (-1, 0, 0), (0, 1, 0), L),
                                  ("FlapPosY", (0, 1, 0), (1, 0, 0), W),
                                  ("FlapNegY", (0, -1, 0), (1, 0, 0), W)):
        # hinge on the outer top edge; flap direction d = out*sin(a) - z*cos(a)
        hx, hy = n[0] * W / 2, n[1] * L / 2
        d = (n[0] * math.sin(a), n[1] * math.sin(a), -math.cos(a))
        nn = (n[0] * math.cos(a), n[1] * math.cos(a), math.sin(a))    # outward face normal
        c = (hx + d[0] * FLAP / 2 + nn[0] * T / 2,
             hy + d[1] * FLAP / 2 + nn[1] * T / 2,
             H + d[2] * FLAP / 2 + nn[2] * T / 2)
        # small offset so the flap hinge sits on (not above) the rim
        c = (c[0], c[1], c[2] - nn[2] * T / 2 - 0.0002)
        mb = K.MeshBuilder()
        mb.obox(c, (along, d, nn), (width / 2 - FLAP_SLOT, FLAP / 2, T / 2))
        out[name] = mb
    return out


def collider_boxes(variant):
    xo, yo, xi, yi = W / 2, L / 2, W / 2 - T, L / 2 - T
    if variant == "carton_open":
        return {
            "Base": ((-xo, -yo, 0.0), (xo, yo, BASE_T)),
            "WallNegX": ((-xo, -yo, BASE_T), (-xi, yo, H)),
            "WallPosX": ((xi, -yo, BASE_T), (xo, yo, H)),
            "WallNegY": ((-xi, -yo, BASE_T), (xi, -yi, H)),
            "WallPosY": ((-xi, yi, BASE_T), (xi, yo, H)),
        }
    if variant == "carton_sealed":
        return {"Box": ((-xo, -yo, 0.0), (xo, yo, H))}
    if variant == "carton_flat":
        return {"Box": ((-BLANK[0] / 2, -BLANK[1] / 2, 0.0), (BLANK[0] / 2, BLANK[1] / 2, BLANK[2]))}
    if variant == "carton_flat_stack":
        return {"Box": ((-BLANK[0] / 2, -BLANK[1] / 2, 0.0),
                        (BLANK[0] / 2, BLANK[1] / 2, BLANK[2] * N_STACK))}
    raise KeyError(variant)


def blank_mesh(z0=0.0, dx=0.0, dy=0.0):
    """Knocked-down RSC blank lying flat: X = flap | body | flap (0.2/0.25/0.2), Y = the
    600 and 400 panels side by side (the hidden half sits underneath), slots between flaps."""
    bx, by, bt = BLANK
    fx = FLAP                                # 0.2 flap rows
    yb = -by / 2 + 0.600                     # panel boundary along Y
    s = 3 * MM
    mb = K.MeshBuilder()
    mb.box((-bx / 2 + fx + dx, -by / 2 + dy, z0), (bx / 2 - fx + dx, by / 2 + dy, z0 + bt))
    for x0, x1 in ((-bx / 2, -bx / 2 + fx), (bx / 2 - fx, bx / 2)):
        mb.box((x0 + dx, -by / 2 + dy, z0), (x1 + dx, yb - s + dy, z0 + bt))
        mb.box((x0 + dx, yb + s + dy, z0), (x1 + dx, by / 2 + dy, z0 + bt))
    return mb


def score_mesh(z):
    bx, by, _ = BLANK
    yb = -by / 2 + 0.600
    w = 1.5 * MM
    mb = K.MeshBuilder()
    for x in (-bx / 2 + FLAP, bx / 2 - FLAP):            # flap fold lines along Y
        mb.box((x - w, -by / 2, z), (x + w, by / 2, z + 0.2 * MM))
    mb.box((-bx / 2 + FLAP, yb - w, z), (bx / 2 - FLAP, yb + w, z + 0.2 * MM))   # corner fold
    return mb


# ---------------------------------------------------------------- per variant
VARIANTS = {
    "carton_open": dict(prim="CartonOpen", title="Erected open-top RSC carton 600x400x250",
                        rigid=True),
    "carton_sealed": dict(prim="CartonSealed", title="Sealed RSC carton 600x400x250, taped",
                          rigid=True),
    "carton_flat": dict(prim="CartonFlat", title="Knocked-down flat carton blank", rigid=False),
    "carton_flat_stack": dict(prim="CartonFlatStack", title="Stack of 20 flat carton blanks",
                              rigid=False),
}


def write_geo(folder, name, v, contents_mass):
    stage = K.new_stage(folder / "geo.usda", GEN)
    root = K.root_xform(stage, v["prim"])
    rp = root.GetPath()
    kraft, tape, score = looks(stage, rp)
    geo = UsdGeom.Xform.Define(stage, rp.AppendChild("Geo"))
    gp = geo.GetPath()

    if name in ("carton_open", "carton_sealed"):
        mb = K.MeshBuilder()
        shell(mb, top_closed=(name == "carton_sealed"))
        mb.write(stage, gp.AppendChild("Body"), kraft, KRAFT, KRAFT_JITTER, seed=1)
    if name == "carton_open":
        UsdGeom.Xform.Define(stage, gp.AppendChild("Flaps"))
        for i, (fn, fm) in enumerate(flap_meshes().items()):
            fm.write(stage, gp.AppendPath(f"Flaps/{fn}"), kraft, KRAFT, KRAFT_JITTER, seed=10 + i)
        iv = K.collider_box(stage, rp.AppendChild("InteriorVolume"),
                            (-W / 2 + T, -L / 2 + T, BASE_T), (W / 2 - T, L / 2 - T, H))
        iv.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        iv.GetPrim().SetDocumentation("Inner volume of the carton for 'bag fully inside' tests. "
                                      "Invisible, guide purpose, no collision.")
    if name == "carton_sealed":
        mb = K.MeshBuilder()
        hw = TAPE_W / 2
        mb.box((-hw, -L / 2, H), (hw, L / 2, H + TAPE_T))                         # top seam
        mb.box((-hw, -L / 2 - TAPE_T, H - TAPE_LEG), (hw, -L / 2, H + TAPE_T))    # end tabs
        mb.box((-hw, L / 2, H - TAPE_LEG), (hw, L / 2 + TAPE_T, H + TAPE_T))
        mb.box((-hw, -L / 2, 0.0), (hw, L / 2, TAPE_T))                           # bottom seam
        mb.box((-hw, -L / 2 - TAPE_T, 0.0), (hw, -L / 2, TAPE_LEG))
        mb.box((-hw, L / 2, 0.0), (hw, L / 2 + TAPE_T, TAPE_LEG))
        mb.write(stage, gp.AppendChild("Tape"), tape, TAPE)
        a = root.GetPrim().CreateAttribute("packoff:contentsMass", Sdf.ValueTypeNames.Float)
        a.Set(contents_mass)
        a.SetDocumentation("Mass of the packed bags (kg); MassAPI mass = 0.55 + this.")
    if name == "carton_flat":
        blank_mesh().write(stage, gp.AppendChild("Body"), kraft, KRAFT, KRAFT_JITTER, seed=3)
        score_mesh(BLANK[2]).write(stage, gp.AppendChild("ScoreLines"), score, SCORE)
    if name == "carton_flat_stack":
        body = UsdGeom.Xform.Define(stage, gp.AppendChild("Body"))
        K.bind(body, kraft)
        for i in range(N_STACK):
            # deterministic +-0.8 mm misalignment so the stack does not look extruded
            dx = 0.0008 * math.sin(i * 2.3)
            dy = 0.0008 * math.cos(i * 1.7)
            blank_mesh(i * BLANK[2], dx, dy).write(
                stage, body.GetPath().AppendChild(f"Blank_{i:02d}"), kraft, KRAFT,
                KRAFT_JITTER, seed=100 + i)

    cols = UsdGeom.Scope.Define(stage, rp.AppendChild("Colliders"))
    for cn, (lo, hi) in collider_boxes(name).items():
        K.collider_box(stage, cols.GetPath().AppendChild(cn), lo, hi)
    stage.GetRootLayer().Save()
    return K.count_triangles(stage)


def write_readme(folder, name, v, tris, contents_mass):
    p = v["prim"]
    a = FLAP_FLARE_DEG
    reach = FLAP * math.sin(math.radians(a)) + T * math.cos(math.radians(a))
    common = [("Frame", "metres, Z up", "origin = footprint centre on the floor, long side +Y",
               "CONVENTIONS.md")]
    if name in ("carton_open", "carton_sealed"):
        rows = [
            ("Outer X x Y x Z", "400 x 600 x 250 mm", "400 x 600 x 250 mm (`Geo/Body` bbox)",
             "task spec (replaces 600x400 crate)"),
            ("Board", "single-wall corrugate ~4 mm", "4 mm walls, 8 mm base (2 flap layers)",
             "ASSUMPTION: B/C flute"),
            ("Mass", "0.55 kg" + (" + contents" if name == "carton_sealed" else ""),
             f"{MASS} kg" + (f" + `packoff:contentsMass` ({contents_mass} kg, build "
                             "`--contents-mass`)" if name == "carton_sealed" else "")
             + " on MassAPI of root", "task spec"),
            ("Body", "movable", f"RigidBodyAPI + MassAPI on `/{p}` (dynamic)", "task spec"),
            ("Look", "kraft brown, roughness ~0.85",
             f"UsdPreviewSurface roughness 0.85 reading a uniform `displayColor` primvar "
             f"(+-{int(KRAFT_JITTER * 100)} % per-face variation around {KRAFT})", "task spec"),
            ("Friction", "-", "Cardboard 0.5 static / 0.45 dynamic", "ASSUMPTION"),
        ]
        if name == "carton_open":
            rows += [
                ("Top", "open, 4 flaps folded out and down",
                 f"200 mm flaps hinged at the outer rim, hanging down flared {a:.0f} deg "
                 f"outward (overall bbox {W * 1000 + 2 * reach * 1000:.0f} x "
                 f"{L * 1000 + 2 * reach * 1000:.0f} x 250 mm); visual only",
                 "ASSUMPTION: typical hand-pack pose"),
                ("Collider", "compound: base + 4 walls",
                 "5 boxes: Base (8 mm), WallNegX/PosX/NegY/PosY (4 mm)", "task spec"),
                ("Interior", "InteriorVolume like crates", "392 x 592 x 242 mm, invisible guide, "
                 "no collision", "task spec"),
            ]
        else:
            rows += [
                ("Top", "closed, 48 mm tape on the long seam",
                 "outer flaps meet on X = 0; 48 mm tape along Y (top + bottom) with 60 mm "
                 "tabs down both end faces, 0.5 mm proud", "task spec; tab length ASSUMPTION"),
                ("Tape look", "darker, glossier", f"{TAPE}, roughness 0.3", "task spec"),
                ("Collider", "single box", "1 box 400 x 600 x 250 mm", "task spec"),
            ]
    else:
        n = 1 if name == "carton_flat" else N_STACK
        rows = [
            ("Blank X x Y x Z", "~1.0 x 0.65 x 0.008 m",
             f"650 x 1000 x {8 * n} mm" + (f" ({n} blanks x 8 mm)" if n > 1 else ""),
             "task spec; KD RSC = (600+400) x (250+2x200), 2 board layers"),
            ("Detail", "-", "panel slots between the 600/400 flap pairs"
             + (", score lines (`Geo/ScoreLines`)" if n == 1 else
                ", +-0.8 mm deterministic misalignment per blank (`Geo/Body/Blank_NN`)"),
             "ASSUMPTION"),
            ("Body", "visual + collider", "no RigidBodyAPI (static magazine prop)", "task spec"),
            ("Collider", "box", "1 box = blank / stack envelope", "task spec"),
            ("Look", "kraft", "as the cartons: roughness 0.85, displayColor variation", "task spec"),
        ]
    rows += common + [("Contact offset", "-", "3 mm, rest 0 (raw `physxCollision:*`)", "crates"),
                      ("Triangles", "< 60k", str(tris), "build.py")]
    lines = [f"# {v['title']}  (`{name}`)", "",
             f"Generated by `{GEN}`; do not edit the .usda files by hand. No brands or logos.", ""]
    lines += K.readme_table(rows)
    lines += ["", "## Prim paths", "", f"- `/{p}`: root"
              + (" (rigid body)" if v["rigid"] else " (static, no rigid body)"),
              f"- `/{p}/Geo/Body`: carton body" + (" (Xform of 20 blank meshes)"
                                                   if name == "carton_flat_stack" else "")]
    if name == "carton_open":
        lines += [f"- `/{p}/Geo/Flaps/Flap{{NegX,PosX,NegY,PosY}}`: top flaps (visual only)",
                  f"- `/{p}/InteriorVolume`: invisible guide box = inner volume, no collision"]
    if name == "carton_sealed":
        lines += [f"- `/{p}/Geo/Tape`: tape strips (visual only)",
                  f"- `/{p}.packoff:contentsMass`: contents mass attribute (kg)"]
    lines += [f"- `/{p}/Colliders/*`: collider boxes (UsdGeom.Cube, purpose=guide)",
              f"- `/{p}/PhysicsMaterials/Cardboard`: physics material (physics.usda)", ""]
    (folder / "README.md").write_text("\n".join(lines) + "\n")


def build(name, v, contents_mass=0.0):
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    tris = write_geo(folder, name, v, contents_mass)
    cols = {f"/{v['prim']}/Colliders/{c}": "Cardboard" for c in collider_boxes(name)}
    mass = MASS + (contents_mass if name == "carton_sealed" else 0.0) if v["rigid"] else None
    K.write_physics(folder, v["prim"], GEN, cols, rigid_mass=mass)
    K.write_root(folder, name, v["prim"], GEN)
    write_readme(folder, name, v, tris, contents_mass)
    return tris


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents-mass", type=float, default=0.0,
                    help="mass of packed bags in carton_sealed (kg), default 0")
    args = ap.parse_args(argv)
    for name, v in VARIANTS.items():
        print(f"{name}: {build(name, v, args.contents_mass)} triangles")


if __name__ == "__main__":
    main()
