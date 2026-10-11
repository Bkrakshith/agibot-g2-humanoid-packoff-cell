"""Generator for the `floor_bin` asset: open-top red HDPE reject/overflow bin on 4 castors.

Run:  /home/rakshith/ntu_dev/Agibot_dev/.venv-tools/bin/python assets/floor_bin/build.py
Writes floor_bin.usda, geo.usda, physics.usda, README.md next to this script.

Asset frame (assets/CONVENTIONS.md): metres, Z up, origin = centre of the 0.6 (X) x 0.8 (Y)
footprint on the floor, long side along Y. Static prop (no rigid body): compound box colliders
(base + 4 tapered walls) so bags that fall off the end of the line land inside.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cartons"))

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

import usdkit as K  # noqa: E402

GEN = "assets/floor_bin/build.py"
NAME = "FloorBin"
R = Sdf.Path(f"/{NAME}")

RIM_X, RIM_Y, HEIGHT = 0.300, 0.400, 0.700     # half sizes of the rim + overall height
TUB_Z0, TUB_Z1 = 0.100, 0.670                  # tub bottom (on the castor plates) / top
BOT = (0.260, 0.360)                           # tub outer half size at the bottom
TOP = (0.290, 0.390)                           # tub outer half size at the top (draft)
T = 0.006                                      # wall thickness
FLOOR_T = 0.008
RIM_Z0 = 0.655
CASTOR_H = TUB_Z0
WHEEL_R = 0.040

RED = (0.55, 0.03, 0.03)
DARK = (0.06, 0.06, 0.07)
STEEL = (0.62, 0.62, 0.62)


def outer_half(z):
    f = (z - TUB_Z0) / (TUB_Z1 - TUB_Z0)
    return BOT[0] + f * (TOP[0] - BOT[0]), BOT[1] + f * (TOP[1] - BOT[1])


def tub_mesh():
    """Tapered open tub: outer skin, inner skin, floor, top lip (one closed shell)."""
    mb = K.MeshBuilder()
    rings = []
    for z, inset in ((TUB_Z0, 0.0), (TUB_Z1, 0.0), (TUB_Z1, T), (TUB_Z0 + FLOOR_T, T)):
        hx, hy = outer_half(z)
        hx, hy = hx - inset, hy - inset
        b = len(mb.points)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            mb.points.append(Gf.Vec3f(x, y, z))
        rings.append(b)
    o_bot, o_top, i_top, i_bot = rings
    for a, b in ((o_bot, o_top), (o_top, i_top), (i_top, i_bot)):   # side bands
        for k in range(4):
            k1 = (k + 1) % 4
            mb.counts.append(4)
            mb.indices.extend((a + k, a + k1, b + k1, b + k))
    mb.counts += [4, 4]
    mb.indices.extend((o_bot + 3, o_bot + 2, o_bot + 1, o_bot + 0))   # underside
    mb.indices.extend((i_bot + 0, i_bot + 1, i_bot + 2, i_bot + 3))   # inner floor
    # moulded stiffening ribs on the long sides
    for sx in (-1, 1):
        for y in (-0.2, 0.0, 0.2):
            z0, z1 = TUB_Z0 + 0.03, RIM_Z0
            hx0, _ = outer_half(z0)
            hx1, _ = outer_half(z1)
            mb.obox((sx * ((hx0 + hx1) / 2 + 0.004), y, (z0 + z1) / 2),
                    ((1, 0, 0), (0, 1, 0), _wall_up(sx, 0)), (0.004, 0.012, (z1 - z0) / 2 / _cos()))
    return mb


def _angle_x():
    return math.atan2(TOP[0] - BOT[0], TUB_Z1 - TUB_Z0)


def _cos():
    return math.cos(_angle_x())


def _wall_up(sx, sy):
    if sx:
        a = _angle_x()
        return (sx * math.sin(a), 0.0, math.cos(a))
    a = math.atan2(TOP[1] - BOT[1], TUB_Z1 - TUB_Z0)
    return (0.0, sy * math.sin(a), math.cos(a))


def rim_mesh():
    """Flared lip: a 45 mm band standing proud of the tub top out to the 600 x 800 envelope,
    with a chamfered underside flaring out from the wall."""
    mb = K.MeshBuilder()
    ix, iy = outer_half(TUB_Z1)
    ix, iy = ix - T, iy - T
    z0, z1 = RIM_Z0 + 0.015, HEIGHT
    mb.box((-RIM_X, -RIM_Y, z0), (-ix, RIM_Y, z1))
    mb.box((ix, -RIM_Y, z0), (RIM_X, RIM_Y, z1))
    mb.box((-ix, -RIM_Y, z0), (ix, -iy, z1))
    mb.box((-ix, iy, z0), (ix, RIM_Y, z1))
    # flare: thin chamfer plates from the wall at RIM_Z0 out to the lip at z0
    for sx in (-1, 1):
        hx, _ = outer_half(RIM_Z0)
        dx, dz = RIM_X - hx, z0 - RIM_Z0
        ln = math.hypot(dx, dz)
        mb.obox((sx * (hx + RIM_X) / 2, 0.0, (RIM_Z0 + z0) / 2),
                ((sx * dx / ln, 0, dz / ln), (0, 1, 0), (-sx * dz / ln, 0, dx / ln)),
                (ln / 2, RIM_Y - 0.01, 0.002))
    for sy in (-1, 1):
        _, hy = outer_half(RIM_Z0)
        dy, dz = RIM_Y - hy, z0 - RIM_Z0
        ln = math.hypot(dy, dz)
        mb.obox((0.0, sy * (hy + RIM_Y) / 2, (RIM_Z0 + z0) / 2),
                ((1, 0, 0), (0, sy * dy / ln, dz / ln), (0, -sy * dz / ln, dy / ln)),
                (RIM_X - 0.01, ln / 2, 0.002))
    return mb


def castor(x, y):
    """Swivel castor: top plate, swivel, fork, Ø80 wheel (axis X)."""
    steel, wheel = K.MeshBuilder(), K.MeshBuilder()
    steel.box((x - 0.04, y - 0.04, CASTOR_H - 0.006), (x + 0.04, y + 0.04, CASTOR_H))
    steel.cylinder((x, y, CASTOR_H - 0.006), (x, y, CASTOR_H - 0.018), 0.025, 16)
    yo = y + 0.012                                              # trailing offset
    for sx in (-1, 1):
        steel.box((x + sx * 0.020 - 0.002, yo - 0.012, WHEEL_R - 0.004),
                  (x + sx * 0.020 + 0.002, yo + 0.012, CASTOR_H - 0.018))
    steel.box((x - 0.022, yo - 0.012, CASTOR_H - 0.022), (x + 0.022, yo + 0.012, CASTOR_H - 0.018))
    wheel.cylinder((x - 0.016, yo, WHEEL_R), (x + 0.016, yo, WHEEL_R), WHEEL_R, 20)
    wheel.cylinder((x - 0.019, yo, WHEEL_R), (x + 0.019, yo, WHEEL_R), 0.008, 8)
    return steel, wheel


def build_geo(path: Path):
    st = K.new_stage(path, GEN)
    K.root_xform(st, NAME)
    lp = R.AppendChild("Looks")
    UsdGeom.Scope.Define(st, lp)
    hdpe = K.preview_material(st, lp.AppendChild("HDPERed"), RED, 0.55)
    dark = K.preview_material(st, lp.AppendChild("RubberDark"), DARK, 0.7)
    steel = K.preview_material(st, lp.AppendChild("ZincSteel"), STEEL, 0.35, 1.0)

    UsdGeom.Xform.Define(st, R.AppendChild("Geo"))
    tub_mesh().write(st, R.AppendPath("Geo/Body"), hdpe, RED)
    rim_mesh().write(st, R.AppendPath("Geo/Rim"), hdpe, RED)
    for sx in (-1, 1):
        for sy in (-1, 1):
            n = f"Castor_{'N' if sx < 0 else 'P'}X{'N' if sy < 0 else 'P'}Y"
            UsdGeom.Xform.Define(st, R.AppendPath(f"Geo/{n}"))
            s, w = castor(sx * 0.20, sy * 0.29)
            s.write(st, R.AppendPath(f"Geo/{n}/Fork"), steel, STEEL)
            w.write(st, R.AppendPath(f"Geo/{n}/Wheel"), dark, DARK)

    UsdGeom.Scope.Define(st, R.AppendChild("Colliders"))
    bx, by = BOT
    K.collider_box(st, R.AppendPath("Colliders/Base"), (-bx, -by, TUB_Z0 - 0.02),
                   (bx, by, TUB_Z0 + FLOOR_T))
    zc0, zc1 = TUB_Z0 + FLOOR_T, HEIGHT
    zm, hz = (zc0 + zc1) / 2, (zc1 - zc0)
    hxm, hym = outer_half(zm)
    ax = math.degrees(_angle_x())
    ay = math.degrees(math.atan2(TOP[1] - BOT[1], TUB_Z1 - TUB_Z0))
    lx, ly = hz / math.cos(math.radians(ax)), hz / math.cos(math.radians(ay))
    for n, c, size, rot in (
            ("WallNegX", (-(hxm - T / 2), 0, zm), (T, 2 * TOP[1], lx), (0, -ax, 0)),
            ("WallPosX", ((hxm - T / 2), 0, zm), (T, 2 * TOP[1], lx), (0, ax, 0)),
            ("WallNegY", (0, -(hym - T / 2), zm), (2 * TOP[0], T, ly), (ay, 0, 0)),
            ("WallPosY", (0, (hym - T / 2), zm), (2 * TOP[0], T, ly), (-ay, 0, 0))):
        lo = tuple(ci - si / 2 for ci, si in zip(c, size))
        hi = tuple(ci + si / 2 for ci, si in zip(c, size))
        K.collider_box(st, R.AppendPath(f"Colliders/{n}"), lo, hi, rotate=rot)

    ix, iy = BOT[0] - T, BOT[1] - T
    iv = K.collider_box(st, R.AppendChild("InteriorVolume"), (-ix, -iy, TUB_Z0 + FLOOR_T),
                        (ix, iy, HEIGHT))
    iv.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    iv.GetPrim().SetDocumentation("Inner volume of the bin (bottom inner footprint extruded to the "
                                  "rim top) for 'bag inside' checks. Invisible, guide, no collision.")
    st.GetRootLayer().Save()
    return st


def write_readme(tris):
    ix, iy = BOT[0] - T, BOT[1] - T
    rows = [
        ("Type", "open-top reject/overflow bin on castors", "static prop, no rigid body", "task spec"),
        ("Outer X x Y x H", "~0.6 x 0.8 x 0.7 m", "0.60 x 0.80 x 0.70 m (rim envelope, visual bbox)",
         "task spec"),
        ("Tub", "HDPE, slight flared rim", f"tapered tub {2 * BOT[0]:.2f} x {2 * BOT[1]:.2f} m at "
         f"the bottom to {2 * TOP[0]:.2f} x {2 * TOP[1]:.2f} m at Z {TUB_Z1} (~3 deg draft), 6 mm "
         "walls, 8 mm floor, side ribs; 45 mm lip with chamfered flare out to 0.60 x 0.80 m",
         "ASSUMPTION: typical moulded HDPE bin"),
        ("Colour", "red = reject (or grey)", f"red HDPE {RED}, roughness 0.55", "task spec"),
        ("Castors", "4", "Ø80 swivel castors, plates at (+-0.20, +-0.29), tub bottom at 0.10 m",
         "ASSUMPTION"),
        ("Colliders", "compound: base + walls", "Base box + WallNegX/PosX/NegY/PosY boxes tilted "
         "with the wall draft (6 mm), up to the rim top; HDPE 0.5", "task spec / CONVENTIONS"),
        ("Interior", "InteriorVolume", f"{2 * ix:.3f} x {2 * iy:.3f} x {HEIGHT - TUB_Z0 - FLOOR_T:.3f} m "
         "(inner bottom footprint to rim top), invisible guide, no collision", "task spec"),
        ("Triangles", "< 60k", str(tris), "build.py"),
    ]
    lines = ["# floor_bin — red HDPE reject bin on castors", "",
             f"Generated by `{GEN}`; do not hand-edit the .usda files. Unbranded, no logos.", ""]
    lines += K.readme_table(rows)
    lines += ["", "## Prim paths", "",
              "- `/FloorBin/Geo/{Body,Rim}`, `/FloorBin/Geo/Castor_{NXNY,NXPY,PXNY,PXPY}/{Fork,Wheel}`",
              "- `/FloorBin/Colliders/{Base,WallNegX,WallPosX,WallNegY,WallPosY}`",
              "- `/FloorBin/InteriorVolume`", ""]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


def main():
    st = build_geo(HERE / "geo.usda")
    cols = {f"/{NAME}/Colliders/{n}": "HDPE"
            for n in ("Base", "WallNegX", "WallPosX", "WallNegY", "WallPosY")}
    K.write_physics(HERE, NAME, GEN, cols)
    K.write_root(HERE, "floor_bin", NAME, GEN)
    tris = K.count_triangles(Usd.Stage.Open(str(HERE / "floor_bin.usda"), Usd.Stage.LoadAll))
    write_readme(tris)
    print(f"floor_bin: {tris} triangles")


if __name__ == "__main__":
    main()
