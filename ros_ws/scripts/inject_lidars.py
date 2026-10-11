#!/usr/bin/env python3
"""Author OmniLidar prims into the STAGED robot USD of a Genie Sim scene (idempotent).

Why: Genie Sim's lidar path (assemble_scene.py "Lidar render products") only references OmniLidar
prims that already exist in the robot USD (`<robot root>/<lidars[i].prim_path>`). The G2 is built from
URDF (no sensors) and the official G2 USD has no lidar prims either, so nothing gets rendered. This
script adds `<root>/<body>/<leaf>` OmniLidar prims (a plain, non-physics Xform `<body>` next to
`Geometry`) to `<assets>/scenes/<scene stem>/robot.usda`. That file is cache-gated by urdf.sha256, so
the edit survives relaunches until the URDF changes (then just rerun this script).

Profile: a generic 16-channel rotary lidar, vertical -7..52 deg (Livox Mid-360-like FOV, incl. 0 deg),
0.25 deg azimuth step, 0.1-40 m. Mount pose from the YAML `lidars[i].mount` (xyz + yaw_deg), in the
body (base_link) frame, sensor frame = ROS convention (x forward, z up).

Usage (in the container, system python3 with pxr):
  python3 inject_lidars.py --scene /g2cell_ws/config/scenes/scene_g2cell_mobile.yaml \
      [--assets /geniesim_assets] [--seed-from scene_g2cell]
--seed-from copies an already-converted robot (robot.usda, payloads/, urdf.sha256, ...) from another
staged scene of the SAME robot so the first launch already has the lidars (the converted robot USD
is scene-agnostic: fix_base is decided at runtime).
"""
import argparse
import math
import os
import shutil
import sys

import yaml
from pxr import Gf, Sdf, Usd, UsdGeom

ELEV_DEG = [-7, -4, -2, 0, 2, 4, 7, 10, 14, 18, 23, 28, 34, 40, 46, 52]
SEED_ITEMS = ["robot.usda", "robot.urdf", "robot_raw.urdf", "urdf.sha256", "payloads", "pkg",
              "Textures", "genie_sim_robot_model"]


def seed(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for item in SEED_ITEMS:
        s, d = os.path.join(src_dir, item), os.path.join(dst_dir, item)
        if not os.path.lexists(s) or os.path.lexists(d):
            continue
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)
        elif os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True)
        else:
            shutil.copy2(s, d)
    print(f"[inject_lidars] seeded {dst_dir} from {src_dir}")


def author_lidar(stage, path, xyz, yaw_deg):
    prim = stage.DefinePrim(path, "OmniLidar")
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["OmniSensorGenericLidarCoreAPI"]))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    h = math.radians(yaw_deg) / 2.0
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(h), 0.0, 0.0, math.sin(h)))
    n = len(ELEV_DEG)
    A = Sdf.ValueTypeNames
    attrs = {
        "omni:sensor:Core:numberOfEmitters": (A.UInt, n),
        "omni:sensor:Core:numberOfChannels": (A.UInt, n),
        "omni:sensor:Core:scanType": (A.Token, "ROTARY"),
        "omni:sensor:Core:rotPatternFiringResDeg": (A.Float, 0.25),
        "omni:sensor:Core:nearRangeM": (A.Float, 0.1),
        "omni:sensor:Core:farRangeM": (A.Float, 40.0),
        "omni:sensor:Core:maxReturns": (A.UInt, 1),
        "omni:sensor:Core:rangeAccuracyM": (A.Float, 0.01),
        "omni:sensor:Core:azimuthErrorStd": (A.Float, 0.0),
        "omni:sensor:marketName": (A.String, "G2 chassis lidar (generic 16ch, Mid-360-like FOV)"),
        "omni:sensor:Core:emitterState:s001:azimuthDeg": (A.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:elevationDeg": (A.FloatArray, [float(e) for e in ELEV_DEG]),
        "omni:sensor:Core:emitterState:s001:channelId": (A.UIntArray, list(range(1, n + 1))),
        "omni:sensor:Core:emitterState:s001:fireTimeNs": (A.UIntArray, [i * 1500 for i in range(n)]),
    }
    for name, (typ, val) in attrs.items():
        prim.CreateAttribute(name, typ).Set(val)
    return prim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--assets", default="/geniesim_assets")
    ap.add_argument("--seed-from", default="", help="staged scene dir name to copy the converted robot from")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.scene))
    lidars = cfg.get("lidars") or []
    stage_dir = os.path.join(a.assets, "scenes", os.path.splitext(os.path.basename(a.scene))[0])
    robot_usd = os.path.join(stage_dir, "robot.usda")
    if not os.path.isfile(robot_usd) and a.seed_from:
        seed(os.path.join(a.assets, "scenes", a.seed_from), stage_dir)
    if not os.path.isfile(robot_usd):
        sys.exit(f"[inject_lidars] {robot_usd} missing: launch the scene once (assemble_robot) or pass --seed-from")
    stage = Usd.Stage.Open(robot_usd, load=Usd.Stage.LoadNone)
    root = stage.GetDefaultPrim().GetPath().pathString
    for l in lidars:
        rel = l["prim_path"]
        mount = l.get("mount") or {}
        body = rel.split("/")[0]
        body_path = f"{root}/{body}"
        if not stage.GetPrimAtPath(body_path):
            UsdGeom.Xform.Define(stage, body_path)
        p = author_lidar(stage, f"{root}/{rel}", mount.get("xyz", [0, 0, 0]), float(mount.get("yaw_deg", 0.0)))
        print(f"[inject_lidars] {p.GetPath()} xyz={mount.get('xyz')} yaw={mount.get('yaw_deg', 0.0)}")
    stage.GetRootLayer().Save()
    print(f"[inject_lidars] saved {robot_usd}")


if __name__ == "__main__":
    main()
