#!/usr/bin/env python3
"""Start Genie Sim's OVRTX render node (render_ovrtx) with a cost profile, without touching upstream.

Why a wrapper: upstream launches render_ovrtx from physics_isaacsim.launch.py with a hard-coded
render_fps 30 (a dict appended AFTER the launcher YAML, so the YAML can't override it), every camera in
the manifest plus the 1280x720 FreeCam, and assemble_scene.py hard-codes
`omni:rtx:rendermode = "RealTimePathTracing"` on every RenderProduct. stage1_up.sh with PROFILE=...
uses launcher_ovrtx_isaac_physx_norender.yaml (renders: []) and starts this script instead, which:

  1. waits until this launch's assemble_scene has written manifest.json / render_layer.usda
     (the engine process starts only after assemble_scene exits);
  2. writes manifest_<profile>.json + render_layer_<profile>.usda next to them (same dir, so relative
     paths still resolve): render mode replaced on every camera RenderProduct, FreeCam dropped from the
     manifest's camera list when --freecam off (a product that is not listed is never stepped/rendered);
  3. execs genie_sim_render_node with the same name/remaps as upstream plus our render_fps / lidar.

Scene content, cameras, intrinsics, topics and frame ids are unchanged.

Usage: render_profile.py --scene <scene.yaml> --profile collect [--mode raster] [--fps 30] [--freecam off]
       [--lidar 0] [--t0-file /g2cell_output/perf_t0] [--dry-run]
"""
import argparse
import json
import os
import re
import site
import sys
import time

# render_mode name -> omni:rtx:rendermode token (ovrtx rtx_settings schema allowedTokens:
# "RaytracedLighting", "PathTracing", "Real-Time Path-Tracing", "Minimal"). Measured on ovrtx here: the three
# real-time tokens render the same image at the same cost; "PathTracing" is the progressive offline tracer;
# "Minimal" renders black frames with our scene (not offered).
MODES = {
    "raster": "RaytracedLighting",          # RTX real-time: raster + RT shadows/reflections/AO + denoise
    "rtpt": "Real-Time Path-Tracing",       # RTX real-time path tracing (schema spelling)
    "pathtrace": "RealTimePathTracing",     # what upstream assemble_scene.py authors (kept verbatim)
    "offline": "PathTracing",               # offline path tracing (diagnostic; very slow)
}
PROFILES = {
    #            mode         fps   freecam
    "collect": ("raster", 30.0, "off"),     # teleop data collection / policy eval: head RGB+D only
    "teleop": ("raster", 15.0, "off"),      # interactive driving / debugging, lowest GPU load
    "showcase": ("pathtrace", 30.0, "on"),  # videos: upstream look, FreeCam 1280x720
}


def assets_root():
    try:
        import geniesim_assets  # noqa: PLC0415
        return str(geniesim_assets.ASSETS_PATH)
    except Exception:
        return "/geniesim_assets"


def ovrtx_root():
    for d in site.getsitepackages() + [site.getusersitepackages()]:
        c = os.path.join(d, "ovrtx", "bin")
        if os.path.isdir(c):
            return c
    return ""


def wait_for_stage(stage_dir, t0, sim_log, timeout):
    """Fresh manifest + render layer from THIS launch, and the engine started (assemble finished)."""
    man, layer = os.path.join(stage_dir, "manifest.json"), os.path.join(stage_dir, "render_layer.usda")
    deadline = time.time() + timeout
    while time.time() < deadline:
        fresh = all(os.path.exists(p) and os.path.getmtime(p) >= t0 - 1.0 for p in (man, layer))
        started = False
        try:
            with open(sim_log, errors="replace") as f:
                log = f.read()
                # engine process tag: upstream genie_sim_engine_isaacsim.py or g2cell_engine_isaacsim.py (OBJ_RELAY=1)
                started = re.search(r"engine_isaacsim\.py-\d+\]", log) is not None and fresh
        except OSError:
            pass
        if fresh and started:
            time.sleep(3.0)  # same 3 s delay upstream puts in front of render_ovrtx
            return man, layer
        time.sleep(0.5)
    raise SystemExit(f"[render_profile] no fresh manifest in {stage_dir} after {timeout:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene YAML (its stem names the stage dir)")
    ap.add_argument("--profile", default="collect", choices=sorted(PROFILES))
    ap.add_argument("--mode", default="", choices=[""] + sorted(MODES), help="override the profile's mode")
    ap.add_argument("--fps", type=float, default=0.0, help="override the profile's render_fps")
    ap.add_argument("--freecam", default="", choices=["", "on", "off"])
    ap.add_argument("--freecam-first", action="store_true", help="(diagnostic) list FreeCam before the head camera")
    ap.add_argument("--lidar", type=int, default=0)
    ap.add_argument("--t0-file", default="/g2cell_output/perf_t0")
    ap.add_argument("--sim-log", default="/g2cell_output/sim.log")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--no-wait", action="store_true", help="use the current manifest (render-node restart)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mode, fps, freecam = PROFILES[a.profile]
    mode, fps, freecam = a.mode or mode, a.fps or fps, a.freecam or freecam
    token = MODES[mode]
    stage_dir = os.path.join(assets_root(), "scenes", os.path.splitext(os.path.basename(a.scene))[0])
    if a.no_wait:
        man, layer = os.path.join(stage_dir, "manifest.json"), os.path.join(stage_dir, "render_layer.usda")
    else:
        t0 = float(open(a.t0_file).read().split()[0]) if os.path.exists(a.t0_file) else time.time()
        man, layer = wait_for_stage(stage_dir, t0, a.sim_log, a.timeout)

    tag = f"{a.profile}"
    text = open(layer).read()
    text, n = re.subn(r'(token omni:rtx:rendermode = )"[^"]*"', rf'\1"{token}"', text)
    layer_out = os.path.join(stage_dir, f"render_layer_{tag}.usda")
    with open(layer_out, "w") as f:
        f.write(text)

    m = json.load(open(man))
    rel_layer = m.get("render_layer_usda", "")
    m["render_layer_usda"] = os.path.join(os.path.dirname(rel_layer), os.path.basename(layer_out)) if rel_layer \
        else layer_out
    cams = m.get("cameras", [])
    if freecam == "off":
        cams = [c for c in cams if not c.get("is_free_cam")]
    if a.freecam_first:
        cams = sorted(cams, key=lambda c: not c.get("is_free_cam"))
    m["cameras"] = cams
    man_out = os.path.join(stage_dir, f"manifest_{tag}.json")
    with open(man_out, "w") as f:
        json.dump(m, f, indent=4)
    print(f"[render_profile] profile={a.profile} mode={mode} ({token!r} on {n} products) fps={fps:g} "
          f"freecam={freecam} lidar={a.lidar} cameras={[c['topic'] for c in cams]} manifest={man_out}", flush=True)

    cmd = ["ros2", "run", "genie_sim_render", "genie_sim_render_node", "--ros-args", "--log-level", "info",
           "-r", "__node:=render_ovrtx", "-r", "~/free_cam_pose:=/genie_sim_engine/viewer/camera_pose",
           "-p", f"stage_manifest:={man_out}", "-p", f"render_fps:={fps}", "-p", f"ovrtx_root:={ovrtx_root()}",
           "-p", "use_sim_time:=false", "-p", f"lidar:={'true' if a.lidar else 'false'}",
           "-p", "plugin:=[genie_sim_render/RosImagePublisherPlugin]"]
    print("[render_profile] exec:", " ".join(cmd), flush=True)
    if a.dry_run:
        return 0
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
