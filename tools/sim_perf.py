#!/usr/bin/env python3
"""Measure the running sim stack: container probe (ros_ws/scripts/perf_probe.py) + host nvidia-smi sampling.

Usage (host, sim already up in container g2cell):
  python3 tools/sim_perf.py --label baseline [--duration 20] [--out output/perf/baseline.json]

Writes one JSON with: GPU utilisation/memory (1 Hz samples over the window, plus per-process VRAM),
topic rates, CPU per process, render/physics stats from sim.log, and startup times.
"""
import argparse
import json
import os
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smi(query, kind="gpu"):
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def proc_name(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return f"pid {pid}"
    for k, v in (("genie_sim_render_node", "render_ovrtx"), ("engine_isaacsim", "engine (Isaac PhysX)"),
                 ("move_group", "move_group")):
        if k in cmd:
            return v
    return cmd.split(" ")[0][-40:]


MARKERS = {"renderer_created": "Renderer created", "first_frame": "OVRTX first frame complete",
           "engine_ready": "physics engine ready", "engine_loop_stats": "Physics Stats",
           "first_render_stats": "Render Stats"}


def watch_startup(logs, t0_file, timeout=900.0):
    """Poll sim.log from launch until every marker has appeared; wall seconds since stage1_up.sh start."""
    deadline, seen = time.time() + timeout, {}
    while time.time() < deadline and len(seen) < len(MARKERS):
        try:
            t0 = float(open(t0_file).read().split()[0])
            text = "".join(open(p, errors="replace").read() for p in logs if os.path.exists(p))
        except (OSError, ValueError, IndexError):
            time.sleep(0.5)
            continue
        for k, m in MARKERS.items():
            if k not in seen and m in text:
                seen[k] = round(time.time() - t0, 1)
        time.sleep(0.5)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-startup", action="store_true",
                    help="first wait for the startup markers (run right after launching) and record their times")
    ap.add_argument("--label", required=True)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--container", default="g2cell")
    a = ap.parse_args()
    out_path = a.out or os.path.join(ROOT, "output", "perf", f"{a.label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    startup_watch = {}
    if a.watch_startup:
        startup_watch = watch_startup([os.path.join(ROOT, "output", f) for f in ("sim.log", "render.log")],
                                      os.path.join(ROOT, "output", "perf_t0"))
        print("startup:", startup_watch, flush=True)
        time.sleep(20.0)  # let the relay / freecam / stats settle before the steady-state window

    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            r = smi("utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm")
            if r:
                samples.append([float(x) if x.replace(".", "", 1).isdigit() else None for x in r[0]])
            stop.wait(1.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    # pass the host's ROS_DOMAIN_ID through (a sim started on a private domain is invisible otherwise)
    dom = ["-e", f"ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}"] if os.environ.get("ROS_DOMAIN_ID") else []
    cmd = ["docker", "exec", "-u", "isaac-sim", *dom, a.container, "bash", "-lc",
           "source /workspace/devel/setup.bash >/dev/null 2>&1; "
           f"python3 /g2cell_ws/scripts/perf_probe.py --duration {a.duration}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    apps = smi("pid,used_memory", kind="apps")
    stop.set()
    th.join()
    try:
        probe = json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise SystemExit(f"probe failed:\n{res.stdout}\n{res.stderr}")

    util = [s[0] for s in samples if s[0] is not None]
    mem = [s[1] for s in samples if s[1] is not None]
    pwr = [s[4] for s in samples if s[4] is not None]
    gpu = {
        "util_pct_mean": round(sum(util) / len(util), 1) if util else None,
        "util_pct_min": min(util) if util else None,
        "util_pct_max": max(util) if util else None,
        "mem_used_mib": max(mem) if mem else None,
        "mem_total_mib": samples[0][2] if samples else None,
        "power_w_mean": round(sum(pwr) / len(pwr), 1) if pwr else None,
        "samples": len(samples),
        "per_process_mib": {proc_name(p): float(m) for p, m in apps if m.isdigit()},
    }
    report = {"label": a.label, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu": gpu, "startup_watch_s": startup_watch, **probe}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
