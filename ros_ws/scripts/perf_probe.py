#!/usr/bin/env python3
"""Measure the running Genie Sim stack from inside the g2cell container (one JSON on stdout).

Collects, over --duration seconds:
  * topic rates (best-effort QoS, the engine/render publishers' QoS): /joint_states, camera image_raw
  * CPU % per process (from /proc/<pid>/stat deltas; 100 % = one core)
  * render / physics stats parsed from the sim.log lines written during the window
    ("Render Stats", "[prof]" per-camera map/publish, "Physics Stats", "loop fell behind")
  * startup markers from the whole log: time from stage1_up.sh start (/g2cell_output/perf_t0)
    to "Renderer created" and "OVRTX first frame complete"

Host-side GPU sampling (nvidia-smi) is done by tools/sim_perf.py, which calls this script.
Usage: python3 perf_probe.py [--duration 20] [--log /g2cell_output/sim.log]
"""
import argparse
import json
import os
import re
import statistics as st
import time

TOPICS = [
    ("/joint_states", "sensor_msgs/msg/JointState"),
    ("/genie_sim/head_front_camera_rgb/image_raw", "sensor_msgs/msg/Image"),
    ("/genie_sim/head_front_camera_depth/image_raw", "sensor_msgs/msg/Image"),
    ("/genie_sim/free_camera_rgb/image_raw", "sensor_msgs/msg/Image"),
    ("/tf_render", "tf2_msgs/msg/TFMessage"),
    ("/lidar/livox_front", "sensor_msgs/msg/PointCloud2"),
    ("/odom", "nav_msgs/msg/Odometry"),
]
HZ = os.sysconf("SC_CLK_TCK")


def proc_times():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
            rest = s[s.rindex(")") + 2:].split()
            ticks = int(rest[11]) + int(rest[12])
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            out[int(pid)] = (ticks, cmd)
        except (OSError, ValueError, IndexError):
            continue
    return out


def short_name(cmd):
    rules = [("genie_sim_render_node", "render_ovrtx"), ("engine_isaacsim", "engine (Isaac PhysX)"),
             ("head_cam_tf_relay", "head_cam_tf_relay.py"), ("free_cam_pose", "free_cam_pose.py"),
             ("move_group", "move_group"), ("foxglove_bridge", "foxglove_bridge"),
             ("robot_state_publisher", "robot_state_publisher"), ("moveit_joint_states_bridge", "moveit_js_bridge"),
             ("render_profile", "render_profile.py"), ("ik_replay", "ik_replay.py"), ("perf_probe", "perf_probe.py"), ("ros2 launch", "ros2 launch"),
             ("chassis_servo", "chassis_servo")]
    for k, v in rules:
        if k in cmd:
            return v
    return cmd.split(" ")[0][-40:] or "?"


def floats(pattern, text, cast=float):
    return [cast(m) for m in re.findall(pattern, text)]


def mean(xs):
    return round(st.mean(xs), 2) if xs else None


def parse_window(text):
    r = {}
    fps = floats(r"Actual FPS: ([\d.]+)", text)
    r["render_fps"] = mean(fps[1:] or fps)  # first block may straddle the window start
    tot = floats(r"Total:\s+mean=([\d.]+)", text)
    r["render_total_ms"] = mean(tot)
    wait = floats(r"Wait:\s+mean=([\d.]+)", text)
    r["render_wait_ms"] = mean(wait)
    cams = {}
    for m in re.finditer(r"cam\[(\d+)\] (\S+) (\d+x\d+): map=([\d.]+)ms pub=([\d.]+)ms", text):
        c = cams.setdefault(m.group(2), {"res": m.group(3), "map": [], "pub": []})
        c["map"].append(float(m.group(4)))
        c["pub"].append(float(m.group(5)))
    r["cameras"] = {k: {"res": v["res"], "map_ms": mean(v["map"]), "pub_ms": mean(v["pub"])} for k, v in cams.items()}
    r["readback_total_ms"] = mean(floats(r"gpu_readback\(ovrtx_map\)=([\d.]+)ms", text))
    r["memcpy_ms"] = mean(floats(r"memcpy=([\d.]+)ms", text))
    ph = {}
    ph["actual_hz"] = mean(floats(r"Actual Hz: ([\d.]+)", text))
    ov = re.findall(r"overruns: (\d+)/(\d+)", text)
    ph["overruns_per_500"] = mean([int(a) for a, _ in ov])
    ph["step_mean_ms"] = mean(floats(r"engine_isaacsim\.py-\d+\]\s+Step:\s+mean=([\d.]+)", text))
    ph["step_p99_ms"] = mean(floats(r"engine_isaacsim\.py-\d+\]\s+Step:\s+mean=[\d.]+.*?p99=([\d.]+)", text))
    ph["solver_mean_ms"] = mean(floats(r"Solver:\s+mean=([\d.]+)", text))
    ph["publish_mean_ms"] = mean(floats(r"Publish:\s+mean=([\d.]+)", text))
    ph["spin_mean_ms"] = mean(floats(r"Spin:\s+mean=([\d.]+)", text))
    ph["interval_p99_ms"] = mean(floats(r"Interval:\s+mean=[\d.]+.*?p99=([\d.]+) ms \(target=10", text))
    ph["budget_mean_pct"] = mean(floats(r"Budget:\s+mean=[\d.]+ms \((\d+)%\)", text))
    ph["budget_worst_ms"] = mean(floats(r"Budget:.*?worst=([\d.]+)ms", text))
    clamps = floats(r"loop fell behind by >1 tick: clamped (\d+) times", text, int)
    ph["fell_behind_warnings"] = len(clamps)
    r["physics"] = ph
    return r


def startup(full):
    s = {}
    try:
        t0 = float(open("/g2cell_output/perf_t0").read().split()[0])
    except (OSError, ValueError, IndexError):
        t0 = None
    s["t0"] = t0

    def ts(marker):
        m = re.search(r"\[(\d{10}\.\d+)\] \[render_ovrtx\]: " + re.escape(marker), full)
        return float(m.group(1)) if m else None
    for key, marker in (("renderer_create_start", "Creating ovrtx renderer"),
                        ("renderer_created", "Renderer created"),
                        ("first_frame", "OVRTX first frame complete")):
        v = ts(marker)
        s[key + "_s"] = round(v - t0, 1) if (v and t0) else None
    m = re.search(r"\[(\d{10}\.\d+)\] \[robot_state_publisher\]: Robot initialized", full)
    s["rsp_up_s"] = round(float(m.group(1)) - t0, 1) if (m and t0) else None
    hb = floats(r"still compiling shaders / warming GPU \((\d+)s elapsed\)", full, int)
    s["shader_warmup_heartbeat_s"] = max(hb) if hb else 0
    s["viewport_line"] = (re.findall(r"viewport render mode: .*", full) or [None])[-1]
    s["viewport_updates"] = (re.findall(r"viewport updates_enabled = \w+", full) or [None])[-1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--log", action="append", default=[],
                    help="logs to parse (default: /g2cell_output/sim.log + render.log when present)")
    a = ap.parse_args()

    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    rclpy.init()
    node = rclpy.create_node("g2cell_perf_probe")
    stamps = {t: [] for t, _ in TOPICS}
    for t, typ in TOPICS:
        # images are published RELIABLE (a best-effort reader drops the fragmented 1-4 MB frames on
        # loopback); joint_states / tf_render are best-effort
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE) if ("Image" in typ or "PointCloud2" in typ) \
            else qos_profile_sensor_data
        node.create_subscription(get_message(typ), t, lambda _m, t=t: stamps[t].append(time.monotonic()), q)
    logs = a.log or [p for p in ("/g2cell_output/sim.log", "/g2cell_output/render.log") if os.path.exists(p)]
    offs = {p: os.path.getsize(p) for p in logs}
    p0, w0 = proc_times(), time.monotonic()
    while time.monotonic() - w0 < a.duration:
        rclpy.spin_once(node, timeout_sec=0.02)
    p1, w1 = proc_times(), time.monotonic()
    node.destroy_node()
    rclpy.shutdown()

    rates = {}
    for t, s in stamps.items():
        if len(s) >= 2:
            d = [b - c for c, b in zip(s, s[1:])]
            rates[t] = {"hz": round((len(s) - 1) / (s[-1] - s[0]), 2), "max_gap_ms": round(max(d) * 1000, 1)}
        else:
            rates[t] = {"hz": 0.0, "max_gap_ms": None}
    cpu = {}
    for pid, (t1, cmd) in p1.items():
        if pid in p0:
            pct = (t1 - p0[pid][0]) / HZ / (w1 - w0) * 100.0
            if pct >= 0.5:
                k = short_name(cmd)
                cpu[k] = round(cpu.get(k, 0.0) + pct, 1)
    full, window = "", ""
    for p in logs:
        with open(p, errors="replace") as f:
            t = f.read()
        full += t
        window += t[offs[p]:] if len(t) >= offs[p] else t
    out = {"duration_s": a.duration, "topic_rates": rates,
           "cpu_pct_per_process": dict(sorted(cpu.items(), key=lambda kv: -kv[1])),
           "cpu_pct_total": round(sum(cpu.values()), 1),
           "load_avg": open("/proc/loadavg").read().split()[:3],
           "window": parse_window(window), "startup": startup(full)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
