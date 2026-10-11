#!/usr/bin/env python3
"""Live RViz replay of a recorded episode (+ open-loop policy prediction) on ROS 2 Humble.

Runs with the HOST /usr/bin/python3 (rclpy from /opt/ros/humble, numpy only; no h5py/torch):
the episode is first converted by export_replay.py to a .npz. Started by replay.launch.py
(see replay_rviz.sh), which also runs one robot_state_publisher per robot:

    /recorded/joint_states -> robot_state_publisher (frame_prefix recorded/) -> solid robot
    /policy/joint_states   -> robot_state_publisher (frame_prefix policy/)   -> ghost robot
    /replay/markers        MarkerArray in base_link: end-effector paths (both arms, recorded
                           blue/cyan vs policy orange/red: faint full path + solid trail so far
                           + a sphere at the current point) and a text panel (time, speed,
                           gripper states, policy MSE)
    static TF              base_link -> recorded/base_link, base_link -> policy/base_link (identity)

The ghost shows the policy's step-0 prediction (open loop: the policy always sees the recorded
observation). Episodes exported without a checkpoint show the recorded *command* as the ghost.

Parameters (all settable live, e.g. ``ros2 param set /g2_replay paused true``):
    npz (str)  urdf (str)  speed (float, 1.0)  loop (bool, true)  paused (bool, false)
    rate_hz (float, 60)  seek (float s; set to jump, then it resets to -1)
    loop_pause_s (float, 1.0: hold the last frame before looping)
    ghost_offset_y (float m, 0: startup only; shift the ghost sideways when it hides behind the solid robot)
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import rclpy.executors
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

# end-effector point = midpoint of the two fingertip links of each gripper
EE_LINKS = {"l": ("gripper_l_inner_link4", "gripper_l_outer_link4"),
            "r": ("gripper_r_inner_link4", "gripper_r_outer_link4")}
GRIP_MASTER = {"l": "idx31_gripper_l_inner_joint1", "r": "idx71_gripper_r_inner_joint1"}
COL = {("rec", "l"): (0.10, 0.45, 1.00), ("rec", "r"): (0.10, 0.85, 0.95),
       ("pol", "l"): (1.00, 0.55, 0.05), ("pol", "r"): (0.95, 0.15, 0.25)}


# ------------------------------------------------------------------ minimal URDF FK (numpy)

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, q):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


class UrdfFK:
    def __init__(self, urdf_xml: str):
        root = ET.fromstring(urdf_xml)
        self.joints = {}          # child link -> joint dict
        self.mimic = {}           # joint -> (master, multiplier, offset)
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            d = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                     T=T, axis=axis)
            self.joints[j.find("child").get("link")] = d
            m = j.find("mimic")
            if m is not None:
                self.mimic[d["name"]] = (m.get("joint"), float(m.get("multiplier", 1)),
                                         float(m.get("offset", 0)))

    def chain(self, link):
        out = []
        while link in self.joints:
            out.append(self.joints[link])
            link = self.joints[link]["parent"]
        return out[::-1], link

    def position(self, link, q: dict) -> np.ndarray:
        """Origin of `link` in the URDF root frame for joint values q (name -> rad / m)."""
        chain, _ = self.chain(link)
        T = np.eye(4)
        for j in chain:
            T = T @ j["T"]
            v = q.get(j["name"], 0.0)
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], v)
                T = T @ R
            elif j["type"] == "prismatic":
                P = np.eye(4)
                P[:3, 3] = j["axis"] / np.linalg.norm(j["axis"]) * v
                T = T @ P
        return T[:3, 3]

    def apply_mimic(self, names, pos):
        """pos [N,J]: overwrite mimic joints from their master columns (in place)."""
        idx = {n: i for i, n in enumerate(names)}
        for jn, (master, mul, off) in self.mimic.items():
            if jn in idx and master in idx:
                pos[:, idx[jn]] = mul * pos[:, idx[master]] + off
        return pos


# ------------------------------------------------------------------ node

class ReplayNode(Node):
    def __init__(self):
        super().__init__("g2_replay")
        self.declare_parameter("npz", "")
        self.declare_parameter("urdf", "")
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("paused", False)
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("seek", -1.0)
        self.declare_parameter("loop_pause_s", 1.0)
        self.declare_parameter("ghost_offset_y", 0.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.speed, self.loop, self.paused = float(g("speed")), bool(g("loop")), bool(g("paused"))
        self.loop_pause = float(g("loop_pause_s"))
        self.ghost_dy = float(g("ghost_offset_y"))

        d = np.load(g("npz"), allow_pickle=False)
        self.meta = json.loads(str(d["meta_json"]))
        self.names = [str(n) for n in d["js_names"]]
        self.js_t, self.js_pos = d["js_t"], d["js_pos"]
        self.al_t = d["al_t"]
        self.has_policy = bool(d["has_policy"])
        cmd_names = [str(n) for n in d["cmd_names"]]
        with open(g("urdf")) as f:
            self.fk = UrdfFK(f.read())
        if self.has_policy:
            ghost = d["pol_pos"].copy()
            self.ghost_label = "policy (open-loop step-0 prediction)"
        else:   # no checkpoint: ghost = recorded command
            ghost = d["al_pos"].copy()
            for k, n in enumerate(cmd_names):
                ghost[:, self.names.index(n)] = d["al_cmd"][:, k]
            self.ghost_label = "recorded command"
        self.ghost = self.fk.apply_mimic(self.names, ghost)
        self.duration = float(self.al_t[-1])

        # end-effector paths (precomputed, base_link frame)
        self.path = {}
        rec_al = np.array([self.js_pos[self._js_idx(t)] for t in self.al_t])
        for tag, P in (("rec", rec_al), ("pol", self.ghost)):
            for arm, links in EE_LINKS.items():
                pts = []
                for row in P:
                    q = dict(zip(self.names, row))
                    pts.append(np.mean([self.fk.position(lk, q) for lk in links], axis=0))
                self.path[(tag, arm)] = np.array(pts)
        err = {arm: float(np.linalg.norm(self.path[("rec", arm)] - self.path[("pol", arm)], axis=1).mean())
               for arm in EE_LINKS}
        for arm in EE_LINKS:   # ghost robot (and its paths) may be shifted sideways
            self.path[("pol", arm)] = self.path[("pol", arm)] + np.array([0.0, self.ghost_dy, 0.0])
        self.ee_err = err
        self.get_logger().info(
            f"loaded {self.meta['episode']}: {len(self.al_t)} aligned steps, "
            f"{self.duration:.2f} s, ghost = {self.ghost_label}; mean EE distance "
            f"rec-vs-ghost L {err['l']*1000:.1f} mm, R {err['r']*1000:.1f} mm")

        self.pub_rec = self.create_publisher(JointState, "/recorded/joint_states", 10)
        self.pub_pol = self.create_publisher(JointState, "/policy/joint_states", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/replay/markers", 10)
        self.static = StaticTransformBroadcaster(self)
        tfs = []
        for pfx in ("recorded", "policy"):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = "base_link", f"{pfx}/base_link"
            t.transform.rotation.w = 1.0
            if pfx == "policy":
                t.transform.translation.y = self.ghost_dy
            tfs.append(t)
        self.static.sendTransform(tfs)

        self.t = 0.0
        self.hold = 0.0
        self.last_wall = self.get_clock().now()
        self.add_on_set_parameters_callback(self._on_params)
        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self._tick)
        self.mk_every = max(1, int(round(float(g("rate_hz")) / 15.0)))   # markers at ~15 Hz
        self.n = 0

    def _js_idx(self, t):
        return int(np.clip(np.searchsorted(self.js_t, t, side="right") - 1, 0, len(self.js_t) - 1))

    def _al_idx(self, t):
        return int(np.clip(np.searchsorted(self.al_t, t, side="right") - 1, 0, len(self.al_t) - 1))

    def _on_params(self, params):
        for p in params:
            if p.name == "speed":
                self.speed = float(p.value)
            elif p.name == "loop":
                self.loop = bool(p.value)
            elif p.name == "paused":
                self.paused = bool(p.value)
            elif p.name == "loop_pause_s":
                self.loop_pause = float(p.value)
            elif p.name == "seek" and float(p.value) >= 0:
                self.t = min(float(p.value), self.duration)
                self.hold = 0.0
        return SetParametersResult(successful=True)

    def _tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_wall).nanoseconds * 1e-9
        self.last_wall = now
        if self.get_parameter("seek").value >= 0:
            self.set_parameters([Parameter("seek", Parameter.Type.DOUBLE, -1.0)])
        if not self.paused:
            if self.t >= self.duration:
                self.hold += dt
                if self.loop and self.hold >= self.loop_pause:
                    self.t, self.hold = 0.0, 0.0
            else:
                self.t = min(self.t + dt * self.speed, self.duration)
        stamp = now.to_msg()
        ji, ai = self._js_idx(self.t), self._al_idx(self.t)
        for pub, row in ((self.pub_rec, self.js_pos[ji]), (self.pub_pol, self.ghost[ai])):
            m = JointState()
            m.header.stamp = stamp
            m.name = self.names
            m.position = [float(v) for v in row]
            pub.publish(m)
        if self.n % self.mk_every == 0:
            self.pub_mk.publish(self._markers(stamp, ai, ji))
        self.n += 1

    # -- markers
    def _base(self, stamp, ns, mid, mtype):
        m = Marker()
        m.header.frame_id, m.header.stamp = "base_link", stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = Duration(sec=1)
        m.frame_locked = False
        return m

    def _markers(self, stamp, ai, ji):
        arr = MarkerArray()
        mid = 0
        for (tag, arm), pts in self.path.items():
            r, g_, b = COL[(tag, arm)]
            for kind, seg, alpha, width in (("full", pts, 0.45, 0.006),
                                            ("trail", pts[: ai + 1], 1.0, 0.012)):
                if len(seg) < 2:
                    continue
                m = self._base(stamp, f"{tag}_{arm}_{kind}", mid, Marker.LINE_STRIP)
                m.scale.x = width
                m.color = ColorRGBA(r=r, g=g_, b=b, a=alpha)
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in seg]
                arr.markers.append(m)
                mid += 1
            s = self._base(stamp, f"{tag}_{arm}_ee", mid, Marker.SPHERE)
            p = pts[ai]
            s.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            s.scale.x = s.scale.y = s.scale.z = 0.035
            s.color = ColorRGBA(r=r, g=g_, b=b, a=0.9)
            arr.markers.append(s)
            mid += 1

        def grip(row, arm):   # 0 rad = closed, -0.785 rad = fully open (fingertip gap 4.5 -> 11 cm)
            q = float(row[self.names.index(GRIP_MASTER[arm])])
            return f"{min(abs(q) / 0.785, 1) * 100:.0f}% open"
        rec, gh = self.js_pos[ji], self.ghost[ai]
        state = "PAUSED" if self.paused else ("END" if self.t >= self.duration else "PLAY")
        # rviz2 (this Ogre font) draws every space as a ~3-character gap: spaces -> "_" below
        lines = [self.meta["episode"].split("/")[-1],
                 f"t {self.t:.2f}/{self.duration:.2f}s x{self.speed:g} {state}{' loop' if self.loop else ''}",
                 f"grip L: rec {grip(rec, 'l')} | ghost {grip(gh, 'l')}",
                 f"grip R: rec {grip(rec, 'r')} | ghost {grip(gh, 'r')}",
                 f"ghost = {self.ghost_label}"]
        if self.has_policy:
            pi = self.meta["policy"]
            lines.append(f"{pi['policy']} step {pi['step']}, MSE {pi['action_mse_rad2']:.1e} rad^2"
                         f"{', TRAINED ON THIS EPISODE' if pi.get('trained_on_this_episode') else ''}")
        lines.append(f"mean EE gap rec-ghost: L {self.ee_err['l']*1000:.1f} mm, R {self.ee_err['r']*1000:.1f} mm")
        lines.append("paths: blue/cyan = recorded L/R, orange/red = ghost L/R")
        txt = self._base(stamp, "text", mid, Marker.TEXT_VIEW_FACING)
        txt.pose.position = Point(x=0.0, y=self.ghost_dy / 2.0, z=2.05)
        txt.scale.z = 0.04
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        txt.text = "\n".join(lines).replace(" ", "_")
        arr.markers.append(txt)
        return arr


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
