"""Arm-vs-fixture clearance for planned joint paths (numpy only; no ROS imports).

Why: in the mobile scene the G2's arm links collide with the cell fixtures, and the base is free, so a
planned path that brushes e.g. a sort-rack upright kicks the base (measured: 6 deg in 0.16 s, 40 deg in
one reach). The Stage 2 planner only kept the arms 0.20 m from the head. This module adds the
environment: forward kinematics from the staged robot URDF (same model MoveIt uses; checked equal to
/compute_fk) of every arm/gripper link's visual-mesh vertices, against the oriented collider boxes of
cell.usda exported by assets/cell/build.py (cell_colliders.json).

    chk = EnvChecker(urdf_path, colliders_json)
    d, name = chk.clearance({joint: rad, ...}, arm="right", base=(x, y, yaw), base_z=0.04)

Speed (docs/OPTIMISATION.md "Motion planning"): the URDF is parsed once (g2cell_motion.kinematics) and
only the arm/head chains are evaluated per sample; a broad phase bounds every (link, box) pair by the
link's bounding sphere (the box SDF is 1-Lipschitz, so sdf(centre) -+ radius bounds every vertex) and
computes exact vertex distances only for pairs that can still beat the best upper bound. Results are
identical to the brute-force version (test_env_check_fast.py); ~40 ms -> ~1-2 ms per sample.
"""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from g2cell_motion.kinematics import UrdfKinematics

ARM_PREFIX = {"left": ("arm_l_link", "gripper_l_"), "right": ("arm_r_link", "gripper_r_")}


def _rpy(rpy):
    a, b, c = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axang(ax, th):
    ax = ax / np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


def _h(r, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def _origin(e):
    o = e.find("origin")
    xyz = np.array((o.get("xyz", "0 0 0") if o is not None else "0 0 0").split(), float)
    rpy = np.array((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split(), float)
    return _h(_rpy(rpy), xyz)


def _dae_points(path, max_pts):
    t = ET.parse(path).getroot()
    ns = {"c": t.tag.split("}")[0][1:]}
    arrs = [np.array(s.find("c:float_array", ns).text.split(), float).reshape(-1, 3)
            for s in t.findall(".//c:mesh/c:source", ns) if "position" in s.get("id", "").lower()]
    if not arrs:
        return np.zeros((0, 3))
    p = np.vstack(arrs)
    if len(p) > max_pts:
        p = p[:: int(math.ceil(len(p) / max_pts))]
    # add the bounding-box corners so thin parts keep their extent after subsampling
    lo, hi = p.min(0), p.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.vstack([p, corners])


class EnvChecker:
    def __init__(self, urdf_path: str, colliders_json: str, max_pts_per_link: int = 120,
                 ignore=("/World/Room/Colliders/Floor",)):
        root = os.path.dirname(urdf_path)
        r = ET.parse(urdf_path).getroot()
        self.children = {}
        for j in r.findall("joint"):
            self.children.setdefault(j.find("parent").get("link"), []).append(j)
        self.points = {}
        for link in r.findall("link"):
            name = link.get("name")
            pts = []
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None or not m.get("filename", "").endswith(".dae"):
                    continue
                f = os.path.join(root, m.get("filename"))
                if not os.path.exists(f):
                    continue
                o = _origin(v)
                p = _dae_points(f, max_pts_per_link)
                pts.append(p @ o[:3, :3].T + o[:3, 3])
            if pts:
                self.points[name] = np.vstack(pts)
        boxes = [b for b in json.load(open(colliders_json)) if b["name"] not in ignore]
        self.box_names = [b["name"] for b in boxes]
        self.c = np.array([b["c"] for b in boxes])               # (B, 3)
        self.ax = np.array([b["axes"] for b in boxes])           # (B, 3, 3) rows = unit axes
        self.h = np.array([b["h"] for b in boxes])               # (B, 3)
        self.kin = UrdfKinematics(urdf_path)
        self.reset_geometry()

    def reset_geometry(self):
        """(Re)build the per-link bounding spheres and per-arm link lists from self.points."""
        below = set()
        for n in list(self.kin.parent_joint):
            try:
                self.kin.chain(n)
                below.add(n)
            except KeyError:
                pass
        self.links = ["base_link"] + sorted(below)
        self.sphere = {}
        for n, p in self.points.items():
            if len(p):
                c = (p.min(0) + p.max(0)) / 2
                self.sphere[n] = (c, float(np.linalg.norm(p - c, axis=1).max()))
        self.arm_links = {arm: [n for n in self.points if n in below and n.startswith(pref)
                                and not n.endswith("link1") and len(self.points[n])]
                          for arm, pref in ARM_PREFIX.items()}

    def link_transforms(self, joints: dict, links=None):
        """{link: 4x4} in base_link for the given joint values (missing joints = 0); all links by default."""
        return self.kin.link_poses(joints, self.links if links is None else links)

    @staticmethod
    def _base_rt(base, base_z):
        x, y, yaw = base
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.array([x, y, base_z])

    def arm_points(self, joints: dict, arm: str, base=(0.0, 0.0, 0.0), base_z: float = 0.04):
        """World (cell-frame) points of the arm's links 2..7 + gripper for the joint state."""
        tf = self.link_transforms(joints, self.arm_links[arm])
        rot, t = self._base_rt(base, base_z)
        p = np.vstack([self.points[n] @ tf[n][:3, :3].T + tf[n][:3, 3] for n in self.arm_links[arm]])
        return p @ rot.T + t

    @staticmethod
    def _sdf(pts, c, ax, h):
        """Signed distance of points (N,3) to boxes -> (B, N)."""
        rel = pts[None, :, :] - c[:, None, :]                               # (B, N, 3)
        loc = rel @ ax.transpose(0, 2, 1)                                    # (B, N, 3)
        q = np.abs(loc) - h[:, None, :]
        return np.linalg.norm(np.maximum(q, 0.0), axis=2) + np.minimum(q.max(axis=2), 0.0)

    def distances(self, pts):
        """Signed distance of every point to every box -> (min over points per box), shape (B,)."""
        d_min = np.full(len(self.c), np.inf)
        for i in range(0, len(pts), 2000):
            d_min = np.minimum(d_min, self._sdf(pts[i:i + 2000], self.c, self.ax, self.h).min(axis=1))
        return d_min

    def _clearance_tf(self, tf, arm, idx, base, base_z):
        """Exact min signed distance of the arm's vertices to boxes idx, with a sphere broad phase."""
        rot, t = self._base_rt(base, base_z)
        links = self.arm_links[arm]
        C, AX, H = self.c[idx], self.ax[idx], self.h[idx]
        cen = np.array([rot @ (tf[n][:3, :3] @ self.sphere[n][0] + tf[n][:3, 3]) + t for n in links])
        rad = np.array([self.sphere[n][1] for n in links])
        sd = self._sdf(cen, C, AX, H).T                                      # (L, B)
        lb = sd - rad[:, None]
        ub = float((sd + rad[:, None]).min()) + 1e-9      # the true minimum is <= every sphere upper bound
        order = np.argsort(lb.min(axis=1), kind="stable")
        best, best_b = math.inf, -1
        for li in order:
            row = lb[li]
            lim = min(best, ub) + 1e-9
            if row.min() > lim:
                break
            bi = np.flatnonzero(row <= lim)
            n = links[li]
            pts = (self.points[n] @ tf[n][:3, :3].T + tf[n][:3, 3]) @ rot.T + t
            d = self._sdf(pts, C[bi], AX[bi], H[bi]).min(axis=1)
            k = int(np.argmin(d))
            if d[k] < best or (d[k] == best and bi[k] < best_b):
                best, best_b = float(d[k]), int(bi[k])
        return best, self.box_names[int(idx[best_b])]

    def _near(self, base, reach: float = 1.6):
        """Indices of boxes that can be within `reach` of the base origin (cheap prefilter)."""
        key = (round(base[0], 3), round(base[1], 3))
        if getattr(self, "_near_key", None) != key:
            d = np.linalg.norm(self.c[:, :2] - np.array(base[:2]), axis=1) - np.linalg.norm(self.h, axis=1)
            self._near_idx = np.nonzero(d < reach)[0]
            self._near_key = key
        return self._near_idx

    def clearance(self, joints: dict, arm: str, base=(0.0, 0.0, 0.0), base_z: float = 0.04):
        """(min signed distance m, name of the closest collider)."""
        tf = self.link_transforms(joints, self.arm_links[arm])
        return self._clearance_tf(tf, arm, self._near(base), base, base_z)

    def sample(self, joints: dict, arm: str, base=(0.0, 0.0, 0.0), base_z: float = 0.04):
        """(fixture clearance, closest collider, head clearance) from ONE kinematics pass."""
        tf = self.link_transforms(joints, self.arm_links[arm] + _check_links(arm) + [HEAD_LINK])
        d, name = self._clearance_tf(tf, arm, self._near(base), base, base_z)
        return d, name, _head_from_tf(tf, arm)

    def path_clearance(self, path: list, arm: str, base=(0.0, 0.0, 0.0), base_z: float = 0.04,
                       samples: int = 10, fixed: dict | None = None):
        """Min clearance along the linear joint interpolation of consecutive dicts in `path`
        (the minimum-jerk profile follows the same geometric joint path)."""
        best = (math.inf, "")
        fixed = fixed or {}
        for a, b in zip(path, path[1:] or path):
            names = list(a)
            for k in range(samples + 1):
                s = k / samples
                j = {**fixed, **{n: a[n] + (b.get(n, a[n]) - a[n]) * s for n in names}}
                d = self.clearance(j, arm, base, base_z)
                if d[0] < best[0]:
                    best = d
        return best


# ---- head clearance + joint frames (waist-aware) -------------------------------------------------------
HEAD_LINK = "head_link3"
HEAD_CAM_IN_LINK3 = (0.10237, 0.02375, 0.10256)      # official head_front_Camera mount (scene YAML)


def _check_links(arm):
    a = arm[0]
    return [f"arm_{a}_link{i}" for i in (3, 4, 5, 6, 7)] + [f"arm_{a}_end_link", f"gripper_{a}_base_link"]


def _joint_child(checker, name):
    for kids in checker.children.values():
        for j in kids:
            if j.get("name") == name:
                return j.find("child").get("link")
    raise KeyError(name)


def joint_origin(checker: EnvChecker, joint: str, joints: dict):
    """Position (base_link) of a joint's child-link frame, e.g. the waist yaw axis."""
    return checker.link_transforms(joints)[_joint_child(checker, joint)][:3, 3]


def head_clearance(checker: EnvChecker, joints: dict, arm: str) -> float:
    """Stage 2 rule, computed with the full joint state (so a turned waist / head is handled):
    min distance from the arm's check-link origins to head_link3's origin and the head camera."""
    return _head_from_tf(checker.link_transforms(joints, _check_links(arm) + [HEAD_LINK]), arm)


def _head_from_tf(tf, arm) -> float:
    h = tf[HEAD_LINK]
    heads = [h[:3, 3], h[:3, :3] @ np.array(HEAD_CAM_IN_LINK3) + h[:3, 3]]
    return float(min(np.linalg.norm(tf[n][:3, 3] - p) for n in _check_links(arm) if n in tf for p in heads))


def path_check(checker: EnvChecker, path: list, arm: str, base=(0.0, 0.0, 0.0), base_z: float = 0.04,
               samples: int = 10, fixed: dict | None = None, stop_below=None):
    """Along the linear joint interpolation of `path` (dicts may include the waist):
    (min fixture clearance, closest collider, min head clearance).
    stop_below=(env_min, head_min): return as soon as a sample is below either (the path is rejected
    anyway; the values returned are then the minima so far, not over the whole path)."""
    fixed = fixed or {}
    env, name, head = math.inf, "", math.inf
    pairs = list(zip(path, path[1:])) or [(path[0], path[0])]
    for a, b in pairs:
        keys = set(a) | set(b)
        for k in range(samples + 1):
            s = k / samples
            j = dict(fixed)
            for n in keys:
                va, vb = a.get(n, b.get(n)), b.get(n, a.get(n))
                j[n] = va + (vb - va) * s
            d, nm, hd = checker.sample(j, arm, base, base_z)
            if d < env:
                env, name = d, nm
            head = min(head, hd)
            if stop_below is not None and (env < stop_below[0] or head < stop_below[1]):
                return env, name, head
    return env, name, head
