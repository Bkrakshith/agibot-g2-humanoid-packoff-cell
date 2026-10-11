"""Fast local kinematics for the G2 from its URDF (numpy only, no ROS, deterministic).

- UrdfKinematics: parses the URDF ONCE (joint origins, axes, limits) and computes link poses in
  base_link for a joint dict, walking only the chains needed for the requested links.
- ArmIk: batched damped-least-squares 6-D IK for one 7-DoF arm (all other joints, e.g. the waist,
  held at the values in `state`). Same model MoveIt loads (the staged robot.urdf), so a local
  solution handed to /compute_ik as the seed converges there immediately (MoveIt stays the
  confirmation). Deterministic: restarts come from a fixed-seed RNG, so the same inputs always
  give the same answer.

The demo's old path (random-jitter /compute_ik retries, each failing call burning KDL's 0.1 s
timeout) took 2-5 min per run; see docs/OPTIMISATION.md "Motion planning".
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_rot(axis, th):
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(th), math.sin(th)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def quat_to_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Joint:
    __slots__ = ("name", "parent", "child", "type", "T", "axis", "lo", "hi")

    def __init__(self, e):
        self.name = e.get("name")
        self.parent = e.find("parent").get("link")
        self.child = e.find("child").get("link")
        self.type = e.get("type")
        o = e.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        self.T = np.eye(4)
        self.T[:3, :3], self.T[:3, 3] = _rpy(*rpy), xyz
        a = e.find("axis")
        ax = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0.0, 0.0])
        n = np.linalg.norm(ax)
        self.axis = ax / n if n > 0 else np.array([1.0, 0.0, 0.0])
        lim = e.find("limit")
        self.lo = float(lim.get("lower", -math.pi)) if lim is not None else -math.pi
        self.hi = float(lim.get("upper", math.pi)) if lim is not None else math.pi
        if self.type == "continuous":
            self.lo, self.hi = -math.inf, math.inf

    def motion(self, q):
        m = np.eye(4)
        if self.type in ("revolute", "continuous"):
            m[:3, :3] = axis_rot(self.axis, q)
        elif self.type == "prismatic":
            m[:3, 3] = self.axis * q
        return m


class UrdfKinematics:
    """Link poses in `root` (base_link) from a URDF. Parsed once; per-call cost = the joints on the
    requested chains only."""

    def __init__(self, urdf: str, root: str = "base_link"):
        src = ET.fromstring(urdf) if urdf.lstrip().startswith("<") else ET.parse(urdf).getroot()
        self.joints = {}
        self.parent_joint = {}
        for e in src.findall("joint"):
            j = Joint(e)
            self.joints[j.name] = j
            self.parent_joint[j.child] = j
        self.root = root
        self._chains = {}

    def chain(self, link: str) -> list:
        """Joints from root to `link`, root first."""
        if link not in self._chains:
            out, cur = [], link
            while cur != self.root:
                j = self.parent_joint.get(cur)
                if j is None:
                    raise KeyError(f"{link} is not below {self.root}")
                out.append(j)
                cur = j.parent
            self._chains[link] = out[::-1]
        return self._chains[link]

    def link_poses(self, joints: dict, links) -> dict:
        """{link: 4x4} in root for the joint values (missing joints = 0)."""
        memo = {self.root: np.eye(4)}
        out = {}
        for link in links:
            if link not in memo:
                T = np.eye(4)
                for j in self.chain(link):
                    if j.child in memo:
                        T = memo[j.child]
                        continue
                    T = T @ j.T
                    if j.type != "fixed":
                        T = T @ j.motion(joints.get(j.name, 0.0))
                    memo[j.child] = T
            out[link] = memo[link]
        return out

    def limits(self, names):
        return (np.array([self.joints[n].lo for n in names]), np.array([self.joints[n].hi for n in names]))


class ArmIk:
    """Batched DLS IK for a serial arm (joints `names`, root-first) to `end_link`'s pose in root.

    solve(pos, quat, state, seeds) -> {joint: rad} | None. Stage 1 runs the given seeds (previous solution,
    home ...); only if none converges, stage 2 adds `n_jitter` seeds jittered +-jitter around the first
    seed and `n_random` uniform within the limits, all from np.random.default_rng(rng_seed): fixed, so
    repeated calls are bit-identical. Picks the first-seed solution if it converged (continuity), else the
    converged solution closest to the first seed."""

    def __init__(self, kin: UrdfKinematics, names, end_link: str, n_jitter: int = 8, n_random: int = 16,
                 jitter: float = 0.4, max_iter: int = 150, tol_pos: float = 1e-6, tol_rot: float = 1e-5,
                 rng_seed: int = 0, coupled=(), coupled_margin: float = 0.1):
        """coupled: [(joint names, A, b)] linear constraints A.q <= b - coupled_margin on arm joints, as the
        genie KDL plugin enforces them (g2cell_motion.robot.ARM_COUPLED); iterates are projected onto them."""
        self.kin, self.names, self.end = kin, list(names), end_link
        chain = kin.chain(end_link)
        idx = [i for i, j in enumerate(chain) if j.name in self.names]
        if [chain[i].name for i in idx] != self.names:
            raise ValueError(f"{names} are not the ordered joints of the chain to {end_link}")
        self.pre = chain[:idx[0]]                       # root -> first arm joint's parent (waist etc.)
        self.arm = [chain[i] for i in idx]
        self.between = [chain[idx[k] + 1:idx[k + 1]] if k + 1 < len(idx) else chain[idx[k] + 1:]
                        for k in range(len(idx))]       # fixed joints after each arm joint
        for seg in self.between:
            if any(j.type != "fixed" for j in seg):
                raise ValueError("non-arm movable joint inside the arm chain")
        self.lo, self.hi = kin.limits(self.names)
        self.n_jitter, self.n_random, self.jitter = n_jitter, n_random, jitter
        self.max_iter, self.tol_pos, self.tol_rot, self.rng_seed = max_iter, tol_pos, tol_rot, rng_seed
        self.axes = np.array([j.axis for j in self.arm])
        self.T_origin = np.array([j.T for j in self.arm])
        self.T_after = []
        for seg in self.between:
            T = np.eye(4)
            for j in seg:
                T = T @ j.T
            self.T_after.append(T)
        self.T_after = np.array(self.T_after)
        self.stats = {"solves": 0, "stage2": 0, "iters": 0, "fail": 0}
        self.coupled = [([self.names.index(n) for n in names], np.asarray(A, float), np.asarray(b, float))
                        for names, A, b in coupled]
        self.margin = coupled_margin

    def feasible(self, sol: dict, tol: float = 1e-9) -> bool:
        q = np.array([sol[n] for n in self.names])
        return bool(self._feasible(q[None], tol)[0])

    def _feasible(self, Q, tol=1e-9):
        ok = np.ones(len(Q), bool)
        for idx, A, b in self.coupled:
            ok &= ((Q[:, idx] @ A.T) <= b - self.margin + tol).all(axis=1)
        return ok

    def _project(self, Q):
        """Push rows back inside the coupled constraints (a few cyclic half-plane projections)."""
        for _ in range(4):
            moved = False
            for idx, A, b in self.coupled:
                for a, bb in zip(A, b):
                    s = Q[:, idx] @ a - (bb - self.margin - 1e-6)
                    v = s > 0
                    if v.any():
                        Q[np.ix_(v, idx)] -= (s[v] / (a @ a))[:, None] * a
                        moved = True
            if not moved:
                break
        return np.clip(Q, self.lo, self.hi)

    def _prefix(self, state: dict):
        T = np.eye(4)
        for j in self.pre:
            T = T @ j.T
            if j.type != "fixed":
                T = T @ j.motion(state.get(j.name, 0.0))
        return T

    def fk(self, Q, prefix):
        """Q (N,7) -> end pos (N,3), end rot (N,3,3), joint origins (N,7,3), joint axes in root (N,7,3)."""
        N = Q.shape[0]
        T = np.broadcast_to(prefix, (N, 4, 4)).copy()
        org = np.empty((N, 7, 3))
        ax = np.empty((N, 7, 3))
        for i in range(len(self.arm)):
            T = T @ self.T_origin[i]
            org[:, i] = T[:, :3, 3]
            a = self.axes[i]
            ax[:, i] = T[:, :3, :3] @ a
            c, s = np.cos(Q[:, i]), np.sin(Q[:, i])
            C = 1 - c
            x, y, z = a
            R = np.zeros((N, 4, 4))
            R[:, 0, 0] = c + x * x * C
            R[:, 0, 1] = x * y * C - z * s
            R[:, 0, 2] = x * z * C + y * s
            R[:, 1, 0] = y * x * C + z * s
            R[:, 1, 1] = c + y * y * C
            R[:, 1, 2] = y * z * C - x * s
            R[:, 2, 0] = z * x * C - y * s
            R[:, 2, 1] = z * y * C + x * s
            R[:, 2, 2] = c + z * z * C
            R[:, 3, 3] = 1.0
            T = T @ R @ self.T_after[i]
        return T[:, :3, 3], T[:, :3, :3], org, ax

    def _run(self, Q, prefix, p_t, R_t):
        """Iterate DLS on all rows; returns (Q, converged mask)."""
        N = len(Q)
        done = np.zeros(N, bool)
        act = np.arange(N)
        for it in range(self.max_iter):
            Qa = Q[act]
            p, R, org, ax = self.fk(Qa, prefix)
            ep = p_t - p
            er = 0.5 * (np.cross(R[:, :, 0], R_t[:, 0]) + np.cross(R[:, :, 1], R_t[:, 1])
                        + np.cross(R[:, :, 2], R_t[:, 2]))
            npos, nrot = np.linalg.norm(ep, axis=1), np.linalg.norm(er, axis=1)
            conv = (npos < self.tol_pos) & (nrot < self.tol_rot)
            if self.coupled:
                conv &= self._feasible(Qa)
            if conv.any():
                done[act[conv]] = True
            keep = ~conv
            if not keep.any():
                break
            act, Qa, ep, er, org, ax, p = act[keep], Qa[keep], ep[keep], er[keep], org[keep], ax[keep], p[keep]
            npos = npos[keep]
            # limit the Cartesian step (keeps DLS stable far from the target)
            sc = np.minimum(1.0, 0.1 / np.maximum(npos, 1e-12))[:, None]
            e = np.concatenate([ep * sc, er * np.minimum(1.0, 0.5 / np.maximum(nrot[keep], 1e-12))[:, None]], axis=1)
            Jp = np.cross(ax, p[:, None, :] - org).transpose(0, 2, 1)
            Jw = ax.transpose(0, 2, 1)
            J = np.concatenate([Jp, Jw], axis=1)                        # (n, 6, 7)
            lam2 = np.where(npos > 1e-3, 1e-2, 1e-6)[:, None, None]
            JJt = J @ J.transpose(0, 2, 1) + lam2 * np.eye(6)[None]
            dq = (J.transpose(0, 2, 1) @ np.linalg.solve(JJt, e[:, :, None]))[:, :, 0]
            Qn = np.clip(Qa + dq, self.lo, self.hi)
            Q[act] = self._project(Qn) if self.coupled else Qn
            self.stats["iters"] += 1
        return Q, done

    def solve(self, pos, quat, state: dict, seeds) -> dict | None:
        """pos/quat: end-link target in root. state: values for the non-arm joints (waist ...).
        seeds: list of {joint: rad} (first = continuity reference)."""
        self.stats["solves"] += 1
        prefix = self._prefix(state)
        p_t = np.asarray(pos, float)
        R_t = quat_to_mat(quat)
        ref = np.array([seeds[0].get(n, 0.0) for n in self.names]) if seeds else np.zeros(len(self.names))
        S = np.clip(np.array([[s.get(n, 0.0) for n in self.names] for s in seeds] or [ref]), self.lo, self.hi)
        if self.coupled:
            S = self._project(S)
        Q, ok = self._run(S.copy(), prefix, p_t, R_t)
        if ok[0]:
            return dict(zip(self.names, Q[0].tolist()))
        if not ok.any():
            self.stats["stage2"] += 1
            rng = np.random.default_rng(self.rng_seed)
            J = ref + rng.uniform(-self.jitter, self.jitter, size=(self.n_jitter, len(self.names)))
            U = rng.uniform(np.maximum(self.lo, -math.pi), np.minimum(self.hi, math.pi),
                            size=(self.n_random, len(self.names)))
            S2 = np.clip(np.vstack([J, U]), self.lo, self.hi)
            Q2, ok2 = self._run(self._project(S2) if self.coupled else S2, prefix, p_t, R_t)
            Q, ok = np.vstack([Q, Q2]), np.concatenate([ok, ok2])
        if not ok.any():
            self.stats["fail"] += 1
            return None
        cand = np.flatnonzero(ok)
        d = ((Q[cand] - ref) ** 2).sum(1)
        best = cand[int(np.argmin(d))]          # argmin: first index on ties -> deterministic
        return dict(zip(self.names, Q[best].tolist()))
