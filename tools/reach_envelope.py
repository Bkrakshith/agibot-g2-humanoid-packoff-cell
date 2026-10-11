"""Offline reach envelope of the AgiBot G2 (crs arm, omnipicker) arms over the conveyor.

Method
------
* Model: URDF from the genie_sim_robot_model xacro (robot_model:=genie arm:=crs gripper:=omnipicker),
  parsed with pinocchio. Its joint origins/limits are identical to the cached
  genie_g2_crs_omnipicker.urdf that move_group loads.
* Chains as MoveIt uses them (genie.srdf.xacro): simple_arm_l / simple_arm_r = the 7 arm joints from
  arm_base_link to arm_{l,r}_end_link, URDF limits. Body/head joints are NOT in the IK; they are held
  at the scene init pose (idx03_body_joint3 = 20 deg, others 0; waist idx05 = 0 unless stated).
* Frames: cell frame = G2 base_link projected on the floor; base_link sits at z = 0.04
  (scene_g2cell.yaml init_base_pose), so z_base = z_cell - 0.04.
* Tool point = fingertip centre = arm_*_end_link origin + 0.125 m along the end link +Z (the gripper
  axis), exactly g2cell_motion.geometry.TOOL_OFFSET / end_link_for_tool used by reach_check and the demo.
* IK: batched damped-least-squares (numpy, FK/Jacobian built from the pinocchio joint placements and
  cross-checked against pinocchio FK), joint limits by clamping, 24 random restarts per target
  (plus the init arm pose as a seed), 250 iterations, success = tool point within 3 mm and the
  orientation criterion met. Every success is re-verified with pinocchio FK.
* Criteria per target: POS = tool point reached in any orientation; TOPDOWN = gripper axis
  (end link +Z) within 15 deg of straight down, any yaw (the demo's down_quat family). Calibration
  additionally reruns the exact MoveIt reach_check orientation set (8 yaws x tilts 0/+-20/40, 6-D IK).
* Kinematics only: no self- or environment-collision checking (reach_check used avoid_collisions=False).

Rerun:  /home/rakshith/ntu_dev/Agibot_dev/.venv-reach/bin/python tools/reach_envelope.py [--urdf PATH]
        (from g2_packoff_cell/; ~ a few minutes on 32 cores). Outputs go to output/reach/.
Waist:  ... tools/reach_envelope.py --waist-yaw [-100,-90:90:15,100]  -> g2_reach_envelope_waist.png,
        reach_summary_waist.json, reach_grid_waist.npz: top-down reach per waist yaw (idx05_body_joint5),
        union over yaws, smallest working yaw per point, the waist_policy yaw checked, Layout v2 U-cell
        (table left, rack right), table spawn zone + rack tote-opening coverage with/without the waist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
DEFAULT_URDF = Path("/home/rakshith/ntu_dev/Agibot_dev/g2_viz_ws/g2_crs_omnipicker.urdf")
BASE_STL = Path("/home/rakshith/ntu_dev/Agibot_dev/g2_viz_ws/install/genie_sim_robot_model/share/"
                "genie_sim_robot_model/robots/genie/g2/meshes/G2/base_link.STL")
OUT = CELL / "output" / "reach"

TOOL_OFFSET = 0.125
BASE_Z = 0.04
BODY_INIT = {"idx03_body_joint3": math.radians(20.0)}
ARM_INIT = [0.0, math.radians(-45), 0.0, math.radians(-75), 0.0, math.radians(-55), 0.0]
ARMS = {"left": ("idx2", "arm_l_end_link"), "right": ("idx6", "arm_r_end_link")}
TOPDOWN_DEG = 15.0
POS_TOL = 0.003
N_RESTART = 24
N_ITER = 250

# grid (cell frame)
STEP = 0.02
XS = np.round(np.arange(-0.30, 1.20 + 1e-9, STEP), 4)
YS = np.round(np.arange(-1.20, 1.20 + 1e-9, STEP), 4)
ZS_TOP = [0.95, 1.07, 1.17]
ZS_SIDE = np.round(np.arange(0.60, 1.60 + 1e-9, STEP), 4)

# cell layout (cell frame)
CARTON = dict(cx=0.60, cy=0.0, lx=0.40, ly=0.60, z0=0.82, z1=1.07)
BELT_TOP = 0.82
CONV_X = (0.30, 0.90)
CONV_Y = (-1.8, 1.8)
TABLE = dict(cx=-0.05, cy=0.85, lx=0.60, ly=1.00, top=0.87)

CALIB = [  # name, tool point (cell), MoveIt result at waist 0 (left, right) from output/stage2/reach_check
    ("spawn near-inner", (0.20, 0.40, 0.99), True, False),
    ("spawn far-inner", (-0.30, 0.40, 0.99), True, False),
    ("spawn centre", (-0.05, 0.85, 0.99), True, False),
    ("spawn near-outer", (0.20, 1.30, 0.99), False, False),
    ("spawn far-outer", (-0.30, 1.30, 0.99), False, False),
    ("above pack crate", (0.60, 0.00, 1.144), True, True),
]


# ----------------------------------------------------------------------------- kinematics
class Chain:
    """7-R arm chain in base_link with body joints fixed. All arm joints rotate about local Z."""

    def __init__(self, urdf: str, arm: str, waist: float = 0.0):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.createData()
        prefix, end = ARMS[arm]
        m = self.model
        self.jids = [m.getJointId(n) for n in m.names if n.startswith(prefix) and "_arm_" in n]
        assert len(self.jids) == 7, self.jids
        self.qidx = [m.joints[j].idx_q for j in self.jids]
        self.lo = m.lowerPositionLimit[self.qidx].copy()
        self.hi = m.upperPositionLimit[self.qidx].copy()
        self.q0 = pin.neutral(m)
        for n, v in {**BODY_INIT, "idx05_body_joint5": waist}.items():
            self.q0[m.joints[m.getJointId(n)].idx_q] = v
        pin.forwardKinematics(m, self.data, self.q0)
        parent = m.parents[self.jids[0]]
        self.T_base = self.data.oMi[parent].homogeneous.copy()
        self.T_j = [m.jointPlacements[j].homogeneous.copy() for j in self.jids]
        for j in self.jids:
            assert m.joints[j].shortname() == "JointModelRZ"
        self.fid = m.getFrameId(end)
        f = m.frames[self.fid]
        assert f.parentJoint == self.jids[-1]
        self.T_end = f.placement.homogeneous.copy()

    def full_q(self, qa):
        q = self.q0.copy()
        q[self.qidx] = qa
        return q

    def pin_tool(self, qa):
        pin = self.pin
        pin.framesForwardKinematics(self.model, self.data, self.full_q(qa))
        M = self.data.oMf[self.fid]
        return M.translation + TOOL_OFFSET * M.rotation[:, 2], M.rotation.copy()

    def fk(self, Q):
        """Q (N,7) -> tool pos (N,3), end rotation (N,3,3), joint origins (N,7,3), joint axes (N,7,3)."""
        N = Q.shape[0]
        T = np.broadcast_to(self.T_base, (N, 4, 4)).copy()
        org = np.empty((N, 7, 3))
        ax = np.empty((N, 7, 3))
        for i in range(7):
            T = T @ self.T_j[i]
            org[:, i] = T[:, :3, 3]
            ax[:, i] = T[:, :3, 2]
            c, s = np.cos(Q[:, i]), np.sin(Q[:, i])
            Rz = np.zeros((N, 4, 4))
            Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
            Rz[:, 2, 2] = 1; Rz[:, 3, 3] = 1
            T = T @ Rz
        T = T @ self.T_end
        R = T[:, :3, :3]
        p = T[:, :3, 3] + TOOL_OFFSET * R[:, :, 2]
        return p, R, org, ax


def solve(chain: Chain, targets, mode: str, cone_deg=TOPDOWN_DEG, Rt=None, n_restart=N_RESTART,
          n_iter=N_ITER, seed=0):
    """Batched DLS IK. targets (M,3) tool points in base_link. mode: 'pos' | 'cone' | 'full'
    (Rt (M,3,3) target end-link rotation). Returns ok (M,), q (M,7)."""
    rng = np.random.default_rng(seed)
    M = len(targets)
    K = n_restart + 1
    P = np.repeat(targets, K, axis=0)
    Q = rng.uniform(chain.lo, chain.hi, size=(M * K, 7))
    Q[::K] = np.clip(ARM_INIT, chain.lo, chain.hi)
    if mode == "full":
        RT = np.repeat(Rt, K, axis=0)
    down = np.array([0.0, 0.0, -1.0])
    cone = math.radians(cone_deg)
    lam = 0.05
    wr = 0.5  # m per rad weighting of orientation rows
    done = np.zeros(M * K, bool)
    for it in range(n_iter):
        act = ~done
        if not act.any():
            break
        Qa = Q[act]
        p, R, org, ax = chain.fk(Qa)
        ep = P[act] - p
        n = np.linalg.norm(ep, axis=1, keepdims=True)
        ep = ep * np.minimum(1.0, 0.08 / np.maximum(n, 1e-9))
        Jp = np.cross(ax, (p[:, None, :] - org)).transpose(0, 2, 1)       # (N,3,7)
        Jw = ax.transpose(0, 2, 1)                                         # (N,3,7)
        if mode == "pos":
            J, e = Jp, ep
            ok_rot = np.ones(len(Qa), bool)
        elif mode == "cone":
            a = R[:, :, 2]
            cr = np.cross(a, down)
            sn = np.linalg.norm(cr, axis=1)
            th = np.arctan2(sn, (a * down).sum(1))
            axis = cr / np.maximum(sn, 1e-9)[:, None]
            er = axis * np.maximum(0.0, th - cone + math.radians(1.0))[:, None]
            Pp = np.eye(3)[None] - a[:, :, None] * a[:, None, :]
            J = np.concatenate([Jp, wr * (Pp @ Jw)], axis=1)
            e = np.concatenate([ep, wr * er], axis=1)
            ok_rot = th <= cone + 1e-6
        else:
            Rerr = RT[act] @ R.transpose(0, 2, 1)
            from scipy.spatial.transform import Rotation
            er = Rotation.from_matrix(Rerr).as_rotvec()
            J = np.concatenate([Jp, wr * Jw], axis=1)
            e = np.concatenate([ep, wr * er], axis=1)
            ok_rot = np.linalg.norm(er, axis=1) < math.radians(0.5)
        ok = (np.linalg.norm(P[act] - p, axis=1) < POS_TOL) & ok_rot
        idx = np.flatnonzero(act)
        done[idx[ok]] = True
        JJt = J @ J.transpose(0, 2, 1) + (lam ** 2) * np.eye(J.shape[1])[None]
        dq = (J.transpose(0, 2, 1) @ np.linalg.solve(JJt, e[:, :, None]))[:, :, 0]
        dq = np.clip(dq, -0.3, 0.3)
        # small random kick every 60 iterations to escape limit stalls
        if it % 60 == 59:
            dq += rng.normal(0, 0.15, dq.shape)
        Qn = np.clip(Qa + dq, chain.lo, chain.hi)
        Qn[ok] = Qa[ok]
        Q[act] = Qn
    done_r = done.reshape(M, K)
    okM = done_r.any(1)
    first = done_r.argmax(1)
    qsol = Q.reshape(M, K, 7)[np.arange(M), first]
    return okM, qsol


def verify(chain: Chain, targets, ok, qsol, mode, cone_deg=TOPDOWN_DEG, Rt=None):
    """Re-check every success with pinocchio FK; returns the verified mask and the max errors."""
    ok = ok.copy()
    maxe, maxth = 0.0, 0.0
    for i in np.flatnonzero(ok):
        q = qsol[i]
        assert np.all(q >= chain.lo - 1e-9) and np.all(q <= chain.hi + 1e-9)
        p, R = chain.pin_tool(q)
        e = np.linalg.norm(p - targets[i])
        th = math.degrees(math.acos(np.clip(-R[2, 2], -1, 1)))
        good = e < POS_TOL + 1e-6
        if mode == "cone":
            good &= th <= cone_deg + 1e-3
        if mode == "full":
            from scipy.spatial.transform import Rotation
            good &= np.linalg.norm(Rotation.from_matrix(Rt[i] @ R.T).as_rotvec()) < math.radians(0.6)
        ok[i] = good
        maxe = max(maxe, e)
        maxth = max(maxth, th) if mode == "cone" else maxth
    return ok, maxe, maxth


def job(args):
    urdf, arm, mode, pts_cell, seed = args
    ch = Chain(urdf, arm)
    tgt = np.asarray(pts_cell, float) - np.array([0, 0, BASE_Z])
    ok, q = solve(ch, tgt, mode, seed=seed)
    okv, maxe, maxth = verify(ch, tgt, ok, q, mode)
    return ok, okv, maxe, maxth


def run_grid(pool, urdf, pts, chunk=600):
    """Returns {(arm,mode): bool array} over pts (cell frame)."""
    futs = {}
    for arm in ARMS:
        for mode in ("pos", "cone"):
            for k in range(0, len(pts), chunk):
                futs[(arm, mode, k)] = pool.submit(job, (urdf, arm, mode, pts[k:k + chunk], k + 7))
    res = {}
    stats = {"unverified": 0, "max_pos_err_m": 0.0, "max_tilt_deg": 0.0}
    for (arm, mode, k), f in futs.items():
        ok, okv, maxe, maxth = f.result()
        stats["unverified"] += int(ok.sum() - okv.sum())
        stats["max_pos_err_m"] = max(stats["max_pos_err_m"], maxe)
        stats["max_tilt_deg"] = max(stats["max_tilt_deg"], maxth)
        res.setdefault((arm, mode), [None] * ((len(pts) + chunk - 1) // chunk))[k // chunk] = okv
    return {k: np.concatenate(v) for k, v in res.items()}, stats


# ----------------------------------------------------------------------------- calibration
def down_R(yaw, tilt):
    """Rotation matrix of g2cell_motion.geometry.down_quat(yaw, tilt)."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_rotvec([math.pi, 0, 0])
    if tilt:
        q = Rot.from_rotvec([0, -tilt, 0]) * q
    return (Rot.from_rotvec([0, 0, yaw]) * q).as_matrix()


def calibrate(urdf):
    yaws = [0, 45, 90, -45, -90, 135, -135, 180]
    tilts = [0, 20, -20, 40]
    rows = []
    for name, tool, ml, mr in CALIB:
        row = {"target": name, "tool_cell": tool, "moveit_left": ml, "moveit_right": mr}
        for arm in ARMS:
            ch = Chain(urdf, arm)
            t = np.array(tool) - [0, 0, BASE_Z]
            # exact MoveIt orientation set (6-D IK)
            Rs = [down_R(math.radians(y), math.radians(ti)) for ti in tilts for y in yaws]
            labels = [(y, ti) for ti in tilts for y in yaws]
            tg = np.repeat(t[None], len(Rs), 0)
            ok, q = solve(ch, tg, "full", Rt=np.array(Rs), n_restart=32)
            ok, _, _ = verify(ch, tg, ok, q, "full", Rt=np.array(Rs))
            row[f"{arm}_moveit_set"] = bool(ok.any())
            row[f"{arm}_moveit_set_first"] = labels[int(np.argmax(ok))] if ok.any() else None
            for mode in ("cone", "pos"):
                okm, qm = solve(ch, t[None], mode, n_restart=64)
                okm, _, _ = verify(ch, t[None], okm, qm, mode)
                row[f"{arm}_{mode}"] = bool(okm[0])
            # waist turned toward the target (reach_check: right arm needed waist -60 for the table)
            w = math.radians(-60 if arm == "right" else 0)
            chw = Chain(urdf, arm, waist=w)
            okw, qw = solve(chw, tg, "full", Rt=np.array(Rs), n_restart=32)
            okw, _, _ = verify(chw, tg, okw, qw, "full", Rt=np.array(Rs))
            row[f"{arm}_moveit_set_waist{int(round(math.degrees(w)))}"] = bool(okw.any())
        rows.append(row)
    return rows


def fk_crosscheck(urdf, n=200):
    err = 0.0
    for arm in ARMS:
        ch = Chain(urdf, arm)
        Q = np.random.default_rng(1).uniform(ch.lo, ch.hi, (n, 7))
        p, R, _, _ = ch.fk(Q)
        for i in range(n):
            pp, RR = ch.pin_tool(Q[i])
            err = max(err, np.abs(pp - p[i]).max(), np.abs(RR - R[i]).max())
    return err


# ----------------------------------------------------------------------------- plotting
def chassis_hull():
    from scipy.spatial import ConvexHull
    b = BASE_STL.read_bytes()
    n = int(np.frombuffer(b[80:84], np.uint32)[0])
    d = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = d["v"].reshape(-1, 3)[:, :2].astype(float)
    h = ConvexHull(v)
    return v[h.vertices], v.min(0), v.max(0)


C_L, C_R, C_B = "#0072B2", "#E69F00", "#009E73"   # Okabe-Ito blue / orange / bluish green


def plot(path, data, hull):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Polygon, Rectangle
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})
    cmap = ListedColormap([C_L, C_R, C_B])
    fig = plt.figure(figsize=(24, 13.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.12)
    ext = [XS[0] - STEP / 2, XS[-1] + STEP / 2, YS[0] - STEP / 2, YS[-1] + STEP / 2]

    def cat_img(L, R):
        c = np.full(L.shape, np.nan)
        c[L & ~R] = 0; c[R & ~L] = 1; c[L & R] = 2
        return c

    def overlay_pos(ax, X, Y, Lp, Rp):
        for M, col in ((Lp, C_L), (Rp, C_R)):
            ax.contourf(X, Y, M.astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.13)
            ax.contour(X, Y, M.astype(float), levels=[0.5], colors=[col], linewidths=2.2, linestyles="--")

    for k, z in enumerate(ZS_TOP):
        ax = fig.add_subplot(gs[0, k])
        L, R = data[f"top_{z}"]["left_cone"], data[f"top_{z}"]["right_cone"]
        Lp, Rp = data[f"top_{z}"]["left_pos"], data[f"top_{z}"]["right_pos"]
        # arrays are (nx, ny); plot with robot +X up the page? Use plan view: x horizontal = cell Y
        # (robot's left on the LEFT when looking from behind the robot toward the conveyor => -Y right).
        Yg, Xg = np.meshgrid(YS, XS)
        overlay_pos(ax, -Yg, Xg, Lp, Rp)
        img = cat_img(L, R)
        ax.pcolormesh(-Yg, Xg, np.ma.masked_invalid(img), cmap=cmap, vmin=-0.5, vmax=2.5,
                      shading="nearest", alpha=0.85, rasterized=True)
        draw_layout_top(ax, hull, Rectangle, Polygon)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 1.2)
        ax.set_aspect("equal")
        lab = {0.95: "inside carton (0.13 above belt)", 1.07: "carton rim", 1.17: "0.10 m above rim"}[z]
        ax.set_title(f"Top view, fingertips at Z = {z:.2f} m\n{lab}")
        ax.set_xlabel("cell Y [m]   (robot's left  <-  -> robot's right)")
        ticks = np.arange(-1.0, 1.01, 0.5)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{-t:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0") for t in ticks])
        if k == 0:
            ax.set_ylabel("cell X [m]  (toward conveyor ->)")
        else:
            ax.set_yticklabels([])
        ax.grid(alpha=0.25)
        cov = data["coverage"][f"{z}"]
        ax.text(0.02, 0.02, f"carton opening, top-down:\nL {cov['left_cone']*100:.0f}%  R {cov['right_cone']*100:.0f}%"
                f"  either {cov['either_cone']*100:.0f}%", transform=ax.transAxes, fontsize=13,
                va="bottom", bbox=dict(fc="white", ec="0.6", alpha=0.9))

    # side view
    ax = fig.add_subplot(gs[1, :2])
    s = data["side"]
    Zg, Xg = np.meshgrid(ZS_SIDE, XS)
    overlay_pos(ax, Xg, Zg, s["left_pos"], s["right_pos"])
    img = cat_img(s["left_cone"], s["right_cone"])
    ax.pcolormesh(Xg, Zg, np.ma.masked_invalid(img), cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest",
                  alpha=0.85, rasterized=True)
    ax.add_patch(Rectangle((CONV_X[0], BELT_TOP - 0.06), CONV_X[1] - CONV_X[0], 0.06, fc="0.55", ec="k", lw=1.5))
    ax.plot(CONV_X, [BELT_TOP] * 2, color="k", lw=3)
    c = CARTON
    x0, x1 = c["cx"] - c["lx"] / 2, c["cx"] + c["lx"] / 2
    ax.plot([x0, x0, x1, x1], [c["z1"], c["z0"], c["z0"], c["z1"]], color="#6b3e1d", lw=3.5)
    ax.text(c["cx"], c["z1"] + 0.015, "open carton", ha="center", va="bottom", fontsize=13, color="#6b3e1d")
    ax.text(CONV_X[1] + 0.02, BELT_TOP - 0.03, "belt 0.82", va="center", fontsize=13)
    ax.set_xlim(XS[0] - 0.01, XS[-1] + 0.01); ax.set_ylim(0.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("cell X [m]  (toward conveyor ->)"); ax.set_ylabel("Z [m]")
    ax.set_title("Side view, slice at Y = 0 (pack position)")
    ax.grid(alpha=0.25)
    for z in ZS_TOP:
        ax.axhline(z, color="0.4", lw=1, ls=":")
        ax.text(XS[-1] - 0.005, z + 0.008, f"Z {z:.2f}", ha="right", fontsize=11, color="0.3")

    # legend panel
    lax = fig.add_subplot(gs[1, 2]); lax.axis("off")
    hs = [Patch(fc=C_L, alpha=0.85, label="left arm only"),
          Patch(fc=C_R, alpha=0.85, label="right arm only"),
          Patch(fc=C_B, alpha=0.85, label="both arms"),
          Line2D([], [], color="0.3", lw=0, label=f"solid = top-down grasp (axis within {TOPDOWN_DEG:.0f} deg of vertical, any yaw)"),
          Patch(fc=C_L, alpha=0.18, ec=C_L, ls="--", lw=2, label="left, position only (any orientation)"),
          Patch(fc=C_R, alpha=0.18, ec=C_R, ls="--", lw=2, label="right, position only (any orientation)"),
          Patch(fc="none", ec="#6b3e1d", lw=3, label="open carton 0.6 x 0.4 x 0.25 m at pack pos."),
          Patch(fc="0.75", ec="k", label="conveyor frame / belt (X 0.30-0.90)"),
          Patch(fc="#f2e6cf", ec="#8a6d3b", label="pick table (top 0.87)"),
          Patch(fc="0.85", ec="0.3", hatch="xx", label="G2 chassis footprint (URDF mesh)"),
          Patch(fc="none", ec="0.35", ls="--", label="sort rack (planned, approximate)")]
    lax.legend(handles=hs, loc="upper left", frameon=False, fontsize=13.5, handlelength=2.2,
               title="Fingertip point = arm_*_end_link + 0.125 m", title_fontsize=14, alignment="left")
    lax.text(0.0, -0.06, "Kinematics only (no collision check). Body held at scene init:\n"
             "body_joint3 = 20 deg, waist 0. base_link 0.04 m above floor.\n"
             "Pinocchio + DLS IK, 24 random restarts/target, URDF joint limits.",
             transform=lax.transAxes, fontsize=12, va="top", color="0.25")
    fig.suptitle("AgiBot G2 (crs arms, omnipicker): reach envelope over the conveyor, cell frame", fontsize=22, y=0.955)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def draw_layout_top(ax, hull, Rectangle, Polygon):
    """Plan view with page-x = -Y (cell), page-y = X (cell)."""
    def rect(x0, x1, y0, y1, **kw):
        ax.add_patch(Rectangle((-y1, x0), y1 - y0, x1 - x0, **kw))
    rect(CONV_X[0], CONV_X[1], CONV_Y[0], CONV_Y[1], fc="0.75", ec="k", lw=1.5, alpha=0.35, zorder=0)
    ax.plot([-1.25, 1.25], [0.60, 0.60], color="0.35", lw=1, ls="-.", zorder=1)
    t = TABLE
    rect(t["cx"] - t["lx"] / 2, t["cx"] + t["lx"] / 2, t["cy"] - t["ly"] / 2, t["cy"] + t["ly"] / 2,
         fc="#f2e6cf", ec="#8a6d3b", lw=2, alpha=0.55, zorder=0)
    ax.text(-t["cy"], t["cx"] - 0.25, "pick table", ha="center", fontsize=12, color="#5a4520")
    c = CARTON
    rect(c["cx"] - c["lx"] / 2, c["cx"] + c["lx"] / 2, c["cy"] - c["ly"] / 2, c["cy"] + c["ly"] / 2,
         fc="none", ec="#6b3e1d", lw=3, zorder=5)
    # sort rack: planned, approximate (moving to the robot's right, facing +Y)
    rect(-0.45, 0.25, -1.25, -0.80, fc="none", ec="0.35", lw=1.8, ls="--", zorder=4)
    ax.text(-(-1.0), -0.45 - 0.0, "", fontsize=1)
    ax.text(1.02, -0.10, "sort rack\n(planned)", ha="center", va="center", fontsize=12, color="0.3", zorder=6)
    hv, _, _ = hull
    ax.add_patch(Polygon(np.c_[-hv[:, 1], hv[:, 0]], closed=True, fc="0.85", ec="0.3", lw=1.5, hatch="xx",
                         alpha=0.9, zorder=3))
    ax.annotate("", xy=(0, 0.22), xytext=(0, 0.0), arrowprops=dict(arrowstyle="->", lw=2), zorder=6)
    ax.text(0.03, 0.10, "+X", fontsize=12, zorder=6)
    ax.text(-1.17, 0.62, "conveyor centreline", fontsize=11, va="bottom", color="0.3")


# ----------------------------------------------------------------------------- waist sweep
# --waist-yaw: the same top-down (cone) reach, but with the waist (idx05_body_joint5) turned. Reach with the
# waist = union over the swept yaws; the smallest |yaw| that works is recorded per point, and the yaw that
# g2cell_motion.waist_policy would choose (nearest swept yaw) is checked as well.
W_STEP = 0.03
W_XS = np.round(np.arange(-1.14, 1.20 + 1e-9, W_STEP), 4)
W_YS = np.round(np.arange(-1.32, 1.50 + 1e-9, W_STEP), 4)
W_ZS = [0.95, 1.07, 1.17, 0.97]            # 0.97 = pick-table top 0.87 + 0.10
SHOULDER_REACH = 0.84                      # max |tool - arm joint2 origin| over the joint box is 0.822 m
DEFAULT_WAIST = "-100,-90:90:15,100"
# Layout v2 (assets/cell/build.py LAYOUT, 2026-10-11; assets/cell/README.md has no "Layout v2" section yet):
# U opening to -X: pick table left (+Y), conveyor front (+X), sort rack right (-Y) with its open front facing +Y.
V2_TABLE = dict(cx=-0.05, cy=0.93, lx=0.60, ly=1.00, top=0.87)
V2_SPAWN = dict(x0=-0.30, x1=0.20, y0=0.48, y1=1.38)     # 0.50 x 0.90 spawn zone centred on the table
V2_RACK = dict(x=-0.41, y=-0.82, yaw_deg=90.0, depth=0.65, width=1.35)
# sort_rack/build.py geometry (local frame: open front +X, lanes along local Y)
_R = dict(profile=0.040, label_t=0.003, stop_t=0.030, stop_above=0.040, lips=(0.75, 1.20), tilt=20.0,
          tote=(0.600, 0.400, 0.325), backoff=0.014)
RACK_DROP_ABOVE = 0.10                     # fingertips this far above the tote opening plane
RACK_INSET = 0.05                          # keep sample points this far inside the tote walls


def parse_yaws(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            a, b, s = (float(v) for v in part.split(":"))
            out += list(np.arange(a, b + 1e-9, s))
        elif part:
            out.append(float(part))
    return sorted(set(round(v, 6) for v in out))


def rack_world(lx, ly, z):
    r = V2_RACK
    c, s = math.cos(math.radians(r["yaw_deg"])), math.sin(math.radians(r["yaw_deg"]))
    return (r["x"] + c * lx - s * ly, r["y"] + s * lx + c * ly, z)


def rack_openings():
    """{lane: (corners (4,3) of the tote opening, sample points (N,3) RACK_DROP_ABOVE over it)}, cell frame."""
    t = math.radians(_R["tilt"])
    hx, hy = V2_RACK["depth"] / 2, V2_RACK["width"] / 2
    deck_xf = hx - _R["label_t"] - _R["stop_t"]
    lane_w = (V2_RACK["width"] - 3 * _R["profile"]) / 2
    lane_y = [-hy + _R["profile"] + lane_w / 2 + i * (lane_w + _R["profile"]) for i in range(2)]
    ex = np.array([math.cos(t), 0.0, -math.sin(t)])
    ez = np.array([math.sin(t), 0.0, math.cos(t)])
    L, W, H = _R["tote"]
    out = {}
    for li, lip in enumerate(_R["lips"], start=1):
        zf = lip - _R["stop_above"]
        for ni, ly in enumerate(lane_y, start=1):
            front_top = np.array([deck_xf, ly, zf]) - ex * _R["backoff"] + ez * H
            def P(u, v):
                q = front_top - ex * u + np.array([0, v, 0])
                return rack_world(q[0], q[1], q[2])
            corners = np.array([P(0, -W / 2), P(0, W / 2), P(L, W / 2), P(L, -W / 2)])
            pts = [np.array(P(u, v)) + [0, 0, RACK_DROP_ABOVE]
                   for u in np.linspace(RACK_INSET, L - RACK_INSET, 4)
                   for v in np.linspace(-W / 2 + RACK_INSET, W / 2 - RACK_INSET, 3)]
            out[f"L{li}_{ni}"] = (corners, np.array(pts))
    return out


def job_w(args):
    """Top-down reach of pts (cell frame) for one arm at one waist yaw; far points are skipped."""
    urdf, arm, yaw_deg, pts_cell, seed = args
    ch = Chain(urdf, arm, waist=math.radians(yaw_deg))
    tgt = np.asarray(pts_cell, float) - np.array([0, 0, BASE_Z])
    _, _, org, _ = ch.fk(np.array([ARM_INIT]))
    near = np.linalg.norm(tgt - org[0, 1], axis=1) <= SHOULDER_REACH
    res = np.zeros(len(tgt), bool)
    if near.any():
        ok, q = solve(ch, tgt[near], "cone", seed=seed)
        okv, _, _ = verify(ch, tgt[near], ok, q, "cone")
        res[near] = okv
    return res


def _policy_yaws(pts, yaws, fallbacks=False):
    """Nearest swept yaw to what g2cell_motion.waist_policy picks from waist 0, per point (deg).
    fallbacks=True: (N, nyaw) mask of the swept yaws nearest to the planner's whole candidate list
    (policy yaw, then 30 deg steps back to 0, one past, 0), i.e. what motion_server would try."""
    sys.path.insert(0, str(CELL / "ros_ws" / "src" / "g2cell_motion"))
    from g2cell_motion.waist_policy import WaistConfig, choose_waist_yaw, waist_candidates
    cfg = WaistConfig()
    ya = np.array(yaws)
    near = lambda r: int(np.argmin(np.abs(ya - math.degrees(r))))   # noqa: E731
    out = np.zeros((len(pts), len(ya)), bool) if fallbacks else np.empty(len(pts))
    for i, p in enumerate(pts):
        d = choose_waist_yaw((p[0], p[1]), 0.0, cfg)
        if fallbacks:
            for c in waist_candidates(d.yaw, 0.0, cfg, step_deg=30.0):
                out[i, near(c)] = True
        else:
            out[i] = ya[near(d.yaw)]
    return out


def best_yaw(reach, yaws, bear_sign):
    """reach (nyaw, ...) bool -> smallest-|yaw| reachable (deg), ties toward the target side; NaN if none."""
    ya = np.array(yaws)
    out = np.full(reach.shape[1:], np.nan)
    # write larger |yaw| first so the smallest reachable |yaw| wins; at equal |yaw| the target's side last
    for k in sorted(range(len(ya)), key=lambda k: (-abs(ya[k]), 0)):
        out[reach[k]] = ya[k]
    for k in range(len(ya)):
        m = reach[k] & (np.sign(ya[k]) == bear_sign) & (np.abs(out) == abs(ya[k]))
        out[m] = ya[k]
    return out


def run_waist_sweep(urdf, workers, yaws, chunk=500):
    Xg, Yg = np.meshgrid(W_XS, W_YS, indexing="ij")
    openings = rack_openings()
    rack_pts = np.concatenate([p for _, p in openings.values()])
    t0 = time.time()
    futs = {}
    with ProcessPoolExecutor(workers) as pool:
        for z in W_ZS:
            pts = np.c_[Xg.ravel(), Yg.ravel(), np.full(Xg.size, z)]
            for arm in ARMS:
                for yi, y in enumerate(yaws):
                    for k in range(0, len(pts), chunk):
                        futs[("g", z, arm, yi, k)] = pool.submit(job_w, (urdf, arm, y, pts[k:k + chunk], k + yi * 7 + 3))
        for arm in ARMS:
            for yi, y in enumerate(yaws):
                futs[("r", 0, arm, yi, 0)] = pool.submit(job_w, (urdf, arm, y, rack_pts, 11 + yi))
        grids = {(z, arm): np.zeros((len(yaws), Xg.size), bool) for z in W_ZS for arm in ARMS}
        rack = {arm: np.zeros((len(yaws), len(rack_pts)), bool) for arm in ARMS}
        for i, (key, f) in enumerate(futs.items()):
            kind, z, arm, yi, k = key
            r = f.result()
            if kind == "g":
                grids[(z, arm)][yi, k:k + len(r)] = r
            else:
                rack[arm][yi] = r
            if i % 200 == 0:
                print(f"  {i}/{len(futs)} jobs, {time.time() - t0:.0f}s", flush=True)
    grids = {k: v.reshape((len(yaws),) + Xg.shape) for k, v in grids.items()}
    return grids, rack, openings, rack_pts


def summarise_waist(grids, rack, openings, rack_pts, yaws):
    ya = np.array(yaws)
    i0 = int(np.argmin(np.abs(ya)))
    Xg, Yg = np.meshgrid(W_XS, W_YS, indexing="ij")

    def stats(R, pol_idx, fb=None):
        """R: {arm: (nyaw, N)} -> coverage fractions waist0 / union / policy (/ planner with fallbacks)."""
        out = {}
        N = next(iter(R.values())).shape[1]
        for name, M in list(R.items()) + [("either", R["left"] | R["right"])]:
            out[name] = {"waist0": round(float(M[i0].mean()), 3),
                         "with_waist_union": round(float(M.any(0).mean()), 3),
                         "policy_yaw": round(float(M[pol_idx, np.arange(N)].mean()), 3)}
            if fb is not None:
                out[name]["planner_with_fallbacks"] = round(float((M.T & fb).any(1).mean()), 3)
        return out

    summ = {"yaws_deg": list(map(float, yaws)), "grid_step_m": W_STEP, "criterion": "top-down cone 15 deg",
            "layout": "Layout v2 from assets/cell/build.py LAYOUT (README section not yet written)",
            "table_spawn_zone": {}, "carton_opening": {}, "rack_openings": {}}
    # table spawn zone at Z 0.97
    m = (Xg >= V2_SPAWN["x0"] - 1e-9) & (Xg <= V2_SPAWN["x1"] + 1e-9) & (Yg >= V2_SPAWN["y0"] - 1e-9) & (Yg <= V2_SPAWN["y1"] + 1e-9)
    pts = np.c_[Xg[m], Yg[m]]
    pol = np.searchsorted(ya, _policy_yaws(pts, yaws))
    R = {arm: grids[(0.97, arm)][:, m] for arm in ARMS}
    summ["table_spawn_zone"] = {"z": 0.97, "points": int(m.sum()), **stats(R, pol, _policy_yaws(pts, yaws, True))}
    # carton opening (pack position) at the three carton heights
    c = CARTON
    mc = (np.abs(Xg - c["cx"]) <= c["lx"] / 2 + 1e-9) & (np.abs(Yg - c["cy"]) <= c["ly"] / 2 + 1e-9)
    polc = np.searchsorted(ya, _policy_yaws(np.c_[Xg[mc], Yg[mc]], yaws))
    fbc = _policy_yaws(np.c_[Xg[mc], Yg[mc]], yaws, True)
    for z in (0.95, 1.07, 1.17):
        summ["carton_opening"][f"{z}"] = stats({arm: grids[(z, arm)][:, mc] for arm in ARMS}, polc, fbc)
    # rack tote openings
    k = 0
    polr = np.searchsorted(ya, _policy_yaws(rack_pts, yaws))
    fbr = _policy_yaws(rack_pts, yaws, True)
    for lane, (_, p) in openings.items():
        sl = slice(k, k + len(p))
        summ["rack_openings"][lane] = {"points": len(p), "z_range": [round(float(p[:, 2].min()), 3), round(float(p[:, 2].max()), 3)],
                                       **stats({arm: rack[arm][:, sl] for arm in ARMS}, polr[sl], fbr[sl])}
        Rl = rack["left"][:, sl] | rack["right"][:, sl]
        okyaw = [float(y) for y, r in zip(yaws, Rl.any(1)) if r]
        summ["rack_openings"][lane]["yaws_that_reach_any_point"] = okyaw
        summ["rack_openings"][lane]["policy_yaws_deg"] = sorted(set(float(ya[i]) for i in polr[sl]))
        k += len(p)
    allr = {arm: rack[arm] for arm in ARMS}
    summ["rack_openings"]["all"] = stats(allr, polr, fbr)
    return summ


def plot_waist(path, grids, openings, summ, yaws, hull):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})
    ya = np.array(yaws)
    i0 = int(np.argmin(np.abs(ya)))
    Xg, Yg = np.meshgrid(W_XS, W_YS, indexing="ij")
    bsign = np.sign(np.arctan2(Yg, Xg - 0.258))
    edges = np.concatenate([[ya[0] - 7.5], (ya[1:] + ya[:-1]) / 2, [ya[-1] + 7.5]])
    cmap = plt.get_cmap("PuOr", len(ya))
    norm = BoundaryNorm(edges, cmap.N)
    fig = plt.figure(figsize=(26, 16.5))
    gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.12)
    panels = [(0.95, gs[0, 0], "inside carton (0.13 above belt)"), (1.07, gs[0, 1], "carton rim"),
              (1.17, gs[0, 2], "0.10 m above carton rim"), (0.97, gs[1, 0], "table top 0.87 + 0.10")]
    mesh = None
    for z, g, lab in panels:
        ax = fig.add_subplot(g)
        L, R = grids[(z, "left")], grids[(z, "right")]
        E = L | R
        by = best_yaw(E, yaws, bsign)
        mesh = ax.pcolormesh(-Yg, Xg, np.ma.masked_invalid(by), cmap=cmap, norm=norm, shading="nearest",
                             alpha=0.80, rasterized=True)
        for M, col, ls, lw in ((L[i0], C_L, "-", 2.6), (R[i0], C_R, "-", 2.6), (E.any(0), "k", "--", 2.0)):
            ax.contour(-Yg, Xg, M.astype(float), levels=[0.5], colors=[col], linewidths=lw, linestyles=ls, zorder=7)
        draw_layout_v2(ax, hull, openings, Rectangle, Polygon)
        ax.set_xlim(-1.45, 1.55); ax.set_ylim(-1.18, 1.22)
        ax.set_aspect("equal")
        ax.set_title(f"Fingertips at Z = {z:.2f} m: {lab}")
        ax.set_xlabel("cell Y [m]   (robot's left  <-  -> robot's right)")
        ticks = np.arange(-1.0, 1.51, 0.5)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{-t:.1f}".replace("-0.0", "0.0") for t in ticks])
        ax.set_ylabel("cell X [m]  (toward conveyor ->)")
        ax.grid(alpha=0.25)
        if z == 0.97:
            s = summ["table_spawn_zone"]
            txt = (f"table spawn zone (dotted), top-down:\n"
                   f"waist 0: L {s['left']['waist0']*100:.0f}%  R {s['right']['waist0']*100:.0f}%\n"
                   f"waist swept: L {s['left']['with_waist_union']*100:.0f}%  R {s['right']['with_waist_union']*100:.0f}%\n"
                   f"policy yaw only: L {s['left']['policy_yaw']*100:.0f}%  R {s['right']['policy_yaw']*100:.0f}%\n"
                   f"planner (+fallbacks): L {s['left']['planner_with_fallbacks']*100:.0f}%  "
                   f"R {s['right']['planner_with_fallbacks']*100:.0f}%")
        else:
            s = summ["carton_opening"][f"{z}"]
            txt = (f"carton opening, either arm:\nwaist 0 {s['either']['waist0']*100:.0f}%, "
                   f"swept {s['either']['with_waist_union']*100:.0f}%")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=13, va="bottom",
                bbox=dict(fc="white", ec="0.6", alpha=0.92), zorder=9)
    cb = fig.colorbar(mesh, ax=ax, ticks=ya, fraction=0.05, pad=0.02, shrink=0.95)
    cb.set_label("best waist yaw [deg] (+ = torso left)")
    # legend + rack table
    lax = fig.add_subplot(gs[1, 1:]); lax.axis("off")
    hs = [Line2D([], [], color=C_L, lw=2.6, label="left arm, waist 0 (no turn)"),
          Line2D([], [], color=C_R, lw=2.6, label="right arm, waist 0 (no turn)"),
          Line2D([], [], color="k", lw=2.0, ls="--", label="either arm, waist swept (union)"),
          Patch(fc=cmap(0.5), alpha=0.8, label="fill: smallest |waist yaw| that reaches (colour bar)"),
          Patch(fc="#f2e6cf", ec="#8a6d3b", label="pick table (top 0.87), dotted = spawn zone"),
          Patch(fc="none", ec="0.35", ls="--", label="sort rack footprint (approx.)"),
          Patch(fc="none", ec="#D55E00", lw=1.6, label="tote openings, lanes 1/2 (levels L1, L2 overlap in plan)"),
          Patch(fc="none", ec="#6b3e1d", lw=3, label="open carton at pack position"),
          Patch(fc="0.75", ec="k", label="conveyor (X 0.30-0.90)"),
          Patch(fc="0.85", ec="0.3", hatch="xx", label="G2 chassis footprint")]
    lax.legend(handles=hs, loc="upper left", frameon=False, fontsize=14, bbox_to_anchor=(0.06, 1.0),
               title="Top-down grasp: gripper axis within 15 deg of vertical, any yaw", title_fontsize=14,
               alignment="left")
    rows = ["Rack tote openings (fingertips 0.10 m above the opening), either arm:",
            f"{'lane':6s} {'Z':>11s} {'waist 0':>8s} {'swept':>6s} {'policy':>7s} {'planner':>8s}",
            ]
    for lane, s in summ["rack_openings"].items():
        if lane == "all":
            continue
        e = s["either"]
        rows.append(f"{lane:6s} {s['z_range'][0]:.2f}-{s['z_range'][1]:.2f} {e['waist0']*100:7.0f}% "
                    f"{e['with_waist_union']*100:5.0f}% {e['policy_yaw']*100:6.0f}% "
                    f"{e['planner_with_fallbacks']*100:7.0f}%")
    rows += ["policy = waist_policy yaw only; planner = + motion_server",
             "fallback yaws (30 deg steps back to 0).", "", "Kinematics only (no collision check). Body at scene init (body_joint3 20 deg),",
             f"waist swept {int(ya[0])}..{int(ya[-1])} deg; base fixed at the cell origin.",
             "Layout v2 from assets/cell/build.py (rack outline approximate)."]
    lax.text(0.53, 1.0, "\n".join(rows), transform=lax.transAxes, fontsize=13, va="top", family="monospace")
    fig.suptitle("AgiBot G2: reach with waist-first turning (idx05_body_joint5), Layout v2 U-cell, cell frame",
                 fontsize=22, y=0.95)
    fig.savefig(path, dpi=100, bbox_inches="tight")


def draw_layout_v2(ax, hull, openings, Rectangle, Polygon):
    """Plan view, page-x = -Y, page-y = X (as draw_layout_top)."""
    def rect(x0, x1, y0, y1, **kw):
        ax.add_patch(Rectangle((-y1, x0), y1 - y0, x1 - x0, **kw))
    rect(CONV_X[0], CONV_X[1], -1.8, 1.8, fc="0.75", ec="k", lw=1.5, alpha=0.30, zorder=0)
    t = V2_TABLE
    rect(t["cx"] - t["lx"] / 2, t["cx"] + t["lx"] / 2, t["cy"] - t["ly"] / 2, t["cy"] + t["ly"] / 2,
         fc="#f2e6cf", ec="#8a6d3b", lw=2, alpha=0.45, zorder=1)
    s = V2_SPAWN
    rect(s["x0"], s["x1"], s["y0"], s["y1"], fc="none", ec="#5a4520", lw=2.0, ls=":", zorder=8)
    ax.text(-t["cy"], t["cx"] + t["lx"] / 2 + 0.02, "pick table", ha="center", va="bottom", fontsize=12, color="#5a4520", zorder=8)
    c = CARTON
    rect(c["cx"] - c["lx"] / 2, c["cx"] + c["lx"] / 2, c["cy"] - c["ly"] / 2, c["cy"] + c["ly"] / 2,
         fc="none", ec="#6b3e1d", lw=3, zorder=6)
    r = V2_RACK
    rect(r["x"] - r["width"] / 2, r["x"] + r["width"] / 2, r["y"] - r["depth"] / 2, r["y"] + r["depth"] / 2,
         fc="none", ec="0.35", lw=1.8, ls="--", zorder=6)
    for lane, (cor, _) in openings.items():
        if lane.startswith("L1"):
            ax.add_patch(Polygon(np.c_[-cor[:, 1], cor[:, 0]], closed=True, fc="none", ec="#D55E00", lw=1.6, zorder=8))
            ax.text(-cor[:, 1].mean(), cor[:, 0].mean(), lane[3:], ha="center", va="center", fontsize=11, color="#D55E00", zorder=8)
    ax.text(-r["y"], r["x"] - r["width"] / 2 - 0.04, "sort rack", ha="center", va="top", fontsize=12, color="0.3", zorder=8)
    hv, _, _ = hull
    ax.add_patch(Polygon(np.c_[-hv[:, 1], hv[:, 0]], closed=True, fc="0.85", ec="0.3", lw=1.5, hatch="xx",
                         alpha=0.9, zorder=3))
    ax.annotate("", xy=(0, 0.22), xytext=(0, 0.0), arrowprops=dict(arrowstyle="->", lw=2), zorder=9)


def waist_main(a):
    yaws = parse_yaws(a.waist_yaw)
    hull = chassis_hull()
    npz = OUT / "reach_grid_waist.npz"
    if a.plot_only and npz.exists():
        z = np.load(npz)
        grids = {(zz, arm): z[f"z{zz}_{arm}"] for zz in W_ZS for arm in ARMS}
        rack = {arm: z[f"rack_{arm}"] for arm in ARMS}
        openings = rack_openings()
        rack_pts = np.concatenate([p for _, p in openings.values()])
        yaws = list(z["yaws"])
    else:
        print(f"waist sweep over {yaws} deg, {len(W_XS)}x{len(W_YS)} grid at Z {W_ZS}", flush=True)
        grids, rack, openings, rack_pts = run_waist_sweep(a.urdf, a.workers, yaws)
        np.savez_compressed(npz, xs=W_XS, ys=W_YS, yaws=np.array(yaws), zs=np.array(W_ZS),
                            **{f"z{zz}_{arm}": grids[(zz, arm)] for zz in W_ZS for arm in ARMS},
                            **{f"rack_{arm}": rack[arm] for arm in ARMS}, rack_pts=rack_pts)
    summ = summarise_waist(grids, rack, openings, rack_pts, yaws)
    (OUT / "reach_summary_waist.json").write_text(json.dumps(summ, indent=1))
    plot_waist(OUT / "g2_reach_envelope_waist.png", grids, openings, summ, yaws, hull)
    print(json.dumps({k: summ[k] for k in ("table_spawn_zone", "rack_openings")}, indent=1))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--waist-yaw", nargs="?", const=DEFAULT_WAIST, default=None,
                    help="sweep the waist yaw (deg list, a:b:step ranges allowed; default "
                         f"{DEFAULT_WAIST}) -> g2_reach_envelope_waist.png")
    a = ap.parse_args()
    if a.waist_yaw is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        return waist_main(a)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / "reach_grid.npz"
    t0 = time.time()
    hull = chassis_hull()
    if a.plot_only:
        z = np.load(npz)
        summ = json.loads((OUT / "reach_summary.json").read_text())
        data = unpack(z, summ)
        plot(OUT / "g2_reach_envelope.png", data, hull)
        return
    fkerr = fk_crosscheck(a.urdf)
    print(f"batched FK vs pinocchio max abs diff: {fkerr:.2e}", flush=True)
    assert fkerr < 1e-9
    with ProcessPoolExecutor(a.workers) as pool:
        cal_f = pool.submit(calibrate, a.urdf)
        grids = {}
        stats_all = {}
        Xg, Yg = np.meshgrid(XS, YS, indexing="ij")
        for z in ZS_TOP:
            pts = np.c_[Xg.ravel(), Yg.ravel(), np.full(Xg.size, z)]
            res, st = run_grid(pool, a.urdf, pts)
            for (arm, mode), v in res.items():
                grids[f"top_{z}_{arm}_{mode}"] = v.reshape(Xg.shape)
            stats_all[f"top_{z}"] = st
            print(f"Z {z}: {time.time()-t0:.0f}s {st}", flush=True)
        Xs, Zs = np.meshgrid(XS, ZS_SIDE, indexing="ij")
        pts = np.c_[Xs.ravel(), np.zeros(Xs.size), Zs.ravel()]
        res, st = run_grid(pool, a.urdf, pts)
        for (arm, mode), v in res.items():
            grids[f"side_{arm}_{mode}"] = v.reshape(Xs.shape)
        stats_all["side"] = st
        print(f"side: {time.time()-t0:.0f}s {st}", flush=True)
        cal = cal_f.result()
    np.savez_compressed(npz, xs=XS, ys=YS, zs_top=np.array(ZS_TOP), zs_side=ZS_SIDE, **grids)

    # summary
    cell_area = STEP * STEP
    c = CARTON
    inx = (XS >= c["cx"] - c["lx"] / 2 - 1e-9) & (XS <= c["cx"] + c["lx"] / 2 + 1e-9)
    iny = (YS >= c["cy"] - c["ly"] / 2 - 1e-9) & (YS <= c["cy"] + c["ly"] / 2 + 1e-9)
    summ = {"method": {"tool_offset_m": TOOL_OFFSET, "base_z_m": BASE_Z, "topdown_cone_deg": TOPDOWN_DEG,
                       "restarts": N_RESTART, "iters": N_ITER, "pos_tol_m": POS_TOL, "grid_step_m": STEP,
                       "body_pose": {"idx03_body_joint3_deg": 20, "others": 0}, "urdf": a.urdf,
                       "fk_crosscheck_max_diff": fkerr, "collision_checking": False},
            "solver_stats": stats_all, "areas_m2": {}, "coverage": {}, "calibration": cal}
    for z in ZS_TOP:
        d = {}
        cov = {}
        for arm in ARMS:
            for mode in ("cone", "pos"):
                g = grids[f"top_{z}_{arm}_{mode}"]
                d[f"{arm}_{mode}"] = round(float(g.sum() * cell_area), 4)
                cov[f"{arm}_{mode}"] = round(float(g[np.ix_(inx, iny)].mean()), 4)
        for mode in ("cone", "pos"):
            L, R = grids[f"top_{z}_left_{mode}"], grids[f"top_{z}_right_{mode}"]
            d[f"both_{mode}"] = round(float((L & R).sum() * cell_area), 4)
            cov[f"both_{mode}"] = round(float((L & R)[np.ix_(inx, iny)].mean()), 4)
            cov[f"either_{mode}"] = round(float((L | R)[np.ix_(inx, iny)].mean()), 4)
        summ["areas_m2"][f"{z}"] = d
        summ["coverage"][f"{z}"] = cov
    summ["coverage_note"] = ("fraction of grid points inside the carton opening X 0.40-0.80, Y -0.30..0.30 "
                             f"({int(inx.sum())}x{int(iny.sum())} points) where the fingertip point is reachable")
    (OUT / "reach_summary.json").write_text(json.dumps(summ, indent=1))
    data = unpack(np.load(npz), summ)
    plot(OUT / "g2_reach_envelope.png", data, hull)
    print(json.dumps(cal, indent=0))
    print(json.dumps(summ["coverage"], indent=1))
    print(f"done in {time.time()-t0:.0f}s")


def unpack(z, summ):
    data = {"coverage": summ["coverage"], "side": {}}
    for zz in ZS_TOP:
        data[f"top_{zz}"] = {f"{arm}_{m}": z[f"top_{zz}_{arm}_{m}"] for arm in ARMS for m in ("cone", "pos")}
    data["side"] = {f"{arm}_{m}": z[f"side_{arm}_{m}"] for arm in ARMS for m in ("cone", "pos")}
    return data


if __name__ == "__main__":
    sys.exit(main())
