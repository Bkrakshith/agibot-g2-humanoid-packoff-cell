"""Offline kinematic clip of a recorded episode: G2 URDF meshes posed by FK, synced plots, camera.

    .venv-viz/bin/python tools/viz/render_clip.py EPISODE.hdf5 [--ckpt CKPT.pt] [--out OUT.mp4]

Frame layout (1920x1080 or 1280x720, 30 fps = the episode's /aligned clock, so real time):
  * title bar      episode, time, gripper states, open-loop policy error at this step
  * 3/4 view       G2 rendered from its URDF visual meshes (decimated), posed by pinocchio FK from
                   /aligned/joint_states/positions (measured). With --ckpt: semi-transparent orange
                   ghost of the policy's predicted pose (commanded joints replaced by the step-0
                   prediction, gripper mimic joints from the URDF multipliers). Tool-point trails
                   (fingertip centre = arm_*_end_link + 0.125 m along +Z, as in reach_envelope.py):
                   recorded solid, predicted dashed. Left arm blue, right arm green.
  * top-down view  same scene from above
  * cameras        head_front_rgb (and free_rgb) frame at that aligned step, from the HDF5
  * plots          left arm joints 1-7 + gripper: recorded command, measured state, prediction,
                   with a moving time cursor

Frame: cell frame = base_link on the floor (base_link z = 0.04, see tools/reach_envelope.py);
the mobile base is fixed at the origin, x forward, y left.

Policy predictions come from tools/viz/predict_episode.py, run in .venv-learn (torch, CPU) as a
subprocess, so this script only needs .venv-viz (pinocchio, trimesh, matplotlib, h5py,
imageio-ffmpeg). Rendering is matplotlib (Agg, headless) split over worker processes; each worker
encodes an H.264 segment and the segments are concatenated without re-encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                      # g2_packoff_cell/
WS = ROOT.parent                                                # Agibot_dev/
DEFAULT_URDF = WS / "g2_viz_ws/g2_crs_omnipicker.urdf"
PKG_DIRS = [WS / "g2_viz_ws/install/genie_sim_robot_model/share"]
LEARN_PY = WS / ".venv-learn/bin/python"
TOOL_OFFSET = 0.125
BASE_Z = 0.04
ARM_L = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
GRIP_L = "idx31_gripper_l_inner_joint1"
GRIP_R = "idx71_gripper_r_inner_joint1"
GRIP_OPEN, GRIP_CLOSED = None, None                              # set from URDF limits in load_model

C_REC_L, C_REC_R = "#1f5fbf", "#1e9e5a"
C_PRED = "#e8761c"
C_CMD, C_STATE = "#1f3b63", "#8c8c8c"

# ----------------------------------------------------------------------------- model + meshes


def _mimics(urdf):
    """{joint: (master, multiplier, offset)} from the URDF <mimic> tags."""
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1)),
                                  float(m.get("offset", 0)))
    return out


def _link_color(name):
    n = name.lower()
    if "gripper" in n:
        return (0.30, 0.31, 0.34)
    if "arm" in n:
        return (0.80, 0.81, 0.83)
    if "head" in n:
        return (0.92, 0.92, 0.93)
    if "wheel" in n or n.startswith("base_link"):
        return (0.42, 0.43, 0.46)
    return (0.90, 0.90, 0.91)                                   # body links


def _cluster_decimate(v, f, h):
    """Vertex-clustering decimation on an h-metre grid (shape/area preserving, unlike quadric
    decimation, which collapses these non-manifold CAD exports)."""
    key = np.floor(v / h).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = np.zeros((inv.max() + 1, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv)[:, None]
    nf = inv[f]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    _, first = np.unique(np.sort(nf, 1), axis=0, return_index=True)
    return nv, nf[np.sort(first)]


def _grid_size(name, n_faces):
    if n_faces < 5000:
        return None                                             # body/chassis: already coarse
    return 0.006 if "gripper" in name else 0.010


def load_meshes(urdf, cache_dir):
    """Decimated visual meshes, one per geometry object, cached as npz."""
    import pinocchio as pin
    import trimesh

    model = pin.buildModelFromUrdf(str(urdf))
    gm = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.VISUAL, [str(p) for p in PKG_DIRS])
    key = hashlib.sha1((Path(urdf).read_bytes() + b"v3-cluster")).hexdigest()[:12]
    cache = Path(cache_dir) / f"meshes_{key}.npz"
    geoms = []
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        geoms = list(z["geoms"])
    else:
        for g in gm.geometryObjects:
            tm = trimesh.load(g.meshPath, force="mesh")
            tm.apply_scale(np.asarray(g.meshScale))
            v, f = np.asarray(tm.vertices, float), np.asarray(tm.faces)
            h = _grid_size(g.name, len(f))
            if h:
                v, f = _cluster_decimate(v, f, h)
            geoms.append({"name": g.name, "v": v.astype(np.float32), "f": f.astype(np.int32)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, geoms=np.array(geoms, dtype=object))
    return model, gm, geoms


class Kin:
    """pinocchio FK for full-robot poses given named joint values."""

    def __init__(self, urdf, cache_dir):
        import pinocchio as pin
        self.pin = pin
        self.model, self.gm, self.geoms = load_meshes(urdf, cache_dir)
        self.data = self.model.createData()
        self.gdata = pin.GeometryData(self.gm)
        self.mimic = _mimics(urdf)
        m = self.model
        self.qidx = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)
                     if m.joints[j].nq == 1}
        self.fid = {a: m.getFrameId(f"arm_{a}_end_link") for a in "lr"}
        global GRIP_OPEN, GRIP_CLOSED
        jid = m.getJointId(GRIP_L)
        lo, hi = m.lowerPositionLimit[m.joints[jid].idx_q], m.upperPositionLimit[m.joints[jid].idx_q]
        GRIP_OPEN, GRIP_CLOSED = float(lo), float(hi)           # verified visually, see README

    def q_from(self, names, values):
        q = self.pin.neutral(self.model)
        for n, v in zip(names, values):
            if n in self.qidx and np.isfinite(v):
                q[self.qidx[n]] = v
        return q

    def poses(self, Q):
        """Q [T,nq] -> geometry transforms [T,G,4,4] (cell frame), tool points [T,2,3]."""
        pin, m = self.pin, self.model
        T, G = len(Q), len(self.gm.geometryObjects)
        out = np.zeros((T, G, 4, 4), np.float32)
        tool = np.zeros((T, 2, 3))
        lift = np.eye(4)
        lift[2, 3] = BASE_Z
        for t, q in enumerate(Q):
            pin.forwardKinematics(m, self.data, q)
            pin.updateFramePlacements(m, self.data)
            pin.updateGeometryPlacements(m, self.data, self.gm, self.gdata)
            for g in range(G):
                out[t, g] = lift @ self.gdata.oMg[g].homogeneous
            for k, a in enumerate("lr"):
                M = self.data.oMf[self.fid[a]]
                tool[t, k] = M.translation + TOOL_OFFSET * M.rotation[:, 2] + [0, 0, BASE_Z]
        return out, tool

    def apply_mimic(self, names, values):
        """Fill mimic joints from their masters (for the predicted pose)."""
        d = dict(zip(names, values))
        for j, (master, mul, off) in self.mimic.items():
            if master in d and np.isfinite(d[master]):
                d[j] = mul * d[master] + off
        return list(d.keys()), np.array(list(d.values()))


# ----------------------------------------------------------------------------- episode data


def load_episode(path):
    import h5py
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["joint_states"].attrs["names"]]
        cmd_names = ([n.decode() if isinstance(n, bytes) else str(n)
                      for n in f["joint_command"].attrs["names"]] if "joint_command" in f else [])
        pos = f["aligned/joint_states/positions"][:]
        cmd = f["aligned/joint_command/actions"][:] if "aligned/joint_command/actions" in f else None
        t_ns = f["aligned/t_ns"][:]
        cams = {}
        for cam in ("head_front_rgb", "free_rgb"):
            k = f"aligned/images/{cam}/index"
            if k in f and f"images/{cam}/data" in f:
                cams[cam] = f[k][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    return {"names": names, "cmd_names": cmd_names, "pos": pos, "cmd": cmd,
            "t": (t_ns - t_ns[0]) * 1e-9, "cams": cams, "attrs": attrs, "path": str(path)}


def run_prediction(ckpt, episode, out_npz):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([str(LEARN_PY), str(Path(__file__).with_name("predict_episode.py")),
                        "--ckpt", str(ckpt), "--episode", str(episode), "--out", str(out_npz)],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"prediction failed:\n{r.stderr[-3000:]}")
    z = np.load(out_npz, allow_pickle=False)
    return {"pred0": z["pred0"], "recorded": z["recorded"],
            "names": [str(n) for n in z["joint_names"]], "meta": json.loads(str(z["meta_json"]))}


# ----------------------------------------------------------------------------- rendering

LIGHT = np.array([0.45, 0.35, 0.82])
LIGHT /= np.linalg.norm(LIGHT)
G = {}                                                          # per-process render state


def _scene_faces(T_g, geoms, include, base_rgba):
    """World-space triangles [F,3,3] and shaded RGBA [F,4] for the geometry ids in include."""
    tris, cols = [], []
    for g in include:
        v = geoms[g]["v"] @ T_g[g, :3, :3].T + T_g[g, :3, 3]
        tri = v[geoms[g]["f"]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lam = np.abs(n @ LIGHT)
        rgba = np.empty((len(tri), 4))
        rgba[:, :3] = np.clip(base_rgba[g][:3] * (0.38 + 0.62 * lam[:, None]), 0, 1)
        rgba[:, 3] = base_rgba[g][3]
        tris.append(tri)
        cols.append(rgba)
    return np.concatenate(tris), np.concatenate(cols)


def _setup_figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    W, H = S["size"]
    dpi = 100
    sc = H / 1080.0
    plt.rcParams.update({"font.size": 10 * sc, "axes.titlesize": 10 * sc,
                         "axes.labelsize": 9 * sc, "xtick.labelsize": 8 * sc,
                         "ytick.labelsize": 8 * sc, "legend.fontsize": 9 * sc})
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    A = {"fig": fig, "sc": sc}
    # title bar
    A["title"] = fig.text(0.008, 0.972, "", fontsize=17 * sc, weight="bold", va="center",
                          family="DejaVu Sans")
    A["subtitle"] = fig.text(0.008, 0.938, "", fontsize=12.5 * sc, va="center",
                             family="DejaVu Sans Mono", color="0.15")
    fig.add_artist(Line2D([0, 1], [0.918, 0.918], color="0.75", lw=1))

    views = {"main": ([0.0, 0.0, 0.43, 0.915], S["view_main"]),
             "top": ([0.425, 0.505, 0.225, 0.41], S["view_top"])}
    xl, yl, zl = S["lims"]
    for key, (rect, (elev, azim)) in views.items():
        ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_zlim(*zl)
        ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]),
                          zoom=S["zoom"][key])
        ax.set_proj_type("persp", focal_length=0.6) if key == "main" else ax.set_proj_type("ortho")
        ax.set_axis_off()
        # floor grid
        gx = np.arange(math.ceil(xl[0] * 4) / 4, xl[1] + 1e-6, 0.25)
        gy = np.arange(math.ceil(yl[0] * 4) / 4, yl[1] + 1e-6, 0.25)
        for x in gx:
            ax.plot([x, x], [yl[0], yl[1]], [0, 0], color="0.85", lw=0.7, zorder=0)
        for y in gy:
            ax.plot([xl[0], xl[1]], [y, y], [0, 0], color="0.85", lw=0.7, zorder=0)
        ax.plot([0, 0.3], [0, 0], [0, 0], color="#c0392b", lw=1.5, zorder=1)   # +x (forward)
        ax.plot([0, 0], [0, 0.3], [0, 0], color="#27ae60", lw=1.5, zorder=1)   # +y (left)
        pc = Poly3DCollection([], linewidths=0, zorder=2)
        ax.add_collection3d(pc)
        tr = {}
        for k, c in ((0, C_REC_L), (1, C_REC_R)):
            tr[("rec", k)], = ax.plot([], [], [], color=c, lw=2.2 * sc, zorder=5)
            tr[("dot", k)], = ax.plot([], [], [], "o", color=c, ms=6 * sc, zorder=6,
                                      mec="white", mew=0.8)
            if S["pred"] is not None:
                tr[("pred", k)], = ax.plot([], [], [], color=C_PRED, lw=1.8 * sc, ls=(0, (4, 3)),
                                           zorder=5)
        A[key] = {"ax": ax, "pc": pc, "trails": tr}
    fig.text(0.01, 0.895, "3/4 view (FK from recorded joint_states)", fontsize=11 * sc,
             color="0.3")
    fig.text(0.43, 0.895, "top-down", fontsize=11 * sc, color="0.3")
    # legend for the 3D panels
    hs = [Line2D([], [], color=C_REC_L, lw=2.2, label="left tool path (recorded)"),
          Line2D([], [], color=C_REC_R, lw=2.2, label="right tool path (recorded)")]
    if S["pred"] is not None:
        hs += [Line2D([], [], color=C_PRED, lw=1.8, ls="--", label="predicted tool path"),
               Line2D([], [], color=C_PRED, lw=8, alpha=0.35, label="predicted pose (ghost; tints the arm where it matches)")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.005, 0.005), frameon=False,
               fontsize=10 * sc)

    # cameras
    A["cams"] = {}
    cam_rects = {"head_front_rgb": [0.432, 0.255, 0.215, 0.24],
                 "free_rgb": [0.432, 0.012, 0.215, 0.23]}
    for cam, rect in cam_rects.items():
        if cam not in S["cams"]:
            continue
        ax = fig.add_axes(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        h, w = S["cam_shape"][cam]
        A["cams"][cam] = ax.imshow(np.zeros((h, w, 3), np.uint8), aspect="equal")
        ax.set_title({"head_front_rgb": "head camera (what the robot saw)",
                      "free_rgb": "free camera (sim overview)"}[cam], fontsize=10 * sc, pad=3)

    # joint plots
    t = S["t"]
    A["cursors"] = []
    A["plots"] = []
    pj = S["plot_joints"]
    n = len(pj)
    rows = (n + 1) // 2
    x0, x1, y0, y1 = 0.685, 0.995, 0.055, 0.875
    cw, rh = (x1 - x0) / 2, (y1 - y0) / rows
    for i, (label, rec, st, pr) in enumerate(pj):
        r, c = divmod(i, 2)
        ax = fig.add_axes([x0 + c * cw + 0.022, y1 - (r + 1) * rh + 0.022, cw - 0.03, rh - 0.042])
        if st is not None:
            ax.plot(t, st, color=C_STATE, lw=2.6 * sc, alpha=0.55, label="measured")
        if rec is not None:
            ax.plot(t, rec, color=C_CMD, lw=1.3 * sc, label="recorded cmd")
        if pr is not None:
            ax.plot(t, pr, color=C_PRED, lw=1.3 * sc, ls="--", label="policy pred.")
        ax.set_title(label, fontsize=9.5 * sc, pad=2, loc="left")
        ax.grid(alpha=0.3)
        ax.set_xlim(t[0], t[-1])
        vals = np.concatenate([np.ravel(a) for a in (rec, st, pr) if a is not None])
        vals = vals[np.isfinite(vals)]
        lo, hi = vals.min(), vals.max()
        pad = max(0.05 * (hi - lo), 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        if r == rows - 1:
            ax.set_xlabel("time [s]")
        A["cursors"].append(ax.axvline(t[0], color="#c0392b", lw=1.2))
        A["plots"].append(ax)
    h, l = A["plots"][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.912), ncol=3, frameon=False,
               fontsize=9.5 * sc)
    fig.text(0.69, 0.895, f"{S['arm_label']} [rad]", fontsize=11 * sc, color="0.3")
    return A


def _grip_text(v):
    if not np.isfinite(v):
        return "  n/a "
    frac = (v - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN)       # 0 open .. 1 closed
    state = "CLOSED" if frac > 0.5 else "open"
    return f"{state:<6}({frac * 100:3.0f}%)"


def render_frame(i, A, S):
    import numpy as np
    W, H = S["size"]
    T_rec = S["T_rec"][i]
    tris, cols = _scene_faces(T_rec, S["geoms"], S["all_g"], S["rgba"])
    if S["pred"] is not None and S["pred_ok"][i]:
        gt, gc = _scene_faces(S["T_pred"][i], S["geoms"], S["ghost_g"], S["ghost_rgba"])
        tris = np.concatenate([tris, gt])
        cols = np.concatenate([cols, gc])
    for key in ("main", "top"):
        P = A[key]
        P["pc"].set_verts(tris)
        P["pc"].set_facecolor(cols)
        for k in (0, 1):
            p = S["tool_rec"][: i + 1, k]
            P["trails"][("rec", k)].set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            P["trails"][("dot", k)].set_data_3d(p[-1:, 0], p[-1:, 1], p[-1:, 2])
            if S["pred"] is not None:
                q = S["tool_pred"][: i + 1, k]
                P["trails"][("pred", k)].set_data_3d(q[:, 0], q[:, 1], q[:, 2])
    t = S["t"][i]
    for c in A["cursors"]:
        c.set_xdata([t, t])
    for cam, im in A["cams"].items():
        im.set_data(S["cam_reader"](cam, i))
    # title
    A["title"].set_text(f"{S['ep_name']}   |   kinematic replay   |   t = {t:5.2f} / {S['t'][-1]:.2f} s"
                        f"   (step {i:4d}/{len(S['t']) - 1})")
    gl = S["grip_state"][i, 0]
    gr = S["grip_state"][i, 1]
    sub = f"L gripper {_grip_text(gl)}   R gripper {_grip_text(gr)}   "
    if S["pred"] is not None:
        e = S["err"][i]
        sub += ("policy err: " + (f"L-arm RMS {e[0]:5.2f} deg, max {e[1]:5.2f} deg, "
                                  f"gripper {e[2]:.3f} rad" if np.isfinite(e[0]) else "n/a")
                + f"   [{S['pred_label']}]")
    else:
        sub += "no policy checkpoint (recorded motion only)"
    A["subtitle"].set_text(sub)
    A["fig"].canvas.draw()
    buf = np.asarray(A["fig"].canvas.buffer_rgba())[:, :, :3]
    return np.ascontiguousarray(buf[:H, :W])


def _cam_reader_factory(path, cams, max_w):
    import h5py
    state = {}

    def read(cam, i):
        if "f" not in state:
            state["f"] = h5py.File(path, "r")
        k = int(cams[cam][i])
        img = state["f"][f"images/{cam}/data"][k]
        step = max(1, int(math.ceil(img.shape[1] / max_w)))
        return img[::step, ::step]
    return read


def _worker(args):
    seg_path, frames = args
    import imageio_ffmpeg
    S = G["S"]
    S["cam_reader"] = _cam_reader_factory(S["ep_path"], S["cams"], S["cam_max_w"])
    A = _setup_figure(S)
    W, H = S["size"]
    w = imageio_ffmpeg.write_frames(seg_path, (W, H), fps=S["fps"], codec="libx264",
                                    pix_fmt_out="yuv420p", quality=None,
                                    output_params=["-crf", "20", "-preset", "medium",
                                                   "-g", str(S["fps"])],
                                    macro_block_size=8, ffmpeg_log_level="error")
    w.send(None)
    for i in frames:
        w.send(render_frame(i, A, S))
    w.close()
    return seg_path, len(frames)


# ----------------------------------------------------------------------------- main


def build_state(a):
    t0 = time.time()
    ep = load_episode(a.episode)
    kin = Kin(a.urdf, a.cache_dir)
    names, pos = ep["names"], ep["pos"]
    T = len(pos)
    if a.max_frames:
        T = min(T, a.max_frames)
    Q_rec = np.stack([kin.q_from(names, pos[i]) for i in range(T)])
    T_rec, tool_rec = kin.poses(Q_rec)

    pred = None
    if a.ckpt:
        tmp = Path(a.cache_dir) / f"pred_{Path(a.episode).stem}.npz"
        pred = run_prediction(a.ckpt, a.episode, tmp)
        pn = pred["names"]
        Q_pred, ok = [], []
        for i in range(T):
            p = pred["pred0"][i]
            if not np.isfinite(p).all():
                ok.append(False)
                Q_pred.append(Q_rec[i])
                continue
            d = dict(zip(names, pos[i]))
            d.update(zip(pn, p))
            nn, vv = kin.apply_mimic(list(d.keys()), list(d.values()))
            Q_pred.append(kin.q_from(nn, vv))
            ok.append(True)
        Q_pred = np.stack(Q_pred)
        T_pred, tool_pred = kin.poses(Q_pred)
        tool_pred[~np.array(ok)] = np.nan
    # geometry groups and colours
    gnames = [g["name"] for g in kin.geoms]
    rgba = np.array([(*_link_color(n), 1.0) for n in gnames])
    ghost_g = [k for k, n in enumerate(gnames) if "arm_" in n or "gripper_" in n]
    ghost_rgba = np.tile(np.array([[0.96, 0.52, 0.16, 0.15]]), (len(gnames), 1))

    # plots: left arm 7 joints + gripper
    def col(nm_list, arr, n):
        return arr[:T, nm_list.index(n)] if arr is not None and n in nm_list else None
    plot_joints = []
    for n in ARM_L + [GRIP_L]:
        lab = n.replace("idx", "").split("_", 1)[1] if n in ARM_L else "gripper_l (inner_joint1)"
        rec = col(ep["cmd_names"], ep["cmd"], n)
        st = col(names, pos, n)
        pr = col(pred["names"], pred["pred0"], n) if pred else None
        plot_joints.append((lab, rec, st, pr))
    grip_state = np.stack([pos[:T, names.index(GRIP_L)], pos[:T, names.index(GRIP_R)]], 1)

    err = None
    pred_label = ""
    if pred:
        pn = pred["names"]
        ia = [pn.index(n) for n in ARM_L]
        ig = pn.index(GRIP_L)
        d = pred["pred0"][:T] - pred["recorded"][:T]
        err = np.stack([np.degrees(np.sqrt(np.mean(d[:, ia] ** 2, 1))),
                        np.degrees(np.max(np.abs(d[:, ia]), 1)), np.abs(d[:, ig])], 1)
        m = pred["meta"]
        seen = any(Path(e).resolve() == Path(a.episode).resolve()
                   for e in [ROOT / p for p in m.get("train_episodes", [])])
        pred_label = (f"{m['policy']} step {m['step']}, open-loop" +
                      (", TRAINED ON THIS EPISODE" if seen else ""))

    # scene limits from the recorded meshes (union over frames, coarse)
    pts = []
    for i in np.linspace(0, T - 1, 6).astype(int):
        for g in range(len(gnames)):
            v = kin.geoms[g]["v"][::7]
            pts.append(v @ T_rec[i, g, :3, :3].T + T_rec[i, g, :3, 3])
    pts = np.concatenate(pts + [tool_rec.reshape(-1, 3)])
    lo, hi = pts.min(0), pts.max(0)
    lo[2] = 0.0
    c = (lo + hi) / 2
    half = (hi - lo).max() / 2 + 0.08
    lims = ((c[0] - half, c[0] + half), (c[1] - half, c[1] + half), (0.0, 2 * half))

    cam_shape = {}
    import h5py
    with h5py.File(a.episode, "r") as f:
        for cam in ep["cams"]:
            h, w = f[f"images/{cam}/data"].shape[1:3]
            step = max(1, int(math.ceil(w / a.cam_max_w)))
            cam_shape[cam] = (len(range(0, h, step)), len(range(0, w, step)))

    S = {"size": tuple(a.size), "fps": 30, "t": ep["t"][:T], "T_rec": T_rec,
         "tool_rec": tool_rec, "geoms": kin.geoms, "all_g": list(range(len(gnames))),
         "rgba": rgba, "pred": pred is not None, "ghost_g": ghost_g, "ghost_rgba": ghost_rgba,
         "lims": lims, "view_main": (a.elev, a.azim), "view_top": (89.9, 180.0),
         "zoom": {"main": 1.4, "top": 1.35}, "cams": ep["cams"], "cam_shape": cam_shape,
         "cam_max_w": a.cam_max_w, "ep_path": str(a.episode), "plot_joints": plot_joints,
         "grip_state": grip_state, "ep_name": Path(a.episode).stem, "arm_label": "left arm",
         "err": err, "pred_label": pred_label}
    if pred:
        S.update({"T_pred": T_pred, "tool_pred": tool_pred, "pred_ok": np.array(ok)})
    else:
        S["pred"] = None
    print(f"[render_clip] prepared {T} frames, {sum(len(g['f']) for g in kin.geoms)} faces "
          f"(+{sum(len(kin.geoms[g]['f']) for g in ghost_g) if pred else 0} ghost) in "
          f"{time.time() - t0:.1f}s", flush=True)
    return S


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", type=Path)
    p.add_argument("--ckpt", type=Path, help="policy checkpoint -> ghost prediction overlay")
    p.add_argument("--out", type=Path, help="default output/viz/<episode>_kinematics.mp4")
    p.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    p.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 4)))
    p.add_argument("--stills", type=int, default=4, help="PNG stills next to the MP4")
    p.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=40.0, help="3/4 view azimuth (deg)")
    p.add_argument("--cam-max-w", type=int, default=640)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "output/viz/.cache")
    a = p.parse_args(argv)
    if a.size[1] not in (720, 1080):
        raise SystemExit("--size must be 1280 720 or 1920 1080")
    a.episode = a.episode.resolve()
    out = a.out or ROOT / "output/viz" / f"{a.episode.stem}_kinematics.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    S = build_state(a)
    G["S"] = S
    T = len(S["t"])

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    nw = max(1, min(a.workers, T // 10 or 1))
    chunks = [c for c in np.array_split(np.arange(T), nw) if len(c)]
    with tempfile.TemporaryDirectory(dir=a.cache_dir) as td:
        segs = [(str(Path(td) / f"seg_{k:03d}.mp4"), c.tolist()) for k, c in enumerate(chunks)]
        import multiprocessing as mp
        with ProcessPoolExecutor(len(segs), mp_context=mp.get_context("fork")) as ex:
            done = list(ex.map(_worker, segs))
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{s}'\n" for s, _ in done))
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr)
    t_video = time.time() - t_start

    stills = []
    if a.stills:
        from PIL import Image
        S["cam_reader"] = _cam_reader_factory(S["ep_path"], S["cams"], S["cam_max_w"])
        A = _setup_figure(S)
        for k, i in enumerate(np.linspace(0.08, 0.92, a.stills) * (T - 1)):
            i = int(round(i))
            sp = out.with_name(f"{out.stem}_still{k + 1}_t{S['t'][i]:05.2f}s.png")
            Image.fromarray(render_frame(i, A, S)).save(sp)
            stills.append(str(sp))
    info = {"out": str(out), "frames": T, "fps": 30, "duration_s": T / 30,
            "size": list(a.size), "workers": len(segs), "render_s": round(t_video, 1),
            "total_s": round(time.time() - t_start, 1), "stills": stills,
            "checkpoint": str(a.ckpt) if a.ckpt else None}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
