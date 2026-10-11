"""The fast EnvChecker (parsed-once kinematics, sphere broad phase, early exit) must give the SAME
clearances as the original implementation (copied below as the reference)."""
import math
import pathlib

import numpy as np
import pytest

from g2cell_demo import env_check as ec

_here = pathlib.Path(__file__).resolve()
ASSETS = next((p for p in [pathlib.Path("/geniesim_assets/scenes")] +
               [q / "geniesim_assets" / "scenes" for q in _here.parents] if p.is_dir()),
              pathlib.Path("/nonexistent"))
URDF = ASSETS / "scene_g2cell_mobile" / "robot.urdf"
COLL = ASSETS / "g2cell" / "cell_colliders.json"
needs = pytest.mark.skipif(not (URDF.is_file() and COLL.is_file()), reason="staged robot / cell colliders missing")


# ---- reference: the original (slow) implementation ---------------------------------------------
def ref_link_transforms(chk, joints):
    out = {"base_link": np.eye(4)}
    stack = ["base_link"]
    while stack:
        p = stack.pop()
        for j in chk.children.get(p, []):
            c = j.find("child").get("link")
            m = out[p] @ ec._origin(j)
            typ = j.get("type")
            if typ in ("revolute", "continuous"):
                a = j.find("axis")
                ax = np.array(a.get("xyz").split(), float) if a is not None else np.array([1.0, 0, 0])
                m = m @ ec._h(ec._axang(ax, joints.get(j.get("name"), 0.0)), np.zeros(3))
            elif typ == "prismatic":
                ax = np.array(j.find("axis").get("xyz").split(), float)
                m = m @ ec._h(np.eye(3), ax * joints.get(j.get("name"), 0.0))
            out[c] = m
            stack.append(c)
    return out


def ref_clearance(chk, joints, arm, base=(0.0, 0.0, 0.0), base_z=0.04):
    tf = ref_link_transforms(chk, joints)
    pref = ec.ARM_PREFIX[arm]
    pts = np.vstack([chk.points[n] @ tf[n][:3, :3].T + tf[n][:3, 3] for n in chk.points
                     if n in tf and n.startswith(pref) and not n.endswith("link1")])
    x, y, yaw = base
    c, s = math.cos(yaw), math.sin(yaw)
    pts = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array([x, y, base_z])
    rel = pts[None, :, :] - chk.c[:, None, :]
    loc = np.einsum("bkj,bnj->bnk", chk.ax, rel)
    q = np.abs(loc) - chk.h[:, None, :]
    d = (np.linalg.norm(np.maximum(q, 0.0), axis=2) + np.minimum(q.max(axis=2), 0.0)).min(axis=1)
    i = int(np.argmin(d))
    return float(d[i]), chk.box_names[i]


def ref_head(chk, joints, arm):
    tf = ref_link_transforms(chk, joints)
    h = tf[ec.HEAD_LINK]
    heads = [h[:3, 3], h[:3, :3] @ np.array(ec.HEAD_CAM_IN_LINK3) + h[:3, 3]]
    return float(min(np.linalg.norm(tf[n][:3, 3] - p) for n in ec._check_links(arm) if n in tf for p in heads))


def _checker():
    chk = ec.EnvChecker(str(URDF), str(COLL))
    rng = np.random.default_rng(0)
    for n in list(chk.kin.parent_joint):          # synthetic "meshes" when the real ones are not resolvable
        if n.startswith(("arm_", "gripper_")) and n not in chk.points:
            chk.points[n] = rng.normal(scale=0.04, size=(60, 3))
    chk.reset_geometry()
    return chk


def _pose(rng, arm_scale=0.6):
    j = {"idx03_body_joint3": math.radians(20), "idx05_body_joint5": rng.uniform(-1.8, 1.8),
         "idx13_head_joint3": math.radians(30)}
    for side, b in (("l", 20), ("r", 60)):
        for k, deg in enumerate((0, -45, 0, -75, 0, -55, 0), start=1):
            j[f"idx{b + k}_arm_{side}_joint{k}"] = math.radians(deg) + rng.uniform(-arm_scale, arm_scale)
    return j


@needs
def test_link_transforms_match_reference():
    chk = _checker()
    rng = np.random.default_rng(3)
    for _ in range(5):
        j = _pose(rng)
        a, b = chk.link_transforms(j), ref_link_transforms(chk, j)
        assert set(b) <= set(a)
        for k in b:
            assert np.allclose(a[k], b[k], atol=1e-12), k


@needs
def test_clearance_and_head_match_reference():
    chk = _checker()
    rng = np.random.default_rng(4)
    for _ in range(30):
        j = _pose(rng)
        for arm in ("left", "right"):
            base = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
            d, name = chk.clearance(j, arm, base=base)
            rd, rname = ref_clearance(chk, j, arm, base=base)
            assert d == pytest.approx(rd, abs=1e-9) and name == rname
            assert ec.head_clearance(chk, j, arm) == pytest.approx(ref_head(chk, j, arm), abs=1e-9)


@needs
def test_path_check_matches_reference():
    chk = _checker()
    rng = np.random.default_rng(5)
    for _ in range(4):
        path = [_pose(rng, 0.4) for _ in range(3)]
        env, name, head = ec.path_check(chk, path, "left", base=(0.01, 0.0, 0.0), samples=4)
        best, best_name, best_head = math.inf, "", math.inf
        for a, b in zip(path, path[1:]):
            for k in range(5):
                s = k / 4
                j = {n: a[n] + (b[n] - a[n]) * s for n in a}
                d, nm = ref_clearance(chk, j, "left", base=(0.01, 0.0, 0.0))
                if d < best:
                    best, best_name = d, nm
                best_head = min(best_head, ref_head(chk, j, "left"))
        assert env == pytest.approx(best, abs=1e-9) and name == best_name and head == pytest.approx(best_head)


@needs
def test_path_check_early_exit_only_below_threshold():
    chk = _checker()
    rng = np.random.default_rng(6)
    path = [_pose(rng, 0.4) for _ in range(3)]
    full = ec.path_check(chk, path, "left", samples=4)
    # threshold below the true minimum: identical result
    assert ec.path_check(chk, path, "left", samples=4, stop_below=(full[0] - 1.0, full[2] - 1.0)) == full
    # threshold above it: stops early but still reports a failing value
    env, _, head = ec.path_check(chk, path, "left", samples=4, stop_below=(full[0] + 10.0, 0.0))
    assert env < full[0] + 10.0
