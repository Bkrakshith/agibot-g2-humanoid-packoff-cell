"""GraspAssist (engine-side sticky grasp for the rigid placeholder bags): pure logic, numpy only."""
import math

import numpy as np
import pytest

from g2cell_bringup.object_relay import GraspAssist, parse_poke, pose_matrix

BAG = "/World/Bags/Bag_01"
OTHER = "/World/Bags/Bag_02"
TIP = (0.0, 0.0, 0.12)     # fingertip in the gripper link frame (+Z of the link = towards the fingers)


def link_down(x, y, z_tip):
    """Gripper link pose with its +Z pointing down (gripper-down) and the fingertip at (x, y, z_tip)."""
    R = np.diag([1.0, -1.0, -1.0])      # 180 deg about X: link +Z -> world -Z
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z_tip + TIP[2])
    return T


def bag_at(x, y, z=0.87, yaw_deg=0.0):
    return pose_matrix((x, y, z), yaw_deg=yaw_deg)


class World:
    def __init__(self):
        self.links = {"left": link_down(0.05, 0.62, 2.0)}
        self.bodies = {BAG: bag_at(0.05, 0.62), OTHER: bag_at(-0.25, 0.62)}

    def link(self, arm):
        return self.links[arm]

    def body(self, path):
        return self.bodies[path]


def assist(**kw):
    return GraspAssist([BAG, OTHER], ["left"], tool_offset=TIP, **kw)


def test_closing_over_a_bag_top_attaches_it():
    w, g = World(), assist()
    assert g.step({"left": -0.78}, w.link, w.body) == []          # open, far above
    w.links["left"] = link_down(0.05, 0.62, 0.93)                  # fingertips 1 cm below the bag top (0.94)
    assert g.step({"left": -0.78}, w.link, w.body) == []          # still open
    ev = g.step({"left": -0.10}, w.link, w.body)                   # closing
    assert ev == [("attach", "left", BAG)]
    assert g.held == {"left": BAG}


def test_closing_in_the_air_or_beside_the_bag_attaches_nothing():
    w, g = World(), assist()
    w.links["left"] = link_down(0.05, 0.62, 1.10)                  # 16 cm above the top
    g.step({"left": -0.78}, w.link, w.body)
    assert g.step({"left": 0.0}, w.link, w.body) == [("miss", "left", BAG)]
    assert g.misses[-1][2] == BAG and abs(g.misses[-1][3][2] - 0.23) < 1e-6   # diagnostics: tip in bag frame
    g2 = assist()
    w.links["left"] = link_down(0.30, 0.62, 0.93)                  # beside the bag (x 0.25 off centre)
    g2.step({"left": -0.78}, w.link, w.body)
    assert [e[0] for e in g2.step({"left": 0.0}, w.link, w.body)] == ["miss"] and g2.held == {}


def test_only_a_fresh_close_attaches():
    # gripper already closed when it arrives over the bag (e.g. moving home closed): no attach
    w, g = World(), assist()
    g.step({"left": 0.0}, w.link, w.body)
    w.links["left"] = link_down(0.05, 0.62, 0.93)
    assert g.step({"left": 0.0}, w.link, w.body) == []


def test_held_bag_follows_the_link_with_the_grasp_offset_and_detaches_on_open():
    w, g = World(), assist()
    w.links["left"] = link_down(0.06, 0.60, 0.93)                  # grasp 1 cm / 2 cm off centre
    g.step({"left": -0.78}, w.link, w.body)
    g.step({"left": -0.05}, w.link, w.body)
    w.links["left"] = link_down(0.56, 0.0, 1.20)                   # carried over the carton
    tg = g.targets(w.link)
    p = tg[BAG][:3, 3]
    assert np.allclose(p, (0.55, 0.02, 0.87 + (1.20 - 0.93))), p   # offset kept
    assert np.allclose(tg[BAG][:3, :3], np.eye(3), atol=1e-9)       # yaw kept
    assert g.step({"left": -0.30}, w.link, w.body) == []            # half open: hysteresis, still held
    assert g.step({"left": -0.70}, w.link, w.body) == [("detach", "left", BAG)]
    assert g.held == {} and g.targets(w.link) == {}


def test_nearest_bag_wins_and_yawed_bag_footprint_is_used():
    w, g = World(), assist()
    w.bodies[BAG] = bag_at(0.05, 0.62, yaw_deg=90.0)              # 0.30 m side now along X
    w.links["left"] = link_down(0.05 + 0.13, 0.62, 0.93)          # inside the yawed footprint only
    g.step({"left": -0.78}, w.link, w.body)
    assert g.step({"left": 0.0}, w.link, w.body) == [("attach", "left", BAG)]


def test_pose_matrix_yaw():
    T = pose_matrix((1.0, 2.0, 3.0), yaw_deg=90.0)
    assert np.allclose(T[:3, 3], (1, 2, 3)) and np.allclose(T[:3, :3] @ (1, 0, 0), (0, 1, 0))


def test_parse_poke_accepts_yaw_and_stop():
    p = parse_poke('{"path": "/World/Bags/Bag_01", "xyz": [0, 0.6, 0.87], "yaw_deg": 30, "stop": true}', {BAG})
    assert p.path == BAG and p.xyz == (0.0, 0.6, 0.87) and p.yaw_deg == 30.0 and p.stop
    with pytest.raises(ValueError):
        parse_poke('{"path": "/World/Bags/Bag_01", "yaw_deg": "x"}', {BAG})
    q = parse_poke('{"path": "/World/Bags/Bag_01", "stop": true}', {BAG})
    assert q.xyz is None and q.stop and math.isnan(q.yaw_deg if q.yaw_deg is not None else math.nan)


# ---- the engine hook around GraspAssist (IO injected) ----------------------------------------------------
from g2cell_bringup.object_relay import GraspAssistHook  # noqa: E402


class FakeIO:
    def __init__(self, w):
        self.w, self.set, self.held = w, [], []

    def link_world(self, arm):
        return self.w.link(arm)

    def body_world(self, path):
        return self.w.body(path)

    def set_body_world(self, path, T):
        self.set.append(path)
        self.w.bodies[path] = T

    def set_held(self, path, held):
        self.held.append((path, held))


def test_hook_detects_at_its_rate_and_moves_held_bags_every_tick():
    w = World()
    io = FakeIO(w)
    q = {"left": -0.78}
    hook = GraspAssistHook(assist(), io, lambda: dict(q), div=4)
    w.links["left"] = link_down(0.05, 0.62, 0.93)
    for _ in range(4):
        hook(0.0)
    q["left"] = 0.0
    for _ in range(4):
        hook(0.0)                         # the close is seen on the next detection tick
    assert io.held == [(BAG, True)]
    w.links["left"] = link_down(0.05, 0.62, 1.20)
    n = len(io.set)
    hook(0.0)
    hook(0.0)
    assert len(io.set) == n + 2 and math.isclose(w.bodies[BAG][2, 3], 0.87 + 0.27)
    q["left"] = -0.78
    for _ in range(4):
        hook(0.0)
    assert io.held[-1] == (BAG, False)
    assert hook.stats()["attach"] == 1 and hook.stats()["detach"] == 1


def test_hook_survives_io_errors():
    w = World()
    io = FakeIO(w)

    def boom():
        raise RuntimeError("articulation read failed")
    hook = GraspAssistHook(assist(), io, boom, div=1)
    hook(0.0)
    assert hook.stats()["errors"] == 1


# ---- USD adapters against an in-memory stage (usd-core; the engine uses the same pxr API) ------------------
pxr = pytest.importorskip("pxr")


def _stage():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/World")
    UsdGeom.Xform.Define(st, "/World/Bags")
    bag = UsdGeom.Xform.Define(st, BAG)
    bag.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.62, 0.87))
    bag.AddRotateZOp().Set(0.0)
    UsdPhysics.RigidBodyAPI.Apply(bag.GetPrim())
    box = UsdGeom.Cube.Define(st, BAG + "/Colliders/Box")
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    other = UsdGeom.Xform.Define(st, "/World/Bags/Bag_02")
    other.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    other.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    g = UsdGeom.Xform.Define(st, "/genie/Geometry/base_link/arm_l_link7/gripper_l_base_link")
    g.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 1.0))
    return st


def test_set_xform_ops_rotate_z_and_orient():
    from pxr import UsdGeom
    from g2cell_bringup.object_relay import set_xform_ops
    st = _stage()
    T = pose_matrix((0.3, 0.1, 1.2), yaw_deg=30.0)
    for path in (BAG, "/World/Bags/Bag_02"):
        xf = UsdGeom.Xformable(st.GetPrimAtPath(path))
        set_xform_ops(xf, T[:3, 3], T[:3, :3])
        m = xf.GetLocalTransformation()
        M = np.array([[m[r][c] for c in range(4)] for r in range(4)]).T
        assert np.allclose(M, T, atol=1e-5), path


def test_usd_grasp_io_moves_and_holds_a_bag():
    from g2cell_bringup.object_relay import UsdGraspIO, find_gripper_links
    st = _stage()
    links = find_gripper_links(st, "genie")
    assert links == {"left": "/genie/Geometry/base_link/arm_l_link7/gripper_l_base_link"}
    io = UsdGraspIO(st, links)
    assert np.allclose(io.link_world("left")[:3, 3], (0.1, 0.2, 1.0))
    T = pose_matrix((0.6, 0.0, 1.1), yaw_deg=-45.0)
    io.set_body_world(BAG, T)
    assert np.allclose(io.body_world(BAG), T, atol=1e-6)
    io.set_held(BAG, True)
    prim = st.GetPrimAtPath(BAG)
    assert prim.GetAttribute("physics:kinematicEnabled").Get() is True
    assert st.GetPrimAtPath(BAG + "/Colliders/Box").GetAttribute("physics:collisionEnabled").Get() is False
    io.set_held(BAG, False)
    assert prim.GetAttribute("physics:kinematicEnabled").Get() is False
    assert st.GetPrimAtPath(BAG + "/Colliders/Box").GetAttribute("physics:collisionEnabled").Get() is True


def test_command_tracker_prefers_the_commanded_gripper_value():
    # live run 2026-10-11: fingers pressed on the rigid bag top cannot close (measured q stays -0.795) until the
    # lift starts, so detection must use /joint_command, falling back to the measurement
    from g2cell_bringup.object_relay import CommandTracker
    tr = CommandTracker({"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"})
    measured = {"left": -0.795, "right": -0.1}
    assert tr.merge(measured) == measured
    tr.update(["idx21_arm_l_joint1", "idx31_gripper_l_inner_joint1"], [0.3, 0.0])
    assert tr.merge(measured) == {"left": 0.0, "right": -0.1}
    tr.update(["idx31_gripper_l_inner_joint1"], [-0.785])
    assert tr.merge(measured)["left"] == -0.785
