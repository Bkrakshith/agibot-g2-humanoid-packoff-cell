"""Pure-logic tests for the object pose relay (no Isaac Sim, no ROS)."""
import math

import pytest

from g2cell_bringup.object_relay import (
    ChangeFilter,
    ObjectPoseRelay,
    patch_launcher,
    pose_changed,
    select_object_bodies,
    tick_divider,
)

I = (1.0, 0.0, 0.0, 0.0)


def rz(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# ---- pose_changed -----------------------------------------------------------------------------

def test_pose_changed_translation_threshold():
    a = ((0.0, 0.0, 0.0), I)
    assert not pose_changed(a, ((0.0004, 0.0, 0.0), I), lin_tol=5e-4, ang_tol=math.radians(0.2))
    assert pose_changed(a, ((0.0006, 0.0, 0.0), I), lin_tol=5e-4, ang_tol=math.radians(0.2))


def test_pose_changed_rotation_threshold_and_sign_ambiguity():
    a = ((0.0, 0.0, 0.0), I)
    assert not pose_changed(a, ((0, 0, 0), rz(0.1)), lin_tol=5e-4, ang_tol=math.radians(0.2))
    assert pose_changed(a, ((0, 0, 0), rz(0.3)), lin_tol=5e-4, ang_tol=math.radians(0.2))
    # q and -q are the same rotation -> no change
    neg = tuple(-c for c in I)
    assert not pose_changed(a, ((0, 0, 0), neg), lin_tol=5e-4, ang_tol=math.radians(0.2))


# ---- ChangeFilter ------------------------------------------------------------------------------

def test_change_filter_first_sample_publishes_all_then_only_movers():
    f = ChangeFilter(keyframe_s=2.0)
    p = {"/World/Bags/Bag_01": ((1, 0, 0.87), I), "/World/Bags/Bag_02": ((1, 0.3, 0.87), I)}
    assert set(f.update(0.0, p)) == set(p)
    assert f.update(0.1, p) == {}
    moved = dict(p)
    moved["/World/Bags/Bag_01"] = ((1, 0, 0.95), I)
    out = f.update(0.2, moved)
    assert list(out) == ["/World/Bags/Bag_01"]
    # change is measured against the last PUBLISHED pose, so slow creep is still caught
    creep = dict(moved)
    for k in range(1, 4):
        creep["/World/Bags/Bag_02"] = ((1, 0.3 + 0.0002 * k, 0.87), I)
        out = f.update(0.2 + 0.03 * k, creep)
    assert "/World/Bags/Bag_02" in out


def test_change_filter_keyframe_republishes_everything():
    f = ChangeFilter(keyframe_s=2.0)
    p = {"/a": ((0, 0, 0), I), "/b": ((1, 0, 0), I)}
    f.update(0.0, p)
    assert f.update(1.9, p) == {}
    assert set(f.update(2.05, p)) == {"/a", "/b"}
    assert f.update(2.1, p) == {}


def test_change_filter_keyframe_disabled():
    f = ChangeFilter(keyframe_s=0.0)
    p = {"/a": ((0, 0, 0), I)}
    f.update(0.0, p)
    assert f.update(100.0, p) == {}


# ---- body selection ---------------------------------------------------------------------------

def test_select_object_bodies_excludes_robot_static_and_disabled():
    entries = [
        # path, rigid, kinematic, enabled
        ("/genie/Geometry/base_link", True, False, True),
        ("/genie_other/x", True, False, True),       # NOT under /genie/ -> kept
        ("/World/Bags/Bag_01", True, False, True),
        ("/World/Conveyor/StopGate", True, True, True),
        ("/World/Off", True, False, False),
        ("/World/Table", False, False, True),
    ]
    assert select_object_bodies(entries, "genie") == ["/genie_other/x", "/World/Bags/Bag_01"]
    assert select_object_bodies(entries, "genie", include_kinematic=True) == [
        "/genie_other/x", "/World/Bags/Bag_01", "/World/Conveyor/StopGate"]


def test_select_object_bodies_skips_nested_bodies():
    # a rigid body under another rigid body (compound child) is moved by its parent; only the top one is relayed
    entries = [("/World/A", True, False, True), ("/World/A/B", True, False, True)]
    assert select_object_bodies(entries, "genie") == ["/World/A"]


# ---- rate --------------------------------------------------------------------------------------

def test_tick_divider():
    assert tick_divider(100.0, 30.0) == 4   # 25 Hz: never above the requested rate
    assert tick_divider(100.0, 50.0) == 2
    assert tick_divider(100.0, 200.0) == 1
    assert tick_divider(120.0, 30.0) == 4
    with pytest.raises(ValueError):
        tick_divider(100.0, 0.0)


# ---- launcher patch ----------------------------------------------------------------------------

def test_patch_launcher_swaps_engine_executable_only():
    cfg = {
        "launcher": {
            "physics": {
                "engine": "genie_sim_engine",
                "engines": {"genie_sim_engine": {"package": "genie_sim_engine",
                                                 "executable": "genie_sim_engine_isaacsim.py",
                                                 "name": "genie_sim_engine"}},
                "industrial_bridge": "",
            },
            "renders": ["render_ovrtx"],
        },
        "render_ovrtx": {"ros__parameters": {"plugin": ["x"]}},
    }
    out = patch_launcher(cfg, package="g2cell_bringup", executable="g2cell_engine_isaacsim.py")
    eng = out["launcher"]["physics"]["engines"]["genie_sim_engine"]
    assert eng == {"package": "g2cell_bringup", "executable": "g2cell_engine_isaacsim.py", "name": "genie_sim_engine"}
    assert out["launcher"]["renders"] == ["render_ovrtx"] and out["render_ovrtx"] == cfg["render_ovrtx"]
    # input untouched
    assert cfg["launcher"]["physics"]["engines"]["genie_sim_engine"]["package"] == "genie_sim_engine"


def test_patch_launcher_rejects_non_isaac_engine():
    cfg = {"launcher": {"physics": {"engine": "mujoco_geniesim", "engines": {
        "mujoco_geniesim": {"package": "p", "executable": "e", "name": "n"}}}}}
    with pytest.raises(ValueError):
        patch_launcher(cfg, package="g2cell_bringup", executable="g2cell_engine_isaacsim.py")


# ---- the hook (fake pose source + publisher) ----------------------------------------------------

class FakeSource:
    def __init__(self, poses):
        self.poses = poses
        self.reads = 0

    def read(self):
        self.reads += 1
        return dict(self.poses)


class FakePub:
    def __init__(self):
        self.sent = []

    def send(self, stamp, items):
        self.sent.append((stamp, dict(items)))


def test_relay_hook_samples_at_divided_rate_and_publishes_changes():
    src = FakeSource({"/World/Bags/Bag_01": ((1, 0, 0.87), I)})
    pub = FakePub()
    relay = ObjectPoseRelay(src, pub, physics_hz=100.0, rate_hz=25.0, keyframe_s=0.0, clock=lambda: 1234.5)
    dt = 0.01
    for k in range(8):                       # 8 ticks -> samples on ticks 0 and 4
        relay(k * dt)
    assert src.reads == 2
    assert len(pub.sent) == 1                # second sample: nothing moved
    stamp, items = pub.sent[0]
    assert stamp == 1234.5                   # wall-clock stamp, like the engine's own /tf_render (node_->now())
    src.poses["/World/Bags/Bag_01"] = ((1, 0, 1.0), I)
    for k in range(8, 12):
        relay(k * dt)
    assert len(pub.sent) == 2 and pub.sent[1][1]["/World/Bags/Bag_01"][0] == (1, 0, 1.0)
    s = relay.stats()
    assert s["samples"] == 3 and s["published_msgs"] == 2 and s["published_tfs"] == 2


def test_relay_hook_survives_source_errors():
    class Boom:
        def read(self):
            raise RuntimeError("prim gone")
    relay = ObjectPoseRelay(Boom(), FakePub(), physics_hz=100.0, rate_hz=100.0)
    relay(0.0)
    relay(0.01)
    assert relay.stats()["errors"] == 2


# ---- poke requests (teleport / velocity, for tests and episode resets) ----------------------------

from g2cell_bringup.object_relay import PokeQueue, parse_poke  # noqa: E402


def test_parse_poke_valid_and_invalid():
    allowed = {"/World/Bags/Bag_01"}
    assert parse_poke('{"path": "/World/Bags/Bag_01", "xyz": [1, 2, 3]}', allowed) == \
        ("/World/Bags/Bag_01", (1.0, 2.0, 3.0), None, None)
    assert parse_poke('{"path": "/World/Bags/Bag_01", "dxyz": [0, 0, 0.3], "vel": [0.5, 0, 0]}', allowed) == \
        ("/World/Bags/Bag_01", None, (0.0, 0.0, 0.3), (0.5, 0.0, 0.0))
    for bad in ("not json", '{"path": "/genie/base_link", "xyz": [0,0,0]}', '{"path": "/World/Bags/Bag_01"}',
                '{"path": "/World/Bags/Bag_01", "xyz": [1, 2]}', '{"path": "/World/Bags/Bag_01", "xyz": [0,0,0], "dxyz": [0,0,1]}'):
        with pytest.raises(ValueError):
            parse_poke(bad, allowed)


def test_poke_queue_drains_in_order():
    q = PokeQueue()
    q.put(("/a", None, (0, 0, 1), None))
    q.put(("/b", (1, 1, 1), None, None))
    assert [r[0] for r in q.drain()] == ["/a", "/b"]
    assert q.drain() == []


def test_relay_hook_applies_pokes_every_tick_before_sampling():
    applied = []

    class Src(FakeSource):
        def apply(self, req):
            applied.append(req)

    q = PokeQueue()
    relay = ObjectPoseRelay(Src({"/a": ((0, 0, 0), I)}), FakePub(), physics_hz=100.0, rate_hz=25.0, pokes=q)
    relay(0.0)
    q.put(("/a", None, (0, 0, 1), None))
    relay(0.01)                              # not a sample tick, the poke is still applied right away
    assert applied == [("/a", None, (0, 0, 1), None)]
    assert relay.stats()["pokes"] == 1
