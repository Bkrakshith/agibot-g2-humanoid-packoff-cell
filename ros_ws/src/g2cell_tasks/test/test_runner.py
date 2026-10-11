"""MotionRunner (PickBag stages on top of MoveArm) against a fake MotionClient: what each stage asks the motion
server for (waist mode, orientation, via / lookahead points) and that recovery always ends at home."""
import math
import pathlib
import re
from types import SimpleNamespace

import pytest
import yaml

from g2cell_tasks.config import load_task_config
from g2cell_tasks.reachability import select_arm
from g2cell_tasks.runner import BUSY, CANCELED, HEAD_CLEARANCE, IK_FAILED, OK, MotionRunner, plan_task
from g2cell_tasks.state_machine import Hooks, PickPlaceMachine

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = load_task_config(yaml.safe_load((ROOT / "g2cell_tasks" / "config" / "tasks.yaml").read_text()))
Q = (1.0, 0.0, 0.0, 0.0)


def res(code=OK, quat=Q, msg=""):
    q = SimpleNamespace(x=quat[0], y=quat[1], z=quat[2], w=quat[3]) if quat else SimpleNamespace(x=0, y=0, z=0, w=0)
    return SimpleNamespace(error_code=code, message=msg or ("ok" if code == OK else "planning failed: ik_failed"),
                           orientation_used=q)


class FakeMC:
    """Records every call; `fail(pred)` -> result code for a call (OK if None)."""

    def __init__(self, fail=lambda c: None):
        self.calls, self.fail = [], fail

    def _r(self, c):
        self.calls.append(c)
        code = self.fail(c)
        return res(OK if code is None else code, Q if c["kind"] == "tool" else None)

    def move_tool(self, arm, xyz, quat=None, via=(), cancelled=lambda: False, waist_mode="", waist_yaw=0.0,
                  lookahead=()):
        return self._r({"kind": "tool", "arm": arm, "xyz": tuple(xyz), "quat": quat, "via": list(via),
                        "lookahead": list(lookahead), "waist_mode": waist_mode, "waist_yaw": waist_yaw})

    def move_named(self, arm, name, cancelled=lambda: False, waist_mode=""):
        return self._r({"kind": "named", "arm": arm, "name": name, "waist_mode": waist_mode})

    def gripper(self, arm, close, cancelled=lambda: False):
        return self._r({"kind": "gripper", "arm": arm, "close": close})


def ctx_for(bag_id="Bag_01"):
    bag = CFG.bags.get(bag_id)
    target, lane = CFG.sort.target_for(bag.label)
    return plan_task(CFG, bag, "left", target, lane)


def runner(mc, **kw):
    return MotionRunner(mc, cancelled=lambda: False, sleep=lambda s: None, **kw)


def test_interface_codes_match_movearm_action():
    text = (ROOT / "g2cell_interfaces" / "action" / "MoveArm.action").read_text()
    codes = {m.group(1): int(m.group(2)) for m in re.finditer(r"int8 (\w+)=(\d+)", text)}
    assert (codes["OK"], codes["IK_FAILED"], codes["HEAD_CLEARANCE"], codes["BUSY"], codes["CANCELED"]) == \
        (OK, IK_FAILED, HEAD_CLEARANCE, BUSY, CANCELED)
    assert "geometry_msgs/Pose[] lookahead" in text


def test_plan_task_waypoints_and_waists():
    c = ctx_for("Bag_01")
    assert c["arm"] == "left" and c["target"] == "crate"
    assert c["waist_pick"] == ("waist_first", 0.0) and c["waist_place"] == ("absolute", 0.0)
    assert c["pick"]["grasp"][2] < c["pick"]["approach"][2] < c["pick"]["lift"][2]
    assert c["place"]["above"][:2] == CFG.targets.crate_xy


def test_approach_plans_the_whole_pick_column():
    mc = FakeMC()
    c = ctx_for()
    assert runner(mc).approach(c).ok
    mv = [x for x in mc.calls if x["kind"] == "tool"][0]
    assert mv["xyz"] == c["pick"]["approach"] and mv["quat"] is None and mv["waist_mode"] == "waist_first"
    assert mv["lookahead"] == [c["pick"]["grasp"], c["pick"]["lift"]]
    assert c["quat"] == Q


def test_grasp_lift_hold_waist_and_orientation():
    mc = FakeMC()
    c = ctx_for()
    r = runner(mc)
    r.approach(c)
    mc.calls.clear()
    assert r.grasp(c).ok and r.lift(c).ok
    tools = [x for x in mc.calls if x["kind"] == "tool"]
    assert [t["xyz"] for t in tools] == [c["pick"]["grasp"], c["pick"]["lift"]]
    assert all(t["waist_mode"] == "fixed" and t["quat"] == Q for t in tools)   # no torso swing over the bag


def test_transfer_via_lookahead_and_reorient_retry():
    n = {"k": 0}

    def fail(call):
        if call["kind"] == "tool" and call["xyz"] == ctx_for()["place"]["above"]:
            n["k"] += 1
            return IK_FAILED if n["k"] == 1 else None
        return None

    mc = FakeMC(fail)
    c = ctx_for()
    c["quat"] = (0.0, 1.0, 0.0, 0.0)
    assert runner(mc).transfer(c).ok
    t1, t2 = [x for x in mc.calls if x["kind"] == "tool"]
    assert t1["via"] == [c["place"]["via"]] and t1["lookahead"] == [c["place"]["release"], c["place"]["retract"]]
    assert t1["waist_mode"] == "absolute" and t1["waist_yaw"] == 0.0
    assert t1["quat"] == (0.0, 1.0, 0.0, 0.0) and t2["quat"] is None          # second try searches again


def test_transfer_falls_back_to_waist_first_toward_the_target():
    """Bag_02: from its elbow-up pick posture the transfer at waist 0 passes 0.198 m from the head; letting the
    motion server choose the smallest waist turn that faces the carton (waist_first) clears it."""
    above = ctx_for("Bag_02")["place"]["above"]
    mc = FakeMC(lambda c: HEAD_CLEARANCE if c["kind"] == "tool" and c["xyz"] == above
                and c["waist_mode"] == "absolute" else None)
    c = ctx_for("Bag_02")
    c["quat"] = Q
    assert runner(mc).transfer(c).ok
    modes = [(t["waist_mode"], t["quat"]) for t in mc.calls if t["kind"] == "tool"]
    assert modes == [("absolute", Q), ("absolute", None), ("waist_first", None)]
    last = [t for t in mc.calls if t["kind"] == "tool"][-1]
    assert last["via"] == [c["place"]["via"]] and last["lookahead"] == [c["place"]["release"], c["place"]["retract"]]


def test_transfer_no_fallback_after_cancel():
    mc = FakeMC(lambda c: CANCELED if c["kind"] == "tool" else None)
    c = ctx_for()
    r = runner(mc).transfer(c)
    assert not r.ok and r.canceled and len([t for t in mc.calls if t["kind"] == "tool"]) == 1


def test_place_and_retract_hold_the_waist_then_home():
    mc = FakeMC()
    c = ctx_for()
    c["quat"] = Q
    r = runner(mc)
    assert r.place(c).ok and r.retract(c).ok
    tools = [x for x in mc.calls if x["kind"] == "tool"]
    assert all(t["waist_mode"] == "fixed" for t in tools)
    assert mc.calls[-1] == {"kind": "named", "arm": "left", "name": "home", "waist_mode": ""}


def test_recover_lifts_clear_then_goes_home():
    mc = FakeMC()
    c = ctx_for()
    r = runner(mc)
    r.approach(c), r.grasp(c), r.lift(c)
    mc.calls.clear()
    rec = r.recover(c)
    assert rec.ok
    assert mc.calls[0] == {"kind": "gripper", "arm": "left", "close": False}
    up = mc.calls[1]
    lx, ly, lz = c["pick"]["lift"]
    assert up["kind"] == "tool" and up["waist_mode"] == "fixed" and up["quat"] == Q
    assert up["xyz"][:2] == (lx, ly) and up["xyz"][2] > lz
    assert mc.calls[-1]["kind"] == "named" and mc.calls[-1]["name"] == "home" and mc.calls[-1]["waist_mode"] == ""


def test_recover_goes_home_even_if_gripper_and_lift_fail():
    mc = FakeMC(lambda c: IK_FAILED if c["kind"] in ("gripper", "tool") else None)
    c = ctx_for()
    c.update(at=c["pick"]["lift"], at_quat=Q)
    rec = runner(mc).recover(c)
    assert rec.ok and mc.calls[-1]["name"] == "home"
    assert "open gripper" in rec.message


def test_recover_retries_busy_server():
    """A MoveArm client timeout cancels the goal while motion_server may still be planning/moving it; the
    recovery's first goals then come back BUSY. They must be retried, not abandon the arm where it is."""
    n = {"busy": 3}

    def fail(call):
        if n["busy"] > 0:
            n["busy"] -= 1
            return BUSY
        return None

    mc = FakeMC(fail)
    rec = runner(mc).recover(ctx_for())
    assert rec.ok and mc.calls[-1]["name"] == "home"
    assert sum(1 for x in mc.calls if x["kind"] == "gripper") == 4


def test_recover_falls_back_to_arm_first_home():
    """home turns the waist and the arm on one segment; if that path is refused (head clearance), the arm
    folds home with the waist held, then the waist turns back."""
    def fail(call):
        if call["kind"] == "named" and call["waist_mode"] == "" and not fail.folded:
            return HEAD_CLEARANCE
        if call["kind"] == "named" and call["waist_mode"] == "fixed":
            fail.folded = True
        return None
    fail.folded = False
    mc = FakeMC(fail)
    rec = runner(mc).recover(ctx_for())
    named = [(x["name"], x["waist_mode"]) for x in mc.calls if x["kind"] == "named"]
    assert rec.ok and named == [("home", ""), ("home", "fixed"), ("home", "")]


def test_recover_reports_failure_when_home_unreachable():
    mc = FakeMC(lambda c: HEAD_CLEARANCE if c["kind"] == "named" else None)
    rec = runner(mc).recover(ctx_for())
    assert not rec.ok and "home" in rec.message


def test_machine_transfer_failure_ends_at_home():
    """The live A6000 failure (transfer ik_failed): plan_failed, and the last command is home."""
    mc = FakeMC(lambda c: IK_FAILED if c["kind"] == "tool" and c["via"] else None)
    c = ctx_for()
    tr = PickPlaceMachine(runner(mc), Hooks(), 2).run(c)
    assert tr.outcome == "plan_failed" and tr.failed_stage == "transfer"
    assert "recovery failed" not in tr.message
    assert mc.calls[-1]["kind"] == "named" and mc.calls[-1]["name"] == "home"


def test_bag03_is_unreachable_without_planning():
    bag = CFG.bags.get("Bag_03")
    target, lane = CFG.sort.target_for(bag.label)
    ch = select_arm(CFG.reach, "table", bag.xy, target, "", lane=lane)
    assert ch.arm is None and "reaches Y <=" in ch.reason
    for b in ("Bag_01", "Bag_02"):
        bb = CFG.bags.get(b)
        assert select_arm(CFG.reach, "table", bb.xy, "crate", "", lane="").arm == "left"


# ---- approach from home: lift clear of the pack carton first (datagen 2026-10-11: home holds the hand 0.10 m
# over the carton and the waist-turning approach swept the gripper through its +Y wall) -----------------------
def test_approach_from_home_goes_to_the_clear_pose_first_with_the_waist_held():
    mc = FakeMC()
    run = runner(mc, clear_pose="clear")
    ctx = ctx_for()
    assert run.approach(ctx).ok
    kinds = [(c["kind"], c.get("name")) for c in mc.calls]
    assert kinds[:3] == [("gripper", None), ("named", "clear"), ("tool", None)]
    assert mc.calls[1]["waist_mode"] == "fixed"


def test_approach_skips_the_clear_pose_when_not_starting_from_home():
    mc = FakeMC()
    run = runner(mc, clear_pose="clear")
    ctx = ctx_for()
    ctx["at"] = (0.0, 0.6, 1.1)          # e.g. a second grasp attempt: the hand is already over the table
    assert run.approach(ctx).ok
    assert not any(c["kind"] == "named" for c in mc.calls)


def test_approach_without_clear_pose_is_unchanged():
    mc = FakeMC()
    assert runner(mc).approach(ctx_for()).ok
    assert [c["kind"] for c in mc.calls] == ["gripper", "tool"]


def test_clear_pose_failure_fails_the_approach():
    mc = FakeMC(fail=lambda c: HEAD_CLEARANCE if c.get("name") == "clear" else None)
    r = runner(mc, clear_pose="clear").approach(ctx_for())
    assert not r.ok and "clear" in r.message
    assert not any(c["kind"] == "tool" for c in mc.calls)


def test_clear_pose_is_defined_for_the_left_arm():
    from g2cell_motion.named_poses import load_named_poses, resolve
    doc = yaml.safe_load((ROOT / "g2cell_motion" / "config" / "named_poses.yaml").read_text())
    t = resolve(load_named_poses(doc), "clear", "left")
    assert t.kind == "tool" and t.xyz[2] >= 1.25 and t.yaw_deg is None   # high, gripper-down orientation searched
