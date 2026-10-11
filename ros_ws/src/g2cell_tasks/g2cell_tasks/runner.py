"""PickBag stages on top of g2cell_motion's MoveArm (no ROS imports: the MotionClient is injected, results are
duck-typed MoveArm results), and plan_task(), the per-pick context.

Motion rules (docs/OPTIMISATION.md "Motion planning" -> "PickBag transfer fix"):
- approach: gripper-down orientation searched with the waist turned toward the table (waist_first), and the
  grasp and lift points below/above it as MoveArm `lookahead`, so the approach never ends in an orientation
  or torso yaw from which the descent has no IK (Bag_02 before the fix).
- clear (datagen 2026-10-11): from home (ctx "at" unknown) the arm first moves to the named joint pose
  `clear_pose` with the waist held: home keeps the hand only ~0.10 m over the pack carton, and the
  waist-turning approach swept the gripper through the carton's +Y wall (the carton was thrown 0.5 m).
- grasp, lift, place, retract, back-off: waist held (`fixed`) and the stored orientation, so the torso does
  not swing while the hand is at the bag or over the carton (before: the waist swung 78 -> 48 -> 78 deg).
- transfer: via point + above the target, waist toward the target (`absolute` 0 for the carton; motion_server
  turns the via points part way, see planner.waist_profiles), release and retract as lookahead. Retried with
  the orientation searched again, then with waist_first (smallest turn facing the target; Bag_02).
- recover: always tries to end at home: open the gripper, back straight up from the last pose, home (waist and
  arm together); if that is refused, fold the arm home with the waist held and then turn the waist back.
  BUSY / no answer (e.g. after a client timeout while motion_server still plans the canceled goal) is retried.
"""
from __future__ import annotations

import time

from g2cell_tasks.reachability import waist_for
from g2cell_tasks.state_machine import StepResult
from g2cell_tasks.waypoints import pick_waypoints, place_waypoints

# g2cell_interfaces/action/MoveArm.action result codes (test_runner checks them against the file)
OK, IK_FAILED, HEAD_CLEARANCE, BUSY, CANCELED = 0, 2, 3, 4, 5
HOLD = ("fixed", 0.0)
WAIST_FIRST = "waist_first"
RECOVER_LIFT = 0.08      # m straight up from the last pose before going home


def ok(r) -> bool:
    return r is not None and r.error_code == OK


def quat_of(r):
    if r is None or getattr(r, "orientation_used", None) is None:
        return None
    o = r.orientation_used
    q = (o.x, o.y, o.z, o.w)
    return None if all(abs(c) < 1e-12 for c in q) else tuple(q)


def step(r, what: str) -> StepResult:
    if r is None:
        return StepResult(False, f"{what}: no answer from motion_server (rejected or timed out)")
    if r.error_code == CANCELED:
        return StepResult(False, f"{what}: canceled", canceled=True)
    if not ok(r):
        return StepResult(False, f"{what}: {r.message} (code {r.error_code})")
    return StepResult(True)


def plan_task(cfg, bag, arm: str, target: str, lane: str) -> dict:
    """Context for one pick: arm, waypoints, waist plan per side, orientation slot."""
    pick = pick_waypoints(bag.xy, bag.top_z)
    return {"arm": arm, "bag": bag, "target": target, "lane": lane, "pick": pick,
            "place": place_waypoints(bag.xy, pick["lift"][2], cfg.targets.point(target, lane)), "quat": None,
            "waist_pick": waist_for("table", cfg.reach.waist_first),
            "waist_place": waist_for(target, cfg.reach.waist_first)}


class MotionRunner:
    """Stage implementations for state_machine.PickPlaceMachine. ctx (plan_task) carries arm, waypoints, the
    orientation found at approach / transfer ("quat"), and the last commanded tool point and orientation
    ("at", "at_quat") for recovery."""

    def __init__(self, mc, cancelled=lambda: False, sleep=time.sleep, busy_retries: int = 10,
                 busy_wait: float = 1.0, clear_pose: str = ""):
        self.mc, self.cancelled, self.sleep = mc, cancelled, sleep
        self.clear_pose = clear_pose
        self.busy_retries, self.busy_wait = busy_retries, busy_wait

    def _chain(self, *calls) -> StepResult:
        for what, fn in calls:
            if self.cancelled():
                return StepResult(False, f"{what}: canceled", canceled=True)
            r = step(fn(), what)
            if not r.ok:
                return r
        return StepResult(True)

    def _move(self, ctx, xyz, via=(), ahead=(), where="hold", waist=None):
        """where: 'pick' / 'place' (ctx waist plan toward that side) or 'hold' (keep the waist);
        waist=(mode, yaw) overrides both."""
        mode, yaw = waist or (HOLD if where == "hold" else ctx.get(f"waist_{where}", ("", 0.0)))
        r = self.mc.move_tool(ctx["arm"], xyz, quat=ctx.get("quat"), via=via, cancelled=self.cancelled,
                              waist_mode=mode, waist_yaw=yaw, lookahead=ahead)
        if ok(r):
            if ctx.get("quat") is None:
                ctx["quat"] = quat_of(r)
            ctx["at"], ctx["at_quat"] = tuple(xyz), quat_of(r) or ctx.get("quat")
        return r

    # ---- stages ---------------------------------------------------------------------------
    def approach(self, ctx):
        a, p = ctx["arm"], ctx["pick"]
        ctx["quat"] = None   # search orientations again on every attempt
        clear = []
        if self.clear_pose and ctx.get("at") is None:   # starting from home (or after a recovery to home)
            clear = [(self.clear_pose, lambda: self.mc.move_named(a, self.clear_pose, cancelled=self.cancelled,
                                                                  waist_mode=HOLD[0]))]
        return self._chain(("open gripper", lambda: self.mc.gripper(a, close=False, cancelled=self.cancelled)),
                           *clear,
                           ("approach", lambda: self._move(ctx, p["approach"], ahead=[p["grasp"], p["lift"]],
                                                           where="pick")))

    def grasp(self, ctx):
        a = ctx["arm"]
        return self._chain(("descend", lambda: self._move(ctx, ctx["pick"]["grasp"])),
                           ("close gripper", lambda: self.mc.gripper(a, close=True, cancelled=self.cancelled)))

    def lift(self, ctx):
        return self._chain(("lift", lambda: self._move(ctx, ctx["pick"]["lift"])))

    def transfer(self, ctx):
        """1. ctx waist plan + the pick orientation, 2. orientation searched again, 3. (if the plan was not
        already waist_first) waist_first: the smallest waist turn that faces the target."""
        w = ctx["place"]
        kw = {"via": [w["via"]], "ahead": [w["release"], w["retract"]]}
        plan = ctx.get("waist_place", ("", 0.0))
        tries = [(plan, False), (plan, True)] + ([] if plan[0] == WAIST_FIRST else [((WAIST_FIRST, 0.0), True)])
        r = None
        for waist, search in tries:
            if search:
                ctx["quat"] = None   # the pick orientation does not work over the target: search again
            elif ctx.get("quat") is None:
                continue             # nothing to keep: the next try searches anyway
            r = self._move(ctx, w["above"], waist=waist, **kw)
            if r is None or ok(r) or r.error_code == CANCELED:
                break
        return step(r, "transfer")

    def place(self, ctx):
        a = ctx["arm"]
        return self._chain(("lower", lambda: self._move(ctx, ctx["place"]["release"])),
                           ("open gripper", lambda: self.mc.gripper(a, close=False, cancelled=self.cancelled)))

    def retract(self, ctx):
        a = ctx["arm"]
        return self._chain(("retract", lambda: self._move(ctx, ctx["place"]["retract"])),
                           ("home", lambda: self.mc.move_named(a, "home", cancelled=self.cancelled)))

    def release(self, ctx):
        a = ctx["arm"]
        return self._chain(("open gripper", lambda: self.mc.gripper(a, close=False, cancelled=self.cancelled)),
                           ("back off", lambda: self._move(ctx, ctx["pick"]["approach"])))

    # ---- recovery -------------------------------------------------------------------------
    def _retry(self, fn):
        """fn() again while motion_server is BUSY or does not answer (bounded)."""
        r = fn()
        for _ in range(self.busy_retries):
            if r is not None and r.error_code != BUSY:
                break
            self.sleep(self.busy_wait)
            r = fn()
        return r

    def recover(self, ctx):
        """Open, back straight up, home. Not cancel-aware on purpose (it is the safe exit). ok = arm at home."""
        a, notes = ctx["arm"], []
        r = self._retry(lambda: self.mc.gripper(a, close=False))
        if not ok(r):
            notes.append(step(r, "open gripper").message)
        at, q = ctx.get("at"), ctx.get("at_quat")
        if at is not None and q is not None:
            up = (at[0], at[1], at[2] + RECOVER_LIFT)
            r = self._retry(lambda: self.mc.move_tool(a, up, quat=q, waist_mode=HOLD[0]))
            if not ok(r):
                notes.append(step(r, "back off up").message + " (tolerated)")
        r = self._retry(lambda: self.mc.move_named(a, "home"))
        if not ok(r):
            notes.append(step(r, "home").message)
            r = self._retry(lambda: self.mc.move_named(a, "home", waist_mode=HOLD[0]))
            if ok(r):
                r = self._retry(lambda: self.mc.move_named(a, "home"))
            if not ok(r):
                notes.append(step(r, "home (arm first, waist held)").message)
                return StepResult(False, "; ".join(notes))
        ctx["at"] = ctx["at_quat"] = None
        return StepResult(True, "; ".join(notes))
