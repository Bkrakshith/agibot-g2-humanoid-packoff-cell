"""Waypoint planning over gripper orientations (no ROS imports; IK and clearance are injected).

The logic g2cell_demo used: for each candidate orientation, IK every waypoint of the chain
(each seeded with the previous solution so the arm stays on one IK branch); the first
orientation whose whole chain solves AND whose joint path start -> solutions -> end stays
min_clearance away from the head wins.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from g2cell_motion.geometry import down_quat, end_link_for_tool
from g2cell_motion.robot import HEAD_CLEARANCE
from g2cell_motion.waist_policy import (TwistLimits, WaistConfig, best_solution, choose_waist_yaw, clamp_waist,
                                        synced_chain, waist_candidates)

WAIST_FIRST, WAIST_FIXED, WAIST_ABSOLUTE = "waist_first", "fixed", "absolute"
WAIST_MODES = (WAIST_FIRST, WAIST_FIXED, WAIST_ABSOLUTE)

YAWS = (0, 90, -90, 45, -45, 135, -135, 180)
TILTS = (0, 20, 40)
ORIENTS = [(y, t) for t in TILTS for y in YAWS]   # degrees, same order as the Stage 2 demo


def ik_seed_sequence(base: dict, names, attempts: int, jitter: float, rng_seed: int = 0) -> list[dict]:
    """IK seeds: `base` first, then attempts-1 copies with the arm joints jittered +-jitter. A private
    fixed-seed RNG (never the global one), so the same inputs always give the same seeds."""
    if attempts <= 0:
        return []
    rng = random.Random(rng_seed)
    out = [dict(base)]
    for _ in range(max(0, attempts - 1)):
        s = dict(base)
        for n in names:
            s[n] = s[n] + rng.uniform(-jitter, jitter)
        out.append(s)
    return out


class PlanBudget:
    """Bounded planning effort: at most max_solves IK calls and max_seconds wall time (None = no bound).
    spend() is called before every IK call; False = stop planning now."""

    def __init__(self, max_solves: int | None = None, max_seconds: float | None = None, clock=time.monotonic):
        self.max_solves, self.max_seconds, self.clock = max_solves, max_seconds, clock
        self.t0 = clock()
        self.solves = 0
        self.exhausted = False

    def spend(self) -> bool:
        self.solves += 1
        if (self.max_solves is not None and self.solves > self.max_solves) or \
                (self.max_seconds is not None and self.clock() - self.t0 > self.max_seconds):
            self.exhausted = True
        return not self.exhausted


@dataclass
class PlanResult:
    ok: bool
    solutions: list = field(default_factory=list)    # one {joint: rad} per waypoint (arm joints only)
    label: object = None                               # (yaw_deg, tilt_deg) or "fixed"
    quat: tuple | None = None                          # end-link orientation in base_link
    clearance: float = -1.0
    reason: str = ""                                   # "" | ik_failed | head_clearance | twist
    waist_yaws: list = field(default_factory=list)    # waist yaw per waypoint (waist-aware planning only)
    base_turn: float = 0.0                             # base rotation the waist could not cover (rad)
    needs_base: bool = False
    twist_rejected: int = 0                            # IK solutions dropped by the backward-twist filter


def orient_candidates(prefer=None, fixed=None, yaw_offset: float = 0.0):
    """[(label, quat)]: a fixed quaternion alone, else `prefer` (yaw, tilt deg) then ORIENTS.
    yaw_offset (rad) turns every searched gripper yaw, e.g. by the waist yaw so that label yaw 0
    means "along the torso's forward direction"; labels stay relative."""
    if fixed is not None:
        return [("fixed", tuple(fixed))]
    order = ([tuple(prefer)] if prefer else []) + [o for o in ORIENTS if o != tuple(prefer or ())]
    return [(o, down_quat(math.radians(o[0]) + yaw_offset, math.radians(o[1]))) for o in order]


def plan_chain(points, solve, clearance, seed: dict, joint_names, orients=None, start=None, end=None,
               min_clearance: float = HEAD_CLEARANCE, tool_frame: bool = True, log=None) -> PlanResult:
    """points: fingertip points in the cell frame (tool_frame) or end-link points in base_link.
    solve(end_link_pos, quat, seed) -> {joint: rad} | None.
    clearance(path) -> metres, path = [start?] + solutions + [end?] restricted to joint_names."""
    orients = orients if orients is not None else orient_candidates()
    best_clear, any_ik = -1.0, False
    for label, q in orients:
        sols, s = [], dict(seed)
        for p in points:
            target = end_link_for_tool(p, q) if tool_frame else tuple(p)
            sol = solve(target, q, s)
            if sol is None:
                break
            sols.append({n: sol[n] for n in joint_names if n in sol})
            s.update(sol)
        if len(sols) != len(points):
            continue
        any_ik = True
        path = ([start] if start else []) + sols + ([end] if end else [])
        path = [{n: p[n] for n in joint_names if n in p} for p in path]
        c = clearance(path)
        best_clear = max(best_clear, c)
        if c < min_clearance:
            if log:
                log(f"orientation {label}: path {c:.2f} m from head, rejected")
            continue
        return PlanResult(True, sols, label, tuple(q), c)
    return PlanResult(False, reason="head_clearance" if any_ik else "ik_failed", clearance=best_clear)


def waist_profiles(points, waist_now: float, mode: str, cfg: WaistConfig, yaw=None, fallback_step_deg: float = 30.0):
    """Candidate waist-yaw sequences (one yaw per waypoint) and the policy decision for the last point.
    waist_first: the policy per waypoint (each from the previous yaw), then whole-chain fallbacks
    around the final choice, then no turn. fixed: the current yaw. absolute: `yaw` (clamped) at the last
    point; with via points first a profile that turns evenly along the chain (current -> yaw), then
    every point at `yaw`, then the via points at the current yaw."""
    n = len(points)
    if mode == WAIST_FIXED:
        return [[waist_now] * n], None
    if mode == WAIST_ABSOLUTE:
        tgt = clamp_waist(float(yaw), cfg)
        now = clamp_waist(waist_now, cfg)
        if n <= 1 or abs(tgt - now) < 1e-9:
            return [[tgt] * n], None
        # via points turn on the way (PickBag transfer: table waist ~78 deg -> carton 0 deg; the midway via
        # point is out of reach with the torso already straight), then all-at-target, then turn last
        interp = [now + (tgt - now) * (i + 1) / n for i in range(n - 1)] + [tgt]
        profiles = [interp]
        for alt in ([tgt] * n, [now] * (n - 1) + [tgt]):
            if alt not in profiles:
                profiles.append(alt)
        return profiles, None
    if mode != WAIST_FIRST:
        raise ValueError(f"waist mode must be one of {WAIST_MODES}, got {mode!r}")
    seq, w, dec = [], waist_now, None
    for p in points:
        dec = choose_waist_yaw(p, w, cfg)
        w = dec.yaw
        seq.append(w)
    profiles = [seq]
    for alt in waist_candidates(seq[-1], waist_now, cfg, step_deg=fallback_step_deg)[1:]:
        if [alt] * n not in profiles:
            profiles.append([alt] * n)
    return profiles, dec


def plan_chain_waist(points, solve, clearance, seed: dict, joint_names, home: dict, waist_now: float,
                     mode: str = WAIST_FIRST, waist_yaw=None, cfg: WaistConfig = WaistConfig(),
                     limits: TwistLimits = TwistLimits(), prefer=None, fixed=None, start=None, end=None,
                     min_clearance: float = HEAD_CLEARANCE, tool_frame: bool = True, log=None,
                     fallback_step_deg: float = 30.0, fallback_orients: int = 8,
                     budget: PlanBudget | None = None, lookahead=()) -> PlanResult:
    """Waist-aware plan_chain. For each waist profile (see waist_profiles) and each orientation, every
    waypoint is solved from several seeds (the previous solution and the forward home posture) with the
    waist joint set to that waypoint's yaw, so the 7-DoF IK works in the turned torso frame. Solutions
    that twist backwards are dropped (waist_policy.twist_violations), the best-scoring one is kept.
    The joint path start -> waypoints (waist included) is then checked for head clearance.
    Fallback waist profiles (after the policy's) try only the first `fallback_orients` orientations
    (0 = all) and the previous-solution seed, to bound the number of /compute_ik calls.
    solve(end_link_pos, quat, seed) -> {joint: rad} | None, seed contains cfg.joint.
    clearance(path) -> metres; path dicts contain the arm joints AND cfg.joint.
    budget: optional PlanBudget; when it runs out the search stops with reason "budget".
    lookahead: points the caller will move to NEXT (e.g. grasp and lift below an approach, release below
    the place approach). They must also solve with the same orientation and the final waist yaw, and their
    path is head-checked, but they are not returned: a plan never ends where the next move is impossible.
    If only they fail, the reason is "lookahead_ik_failed"."""
    wj = cfg.joint
    profiles, dec = waist_profiles(points, waist_now, mode, cfg, waist_yaw, fallback_step_deg)
    base_turn = dec.base_turn if dec else 0.0
    needs_base = bool(dec.needs_base) if dec else False
    if needs_base and log:
        log(f"target beyond the waist: base must turn {math.degrees(base_turn):.0f} deg (not driven)")
    names_w = list(joint_names) + [wj]
    start_w = None
    if start is not None:
        start_w = {**{n: start[n] for n in joint_names if n in start}, wj: start.get(wj, waist_now)}
    best_clear, any_ik, twisted, ahead_failed = -1.0, False, 0, False
    arm_seed = {n: seed[n] for n in joint_names if n in seed}
    home_seed = {n: home[n] for n in joint_names if n in home}
    for pi, yaws in enumerate(profiles):
        offset = yaws[-1] if mode != WAIST_FIXED else 0.0
        orients = orient_candidates(prefer=prefer, fixed=fixed, yaw_offset=offset)
        if pi and fallback_orients:
            orients = orients[:fallback_orients]
        for label, q in orients:
            sols, prev = [], dict(arm_seed)
            for p, yw in zip(points, yaws):
                target = end_link_for_tool(p, q) if tool_frame else tuple(p)
                seeds = [prev] + ([home_seed] if home_seed and home_seed != prev and not pi else [])
                cands = []
                for sd in seeds:
                    if budget is not None and not budget.spend():
                        return PlanResult(False, reason="budget", clearance=best_clear, base_turn=base_turn,
                                          needs_base=needs_base, twist_rejected=twisted)
                    sol = solve(target, q, {**seed, **sd, wj: yw})
                    cands.append(None if sol is None else {n: sol[n] for n in joint_names if n in sol})
                best, rej = best_solution(cands, prev, home, list(joint_names), limits)
                twisted += len(rej)
                if best is None:
                    break
                sols.append(best)
                prev = best
            if len(sols) != len(points):
                continue
            ahead = []
            for p in lookahead:
                target = end_link_for_tool(p, q) if tool_frame else tuple(p)
                seeds = [prev] + ([home_seed] if home_seed and home_seed != prev and not pi else [])
                cands = []
                for sd in seeds:
                    if budget is not None and not budget.spend():
                        return PlanResult(False, reason="budget", clearance=best_clear, base_turn=base_turn,
                                          needs_base=needs_base, twist_rejected=twisted)
                    sol = solve(target, q, {**seed, **sd, wj: yaws[-1]})
                    cands.append(None if sol is None else {n: sol[n] for n in joint_names if n in sol})
                best, rej = best_solution(cands, prev, home, list(joint_names), limits)
                twisted += len(rej)
                if best is None:
                    break
                ahead.append(best)
                prev = best
            if len(ahead) != len(lookahead):
                ahead_failed = True
                continue
            any_ik = True
            chain = synced_chain(sols, yaws, wj)
            path = ([start_w] if start_w else []) + chain + synced_chain(ahead, [yaws[-1]] * len(ahead), wj) \
                + ([end] if end else [])
            path = [{n: c[n] for n in names_w if n in c} for c in path]
            c = clearance(path)
            best_clear = max(best_clear, c)
            if c < min_clearance:
                if log:
                    log(f"waist {[round(math.degrees(y)) for y in yaws]} deg, orientation {label}: "
                        f"path {c:.2f} m from head, rejected")
                continue
            return PlanResult(True, chain, label, tuple(q), c, waist_yaws=list(yaws), base_turn=base_turn,
                              needs_base=needs_base, twist_rejected=twisted)
    reason = "head_clearance" if any_ik else ("lookahead_ik_failed" if ahead_failed else
                                              ("twist" if twisted else "ik_failed"))
    return PlanResult(False, reason=reason, clearance=best_clear, base_turn=base_turn, needs_base=needs_base,
                      twist_rejected=twisted)
