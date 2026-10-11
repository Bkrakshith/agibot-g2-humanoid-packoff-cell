"""Waist-first turning policy for the G2 (no ROS imports, unit-tested).

User rule: to face a target, ROTATE AT THE WAIST (idx05_body_joint5, yaw about body_link4 Z) so the
arms work close to forward relative to the torso. Use the wheels only for what the waist cannot
cover, and never accept IK solutions that twist an arm backwards.

- choose_waist_yaw: the smallest waist turn that brings the target's bearing (seen from the waist
  pivot, relative to the torso) inside +-cone_deg, clamped to the joint limits and to +-max_turn_deg.
  Whatever is left over is returned as a base rotation and flagged (needs_base).
- twist_violations / solution_score / best_solution: IK-solution filter and ranking. Rejects
  arm joint1 (shoulder) more than j1_max_dev_deg from home and jumps of joints 3/5/7 (forearm /
  wrist rolls) larger than jump_max from the seed; ranks valid ones by distance to the forward home
  posture (plus a small pull toward the seed for continuity).
- synced_chain: adds the waist yaw to every waypoint so the streamer interpolates waist and arm on
  the same quintic profile (synchronised, smooth).

Frames: base_link (+X forward, +Y left). Waist yaw > 0 turns the torso to the left (+Y).
The waist axis is tilted 20 deg forward by idx03_body_joint3 (scene init); the default pivot is
where that axis passes the arm base (arm_base_link origin, x 0.102 + 0.4575 sin 20deg = 0.258).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

WAIST_JOINT = "idx05_body_joint5"
WAIST_LIMIT = 3.0456          # sim URDF; real G2 soft limit is 3.046 -> use the tighter one


@dataclass(frozen=True)
class WaistConfig:
    cone_deg: float = 30.0                  # target bearing allowed relative to the torso
    max_turn_deg: float | None = 100.0      # |waist yaw| bound (None = joint limits only)
    lower: float = -WAIST_LIMIT
    upper: float = WAIST_LIMIT
    pivot: tuple = (0.258, 0.0)             # waist axis at arm-base height, base_link xy
    min_radius: float = 0.05                # targets closer to the pivot never turn the waist
    joint: str = WAIST_JOINT


@dataclass(frozen=True)
class TwistLimits:
    j1_max_dev_deg: float = 90.0            # shoulder (arm joint1) from its home value
    jump_max: float = 1.5                   # rad, joints 3/5/7 vs the seed
    jump_joints: tuple = (3, 5, 7)


@dataclass(frozen=True)
class WaistDecision:
    yaw: float              # chosen waist yaw, rad
    current: float          # waist yaw before the move
    bearing: float          # target bearing in base_link (from the pivot)
    rel_bearing: float      # target bearing relative to the torso after the turn
    base_turn: float        # base rotation still needed (rad, same sign convention), 0 if covered
    needs_base: bool
    reason: str = ""

    @property
    def turn(self) -> float:
        return self.yaw - self.current


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def bearing(xy, pivot=(0.0, 0.0)) -> float:
    return math.atan2(xy[1] - pivot[1], xy[0] - pivot[0])


def waist_bounds(cfg: WaistConfig) -> tuple[float, float]:
    lo, hi = cfg.lower, cfg.upper
    if cfg.max_turn_deg is not None:
        m = math.radians(cfg.max_turn_deg)
        lo, hi = max(lo, -m), min(hi, m)
    return lo, hi


def clamp_waist(yaw: float, cfg: WaistConfig) -> float:
    lo, hi = waist_bounds(cfg)
    return min(max(yaw, lo), hi)


def choose_waist_yaw(target, current_yaw: float, cfg: WaistConfig = WaistConfig()) -> WaistDecision:
    """target: (x, y[, z]) in base_link. Turn as little as possible so the target lies within
    +-cone of the torso's forward direction; prefer the waist, report the rest as base rotation."""
    cone = math.radians(cfg.cone_deg)
    r = math.hypot(target[0] - cfg.pivot[0], target[1] - cfg.pivot[1])
    cur = clamp_waist(current_yaw, cfg)
    if r < cfg.min_radius:
        return WaistDecision(cur, current_yaw, 0.0, 0.0, 0.0, False, "target at the pivot")
    b = bearing(target, cfg.pivot)
    rel = wrap(b - cur)
    if abs(rel) <= cone + 1e-12:
        yaw, why = cur, "inside cone"
    else:
        yaw, why = wrap(b - math.copysign(cone, rel)), "turned to cone edge"
        # take the turn direction toward the target, not the wrapped short-cut through the back
        if abs(yaw - cur) > math.pi:
            yaw += math.copysign(2 * math.pi, cur - yaw)
        yaw = clamp_waist(yaw, cfg)
    rel_after = wrap(b - yaw)
    excess = abs(rel_after) - cone
    base_turn = math.copysign(excess, rel_after) if excess > 1e-9 else 0.0
    if base_turn:
        why = f"waist at limit ({math.degrees(yaw):.0f} deg); base must turn {math.degrees(base_turn):.0f} deg"
    return WaistDecision(yaw, current_yaw, b, rel_after, base_turn, bool(base_turn), why)


def waist_candidates(chosen: float, current: float, cfg: WaistConfig = WaistConfig(),
                     step_deg: float = 15.0, count: int = 1) -> list[float]:
    """Fallback waist yaws for planning, in order: the policy's choice, then steps of step_deg back
    toward the current yaw (the offline reach sweep shows a smaller turn often reaches further, e.g.
    the far half of the pick table), then `count` steps past the choice, then the current yaw (no turn).
    Clamped and de-duplicated."""
    step = math.radians(step_deg)
    toward = math.copysign(1.0, current - chosen) if abs(current - chosen) > 1e-9 else 1.0
    raw = [chosen]
    k = 1
    while abs(current - chosen) > 1e-9 and k * step < abs(current - chosen) - 1e-9:
        raw.append(chosen + toward * k * step)
        k += 1
    raw += [chosen - toward * k * step for k in range(1, count + 1)]
    raw.append(current)
    out = []
    for v in raw:
        v = clamp_waist(v, cfg)
        if all(abs(v - o) > 1e-6 for o in out):
            out.append(v)
    return out


# ---- IK solution filter -----------------------------------------------------------------------
def twist_violations(sol: dict, seed: dict, home: dict, names, limits: TwistLimits = TwistLimits()) -> list[str]:
    """Reasons a solution twists backwards ([] = acceptable). names: the arm's 7 joints in order."""
    out = []
    j1 = names[0]
    dev = abs(sol[j1] - home.get(j1, 0.0))
    if dev > math.radians(limits.j1_max_dev_deg):
        out.append(f"joint1 {math.degrees(dev):.0f} deg from home (> {limits.j1_max_dev_deg:.0f})")
    for k in limits.jump_joints:
        n = names[k - 1]
        if n in seed and n in sol:
            jump = abs(sol[n] - seed[n])
            if jump > limits.jump_max:
                out.append(f"joint{k} jumps {jump:.2f} rad from the seed (> {limits.jump_max:.2f})")
    return out


def solution_score(sol: dict, home: dict, seed: dict, names, w_home: float = 1.0, w_seed: float = 0.3) -> float:
    """Lower is better: squared joint distance to the forward home posture, plus a lighter pull
    toward the seed (continuity)."""
    h = sum((sol[n] - home.get(n, 0.0)) ** 2 for n in names)
    s = sum((sol[n] - seed[n]) ** 2 for n in names if n in seed)
    return w_home * h + w_seed * s


def best_solution(candidates, seed: dict, home: dict, names, limits: TwistLimits = TwistLimits()):
    """(best valid solution or None, [(solution, reasons)] rejected). None candidates are skipped."""
    rejected, best, best_s = [], None, None
    for c in candidates:
        if c is None:
            continue
        v = twist_violations(c, seed, home, names, limits)
        if v:
            rejected.append((c, v))
            continue
        s = solution_score(c, home, seed, names)
        if best is None or s < best_s:
            best, best_s = c, s
    return best, rejected


# ---- interpolation ----------------------------------------------------------------------------
def synced_chain(solutions, waist_yaws, joint: str = WAIST_JOINT) -> list[dict]:
    """Waypoint dicts with the waist added, so run_segment moves waist and arm on one quintic."""
    if len(solutions) != len(waist_yaws):
        raise ValueError(f"{len(solutions)} waypoints but {len(waist_yaws)} waist yaws")
    return [{**s, joint: float(w)} for s, w in zip(solutions, waist_yaws)]
