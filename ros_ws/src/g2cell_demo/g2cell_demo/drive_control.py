"""Closed-loop straight-line drive for the (non-holonomic) G2 base. No ROS imports, unit-tested.

The G2 chassis is commanded with a body twist (v forward, w yaw rate) on /cmd_twist; GeneralServo turns
it into 4WS wheel commands. Feedback is /odom, which in Genie Sim is the ground-truth base pose.

Phases:
  align   rotate in place toward the line start -> goal (only if the heading error > align_tol)
  track   drive along the line: trapezoidal speed (accel-limited, <= v_max, slowing to v_min near the
          goal), steering w = k_head * heading_err + k_cross * cross_track (cross-track = signed lateral
          offset from the line, so lateral error is driven to zero while there is distance left)
  final   rotate in place to the goal yaw
  done    zero twist
The along-track stop is at |along-track error| < pos_tol; the final pose error is checked by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class DriveParams:
    v_max: float = 0.25          # m/s (task limit 0.3)
    v_min: float = 0.03          # creep speed for the last centimetres
    accel: float = 0.15          # m/s^2, ramp up and down
    w_max: float = 0.4           # rad/s
    w_accel: float = 0.6         # rad/s^2
    k_head: float = 1.5
    k_cross: float = 2.0         # rad/s per m of lateral offset
    k_yaw: float = 1.2
    align_tol: float = math.radians(3.0)
    pos_tol: float = 0.008       # along-track stop
    yaw_tol: float = math.radians(0.7)
    w_min: float = 0.03          # rad/s floor in the final rotation (servo deadband)


@dataclass
class DriveController:
    start: tuple                 # (x, y)
    goal: tuple                  # (x, y, yaw)
    p: DriveParams = field(default_factory=DriveParams)
    phase: str = "align"
    v: float = 0.0
    w: float = 0.0

    def __post_init__(self):
        dx, dy = self.goal[0] - self.start[0], self.goal[1] - self.start[1]
        self.length = math.hypot(dx, dy)
        self.heading = math.atan2(dy, dx) if self.length > 1e-6 else self.goal[2]
        self.ux, self.uy = math.cos(self.heading), math.sin(self.heading)

    def errors(self, pose):
        """(remaining along-track distance, signed cross-track offset (+ = left of the line), heading err)."""
        x, y, yaw = pose
        rx, ry = x - self.goal[0], y - self.goal[1]
        along = -(rx * self.ux + ry * self.uy)          # > 0 while the goal is ahead
        cross = -rx * self.uy + ry * self.ux
        return along, cross, wrap(self.heading - yaw)

    def _ramp(self, cur, target, rate, dt):
        return cur + max(-rate * dt, min(rate * dt, target - cur))

    def step(self, pose, dt: float):
        """Returns the (v, w) command for this control period."""
        along, cross, herr = self.errors(pose)
        p = self.p
        if self.phase == "align":
            if abs(herr) < p.align_tol or self.length < p.pos_tol:
                self.phase = "track"
            else:
                self.v = 0.0
                w = max(-p.w_max, min(p.w_max, p.k_yaw * herr))
                self.w = self._ramp(self.w, w, p.w_accel, dt)
                return self.v, self.w
        if self.phase == "track":
            if along < p.pos_tol:
                self.phase = "final"
                self.v = 0.0
            else:
                # speed that can still stop at the goal with the decel limit, never below v_min
                v_stop = math.sqrt(2 * p.accel * max(0.0, along - p.pos_tol / 2))
                v_tgt = max(p.v_min, min(p.v_max, v_stop))
                self.v = min(self._ramp(self.v, v_tgt, p.accel, dt), v_tgt)
                w = p.k_head * herr - p.k_cross * cross * (1.0 if self.v >= 0 else -1.0)
                # steering authority scales with speed (no spinning on the spot while tracking)
                w_lim = p.w_max * min(1.0, self.v / 0.1)
                self.w = max(-w_lim, min(w_lim, w))
                return self.v, self.w
        if self.phase == "final":
            yerr = wrap(self.goal[2] - pose[2])
            if abs(yerr) < p.yaw_tol:
                self.phase = "done"
            else:
                w = p.k_yaw * yerr
                w = math.copysign(max(p.w_min, min(p.w_max, abs(w))), w)
                self.v, self.w = 0.0, self._ramp(self.w, w, p.w_accel, dt) if abs(self.w) < abs(w) else w
                return self.v, self.w
        self.v, self.w = 0.0, 0.0
        return 0.0, 0.0


def pose_error(pose, goal):
    """(position error m, |yaw error| deg)."""
    return math.hypot(pose[0] - goal[0], pose[1] - goal[1]), abs(math.degrees(wrap(pose[2] - goal[2])))


# ---- clearance of the chassis footprint to the U fixtures (2D, cell frame) ---------------------
# G2 base_link mesh: X -0.374..0.382, Y +-0.318 (robot.urdf base_link.dae)
CHASSIS = (-0.374, 0.382, -0.318, 0.318)


def footprint(pose, box=CHASSIS):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for px in box[:2] for py in box[2:]]


def rect_distance(pts, rect):
    """Min distance from a convex polygon's corners/edges to an axis-aligned rect (x0, x1, y0, y1);
    0 if they overlap (corner-in-rect or rect-corner-in-polygon)."""
    x0, x1, y0, y1 = rect

    def pt_rect(px, py):
        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        return math.hypot(dx, dy)

    def pt_seg(px, py, a, b):
        ax, ay = a
        bx, by = b
        L2 = (bx - ax) ** 2 + (by - ay) ** 2
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2))
        return math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay))

    # polygon edges (corners are in [x-,y-],[x-,y+],[x+,y-],[x+,y+] order -> reorder to a loop)
    loop = [pts[0], pts[1], pts[3], pts[2]]
    d = min(pt_rect(px, py) for px, py in loop)
    rc = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    for (px, py) in rc:
        d = min(d, min(pt_seg(px, py, loop[i], loop[(i + 1) % 4]) for i in range(4)))
        if _inside(px, py, loop):
            return 0.0
    return d


def _inside(px, py, loop):
    sign = None
    for i in range(4):
        ax, ay = loop[i]
        bx, by = loop[(i + 1) % 4]
        c = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(c) < 1e-12:
            continue
        if sign is None:
            sign = c > 0
        elif sign != (c > 0):
            return False
    return True
