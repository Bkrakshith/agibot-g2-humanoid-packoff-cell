"""Head-clearance check along a joint-space path (no ROS imports; FK is injected).

The G2's arm colliders are disabled in Genie Sim, so nothing stops an arm sweeping through the
head. Every arm/gripper link is kept HEAD_CLEARANCE (0.20 m) from the head_link3 origin and the
head camera along the min-jerk path, sampled between waypoints (as g2cell_demo did).

Fixtures: the gripper and arm links DO push dynamic scene bodies. The first live datagen run (2026-10-11)
showed the approach from home (hand 0.10 m over the pack carton) dipping the gripper through the carton's +Y
wall while the waist turned, which threw the carton 0.5 m. With `fixtures` (oriented boxes from
cell_colliders.json, see load_fixture_boxes) a sampled pose whose link / fingertip points come closer than
`fixture_margin` to any box counts as clearance 0 (unsafe), like an FK failure.
"""
from __future__ import annotations

import math

from g2cell_motion.geometry import interpolate, qrot


def head_points(link_pos, link_quat, cam_offset):
    """[head_link3 origin, head camera] in the FK frame, from head_link3's pose."""
    c = qrot(link_quat, cam_offset)
    o = tuple(link_pos)
    return [o, (o[0] + c[0], o[1] + c[1], o[2] + c[2])]


def min_distance(points, heads) -> float:
    best = float("inf")
    for p in points:
        for h in heads:
            best = min(best, sum((x - y) ** 2 for x, y in zip(p, h)) ** 0.5)
    return best


def path_clearance(chain, fk_points, heads, samples: int = 8) -> float:
    """Min distance from `heads` to the link points of every sampled pose along `chain`
    (list of {joint: value} dicts). fk_points(joints) -> list of xyz, or None on FK failure
    (counted as 0 clearance, i.e. unsafe)."""
    names = list(chain[0])
    poses = [dict(chain[0])] if len(chain) == 1 else []
    for a, b in zip(chain, chain[1:]):
        av, bv = [a[n] for n in names], [b.get(n, a[n]) for n in names]
        poses += [dict(zip(names, q)) for q in interpolate(av, bv, samples)]
    best = float("inf")
    for joints in poses:
        pts = fk_points(joints)
        if pts is None:
            return 0.0
        best = min(best, min_distance(pts, heads))
    return best


MOVABLE_PREFIXES = ("/World/Bags/", "/World/LineBags/")


def load_fixture_boxes(path: str, radius: float = 1.6, centre=(0.0, 0.0), exclude=MOVABLE_PREFIXES) -> list:
    """cell_colliders.json -> list of {name, c, axes, h} within `radius` (horizontal) of `centre` (the robot
    base), without the bags (they are what the gripper is meant to touch)."""
    import json  # noqa: PLC0415
    with open(path) as f:
        boxes = json.load(f)
    out = []
    for b in boxes:
        if any(b["name"].startswith(p) for p in exclude):
            continue
        reach = radius + max(b["h"][0], b["h"][1])
        if math.hypot(b["c"][0] - centre[0], b["c"][1] - centre[1]) > reach:
            continue
        out.append(b)
    return out


def box_distance(p, box) -> float:
    """Signed distance from point p to an oriented box {c, axes (rows = box axes in the world), h}:
    > 0 outside, < 0 inside (minus the depth to the nearest face)."""
    d = [p[i] - box["c"][i] for i in range(3)]
    q = [abs(sum(a * b for a, b in zip(ax, d))) - h for ax, h in zip(box["axes"], box["h"])]
    outside = math.sqrt(sum(max(v, 0.0) ** 2 for v in q))
    return outside if outside > 0 else max(q)


def min_box_distance(points, boxes) -> float:
    best = float("inf")
    for b in boxes:
        for p in points:
            best = min(best, box_distance(p, b))
    return best


def path_clearance_fk(chain, fk, samples: int = 8, fixtures=None, fixture_margin: float = 0.04) -> float:
    """Like path_clearance, but the head is re-evaluated at every sample: fk(joints) ->
    (link points, head points[, fixture-only points]) | None. Needed once the waist turns (the head sits on
    body_link5, so in base_link it moves with the waist). With `fixtures` every sample's link points (and the
    optional fixture-only points, e.g. fingertips) must stay `fixture_margin` away from every box, else the
    path's clearance is 0."""
    names = list(chain[0])
    poses = [dict(chain[0])] if len(chain) == 1 else []
    for a, b in zip(chain, chain[1:]):
        av, bv = [a[n] for n in names], [b.get(n, a[n]) for n in names]
        poses += [dict(zip(names, q)) for q in interpolate(av, bv, samples)]
    best = float("inf")
    for joints in poses:
        r = fk(joints)
        if r is None:
            return 0.0
        pts, heads = r[0], r[1]
        if fixtures:
            extra = r[2] if len(r) > 2 else []
            if min_box_distance(list(pts) + list(extra), fixtures) < fixture_margin:
                return 0.0
        best = min(best, min_distance(pts, heads))
    return best
