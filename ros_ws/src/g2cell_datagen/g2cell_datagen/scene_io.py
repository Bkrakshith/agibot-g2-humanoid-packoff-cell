"""Scene side of a datagen episode through the object pose relay (ROS-free logic, unit-tested).

* reset "teleport": one /g2cell/object_poke request per bag (the picked bag at its sampled pose, the others
  parked) plus the fixtures to restore (the pack carton), each with the yaw and "stop" (zero velocity), applied
  by the engine hook between physics steps (g2cell_bringup object_relay). No sim restart: ~2 s instead of ~1 min.
* check_reset: the poses the relay reports afterwards against the requests.
* placement_of / score: where the bag came to rest (pack carton interior, rack tote, table, floor, conveyor)
  -> the g2cell_tasks outcome (g2cell_tasks.outcome.classify), so an episode whose motion completed is labelled
  success / drop / wrong_target / overflow / grasp_failed instead of "unverified".

Poses: (xyz, (w, x, y, z)) in the parent frame of the body (= world for /World/Bags, Cartons, Totes), as the
relay publishes them on /tf_render. Bag prims have their origin at the bag's bottom centre.
"""
from __future__ import annotations

import math

from g2cell_tasks.outcome import Placement, classify

BAGS_ROOT = "/World/Bags/"
CARTON = "/World/Cartons/Carton_Pack"
TOTES = {"L1_1": "/World/Totes/Tote_L1_1", "L2_1": "/World/Totes/Tote_L2_1"}
LIFT = 0.002                         # teleport this far above the resting height (no initial penetration)
BAG_CENTRE = (0.0, 0.0, 0.035)       # bag 0.21 x 0.30 x 0.07, origin at the bottom
BAG_HALF_H = 0.035
# interiors in the body frame: (centre, half extents) from cell.usda InteriorVolume prims
CARTON_INTERIOR = ((0.0, 0.0, 0.129), (0.196, 0.296, 0.121))
TOTE_INTERIOR = ((0.0, 0.0, 0.183), (0.183, 0.283, 0.142))
TABLE_XY = ((-0.715, 0.285), (0.43, 1.03))
TABLE_TOP = 0.87
ABOVE_SLACK = 0.10                   # a bag centre up to this far above the interior still counts as "in"
SPAWN_TOL = 0.05
FLOOR_Z = 0.30


def teleport_requests(spec, restore: dict) -> list:
    """spec: randomize.EpisodeSpec; restore: {body path: [x, y, z, yaw_deg]} (e.g. the pack carton)."""
    out = []
    for bag, (x, y, yaw, z) in spec.poses().items():
        out.append({"path": BAGS_ROOT + bag, "xyz": [float(x), float(y), float(z) + LIFT], "yaw_deg": float(yaw),
                    "stop": True})
    for path, (x, y, z, yaw) in restore.items():
        out.append({"path": path, "xyz": [float(x), float(y), float(z) + LIFT], "yaw_deg": float(yaw), "stop": True})
    return out


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def check_reset(poses: dict, requests: list, tol_m: float = 0.01, tol_deg: float = 3.0):
    """-> (ok, worst {path, err_m, err_deg}). err_m: horizontal error plus any height error beyond LIFT; the yaw
    error is taken modulo 180 deg (box bags and cartons look the same turned half way)."""
    rows = []
    for r in requests:
        p = poses.get(r["path"])
        if p is None:
            return False, {"path": r["path"], "err_m": float("inf"), "err_deg": float("inf")}
        (x, y, z), q = p
        tx, ty, tz = r["xyz"]
        err = math.dist((x, y), (tx, ty)) + max(0.0, abs(z - (tz - LIFT)) - LIFT)
        dy = (yaw_of(q) - r["yaw_deg"]) % 180.0
        rows.append({"path": r["path"], "err_m": err, "err_deg": min(dy, 180.0 - dy)})
    if not rows:
        return True, {"path": "", "err_m": 0.0, "err_deg": 0.0}
    bad = [w for w in rows if w["err_m"] > tol_m or w["err_deg"] > tol_deg]
    worst = max(bad or rows, key=lambda w: (w["err_m"], w["err_deg"]))
    return not bad, worst


# ---------------------------------------------------------------------------------------------- scorer

def _rot(q, v):
    w, x, y, z = q
    # v' = q v q*
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


def _to_local(pose, p):
    (ox, oy, oz), (w, x, y, z) = pose
    return _rot((w, -x, -y, -z), (p[0] - ox, p[1] - oy, p[2] - oz))


def _inside(local, box, above: float = 0.0):
    (cx, cy, cz), (hx, hy, hz) = box
    return (abs(local[0] - cx) <= hx and abs(local[1] - cy) <= hy
            and cz - hz - 0.01 <= local[2] <= cz + hz + above)


def placement_of(bag_pose, bodies: dict, spawn=None) -> dict:
    """Where the bag (pose) rests. bodies: {CARTON: pose, TOTES[lane]: pose, ...}; spawn: the bag's sampled xy.
    -> {target: crate|rack_lane|table|floor|conveyor, lane, proud_of_rim, at_spawn, centre}."""
    (bx, by, bz), bq = bag_pose
    c = _rot(bq, BAG_CENTRE)
    centre = (bx + c[0], by + c[1], bz + c[2])
    out = {"target": "", "lane": "", "proud_of_rim": False, "at_spawn": False,
           "centre": [round(v, 4) for v in centre]}
    if CARTON in bodies:
        loc = _to_local(bodies[CARTON], centre)
        if _inside(loc, CARTON_INTERIOR, ABOVE_SLACK):
            top = CARTON_INTERIOR[0][2] + CARTON_INTERIOR[1][2]
            return {**out, "target": "crate", "proud_of_rim": loc[2] + BAG_HALF_H > top + 0.01}
    for lane, path in TOTES.items():
        if path in bodies:
            loc = _to_local(bodies[path], centre)
            if _inside(loc, TOTE_INTERIOR, ABOVE_SLACK):
                top = TOTE_INTERIOR[0][2] + TOTE_INTERIOR[1][2]
                return {**out, "target": "rack_lane", "lane": lane, "proud_of_rim": loc[2] + BAG_HALF_H > top + 0.01}
    if centre[2] < FLOOR_Z:
        return {**out, "target": "floor"}
    (x0, x1), (y0, y1) = TABLE_XY
    if x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1 and abs(bz - TABLE_TOP) < 0.05:
        at = spawn is not None and math.dist((bx, by), spawn) <= SPAWN_TOL
        return {**out, "target": "table", "at_spawn": at}
    return {**out, "target": "conveyor"}


def score(task_outcome: str, intended_target: str, lane: str, placement, grasp_attempts: int):
    """-> (outcome, verified). Only a completed motion ("unverified") is re-labelled from the placement; the
    task's own failures (plan_failed, unreachable, ...) stand. A bag still on the table at its spawn point was
    never lifted: grasp_failed."""
    if task_outcome != "unverified" or placement is None:
        return task_outcome, False
    if placement["target"] == "table":
        return ("grasp_failed", True) if placement.get("at_spawn") else ("drop", True)
    p = Placement(placement["target"], placement.get("lane", ""), bool(placement.get("proud_of_rim")))
    return classify(intended_target, lane, p, max(1, int(grasp_attempts)))
