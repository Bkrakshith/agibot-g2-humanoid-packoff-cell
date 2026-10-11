"""Fingertip waypoints (cell frame) for one pick-and-place; heights as in the Stage 2 demo."""
from __future__ import annotations

from dataclasses import dataclass, field

APPROACH_CLEAR = 0.13   # above the bag top before descending
GRASP_DEPTH = 0.01      # fingertips this far below the bag top
LIFT_CLEAR = 0.17
VIA_ABOVE_RIM = 0.12    # midway via point: never lower than this over the target rim
ABOVE_RIM = 0.17
RELEASE_ABOVE_RIM = 0.03
RETRACT_ABOVE_RIM = 0.20


@dataclass(frozen=True)
class CellTargets:
    crate_xy: tuple = (0.60, 0.0)
    crate_rim_z: float = 0.82 + 0.224
    lanes: dict = field(default_factory=dict)   # lane -> (x, y, rim_z) drop point

    @classmethod
    def from_dict(cls, d: dict) -> "CellTargets":
        return cls(tuple(d["crate_xy"]), float(d["crate_rim_z"]),
                   {k: tuple(float(c) for c in v) for k, v in d.get("lanes", {}).items()})

    def point(self, target: str, lane: str):
        """(x, y, rim_z) of the place target. KeyError for unknown target/lane."""
        if target == "crate":
            return (self.crate_xy[0], self.crate_xy[1], self.crate_rim_z)
        if target == "rack_lane":
            return self.lanes[lane]
        raise KeyError(f"unknown target {target!r}")


def pick_waypoints(xy, top_z):
    x, y = xy
    return {"approach": (x, y, top_z + APPROACH_CLEAR),
            "grasp": (x, y, top_z - GRASP_DEPTH),
            "lift": (x, y, top_z + LIFT_CLEAR)}


def place_waypoints(pick_xy, lift_z, target):
    """target = (x, y, rim_z). The via point sits halfway, no lower than the lift: with the G2's
    arm colliders disabled a free joint-space swing could sweep the forearm through the head."""
    px, py, rim = target
    x, y = pick_xy
    return {"via": ((x + px) / 2, (y + py) / 2, max(lift_z, rim + VIA_ABOVE_RIM)),
            "above": (px, py, rim + ABOVE_RIM),
            "release": (px, py, rim + RELEASE_ABOVE_RIM),
            "retract": (px, py, rim + RETRACT_ABOVE_RIM)}
