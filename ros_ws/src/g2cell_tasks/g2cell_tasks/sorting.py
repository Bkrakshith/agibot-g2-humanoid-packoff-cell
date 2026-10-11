"""Sort rule (spec): each bag has a hidden label "good", "reject" or a SKU name. Good bags go to
the pack crate; rejects and other SKUs go to their rack lane."""
from __future__ import annotations

GOOD = "good"
CRATE, RACK_LANE = "crate", "rack_lane"


class SortRules:
    def __init__(self, lanes: dict, default_lane: str, known_lanes, good_label: str = GOOD):
        self.lanes = dict(lanes)
        self.default_lane = default_lane
        self.known = list(known_lanes)
        self.good = good_label
        bad = [v for v in [*self.lanes.values(), default_lane] if v not in self.known]
        if bad:
            raise ValueError(f"sort rules name unknown lanes {bad} (known: {self.known})")

    def target_for(self, label: str):
        if label == self.good:
            return CRATE, ""
        return RACK_LANE, self.lanes.get(label, self.default_lane)

    def resolve(self, goal_target: str, goal_lane: str, label: str):
        """Goal fields win; empty fields come from the sort rule. Raises ValueError."""
        if not goal_target:
            return self.target_for(label)
        if goal_target == CRATE:
            return CRATE, ""
        if goal_target != RACK_LANE:
            raise ValueError(f"target must be crate or rack_lane, got {goal_target!r}")
        lane = goal_lane or self.lanes.get(label, self.default_lane)
        if lane not in self.known:
            raise ValueError(f"unknown rack lane {lane!r} (known: {self.known})")
        return RACK_LANE, lane
