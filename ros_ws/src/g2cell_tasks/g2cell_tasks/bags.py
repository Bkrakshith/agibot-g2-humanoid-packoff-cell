"""Bags known to the task layer. Stage 2: the three rigid placeholders from config/tasks.yaml.
Stage 5 replaces the static list with the spawner's BagInfo stream (TODO in pick_bag_server)."""
from __future__ import annotations

from dataclasses import dataclass

VARIANTS = ("small", "standard", "large")


@dataclass(frozen=True)
class Bag:
    id: str
    variant: str
    label: str
    xy: tuple          # bag centre, cell frame
    top_z: float       # bag top surface, cell frame


class BagCatalog:
    def __init__(self, bags):
        self._bags = {b.id: b for b in bags}
        self._order = [b.id for b in bags]
        self._removed: set[str] = set()

    @classmethod
    def from_list(cls, items) -> "BagCatalog":
        bags, seen = [], set()
        for d in items:
            if d.get("variant") not in VARIANTS:
                raise ValueError(f"bag {d.get('id')!r}: variant must be one of {VARIANTS}")
            if not d.get("label"):
                raise ValueError(f"bag {d.get('id')!r}: empty label")
            xy = d.get("xy")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError(f"bag {d.get('id')!r}: xy needs 2 values")
            if d["id"] in seen:
                raise ValueError(f"duplicate bag id {d['id']!r}")
            seen.add(d["id"])
            bags.append(Bag(str(d["id"]), d["variant"], str(d["label"]), (float(xy[0]), float(xy[1])),
                            float(d["top_z"])))
        return cls(bags)

    def get(self, bag_id: str) -> Bag:
        return self._bags[bag_id]

    def on_table(self):
        return [self._bags[i] for i in self._order if i not in self._removed]

    def choose(self, bag_id: str, can_pick=lambda b: True):
        """'any' -> first bag still on the table that can_pick(bag); an id -> that bag (None if it
        already left the table). Unknown id -> KeyError."""
        if bag_id == "any":
            return next((b for b in self.on_table() if can_pick(b)), None)
        b = self._bags[bag_id]
        return None if bag_id in self._removed else b

    def mark_removed(self, bag_id: str) -> None:
        self._removed.add(bag_id)
