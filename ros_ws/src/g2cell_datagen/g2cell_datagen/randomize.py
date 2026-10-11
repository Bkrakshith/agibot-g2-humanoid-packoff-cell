"""Seeded per-episode randomisation (pure, no ROS).

One episode = one bag picked from the table. sample(cfg, seed) draws, from random.Random(seed) only:
which bag prim is picked, its label (-> crate or rack tote by the sort rule), its centre (x, y) in the
reachable part of the table spawn zone and its yaw. The other bags go to parking spots out of reach.
Rejection sampling keeps the rotated bag inside the table footprint and clear of the parked bags.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from g2cell_tasks.config import load_task_config
from g2cell_tasks.reachability import select_arm


class SamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rect:
    x0: float
    x1: float
    y0: float
    y1: float

    @classmethod
    def of(cls, xs, ys) -> "Rect":
        return cls(float(xs[0]), float(xs[1]), float(ys[0]), float(ys[1]))

    def contains(self, x, y, tol: float = 1e-9) -> bool:
        return self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol

    def inside(self, other: "Rect") -> bool:
        return other.contains(self.x0, self.y0) and other.contains(self.x1, self.y1)


@dataclass
class DatagenConfig:
    footprint: Rect
    spawn_zone: Rect
    reach_band_x: tuple
    top_z: float
    bag_size_xy: tuple
    region: Rect
    yaw_deg: tuple
    bag_ids: list
    labels: dict                 # label -> weight
    edge_margin: float
    min_gap: float
    max_tries: int
    park: list                   # [(x, y, yaw_deg, z_bottom)]
    left_table_max_y: float
    tasks_doc: dict = field(repr=False, default_factory=dict)
    min_x_minus_y: float | None = None   # cuts the far (-X, +Y) corner off the region: x - y >= this


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    bag_id: str
    x: float
    y: float
    yaw_deg: float
    label: str
    target: str                  # crate | rack_lane
    lane: str
    parked: dict                 # bag id -> (x, y, yaw_deg, z_bottom)
    z: float = 0.87
    tries: int = 1

    def poses(self) -> dict:
        """bag id -> (x, y, yaw_deg, z_bottom) for every bag this episode places."""
        return {self.bag_id: (self.x, self.y, self.yaw_deg, self.z), **self.parked}

    def to_dict(self) -> dict:
        return {"seed": self.seed,
                "bag": {"id": self.bag_id, "x": self.x, "y": self.y, "z": self.z, "yaw_deg": self.yaw_deg,
                        "label": self.label},
                "target": self.target, "lane": self.lane,
                "parked": {k: list(v) for k, v in self.parked.items()},
                "tries": self.tries}

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeSpec":
        b = d["bag"]
        return cls(int(d["seed"]), b["id"], b["x"], b["y"], b["yaw_deg"], b["label"], d["target"], d["lane"],
                   {k: tuple(v) for k, v in d["parked"].items()}, b.get("z", 0.87), int(d.get("tries", 1)))


def load_config(datagen_doc: dict, tasks_doc: dict) -> DatagenConfig:
    t, r = datagen_doc["table"], datagen_doc["randomise"]
    return DatagenConfig(
        footprint=Rect.of(t["footprint_x"], t["footprint_y"]),
        spawn_zone=Rect.of(t["spawn_zone_x"], t["spawn_zone_y"]),
        reach_band_x=tuple(datagen_doc["reach_band_x"]),
        top_z=float(t["top_z"]),
        bag_size_xy=tuple(datagen_doc["bag_size_xy"]),
        region=Rect.of(r["region_x"], r["region_y"]),
        yaw_deg=tuple(float(v) for v in r["yaw_deg"]),
        bag_ids=list(r["bag_ids"]),
        labels={str(k): float(v) for k, v in r["labels"].items()},
        edge_margin=float(r.get("edge_margin", 0.02)),
        min_gap=float(r.get("min_gap", 0.03)),
        max_tries=int(r.get("max_tries", 500)),
        park=[_park(p, float(t["top_z"])) for p in r.get("park", [])],
        left_table_max_y=float(tasks_doc.get("reachability", {}).get("left_table_max_y", 0.85)),
        tasks_doc=tasks_doc,
        min_x_minus_y=(float(r["min_x_minus_y"]) if r.get("min_x_minus_y") is not None else None),
    )


def _park(p, top_z: float) -> tuple:
    p = [float(v) for v in p]
    return tuple(p) if len(p) == 4 else (p[0], p[1], p[2], top_z)


# ---- geometry --------------------------------------------------------------------------------
def bag_corners(x, y, yaw_deg, size_xy):
    hx, hy = size_xy[0] / 2.0, size_xy[1] / 2.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def separation(a, b) -> float:
    """Separating-axis distance between two convex polygons (> 0: apart by at least that much along
    the best edge normal; <= 0: overlapping)."""
    best = -math.inf
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            ln = math.hypot(ex, ey)
            nx, ny = -ey / ln, ex / ln
            pa = [nx * px + ny * py for px, py in a]
            pb = [nx * px + ny * py for px, py in b]
            best = max(best, max(min(pb) - max(pa), min(pa) - max(pb)))
    return best


def _on_table(cfg: DatagenConfig, corners) -> bool:
    m, f = cfg.edge_margin, cfg.footprint
    return all(f.x0 + m - 1e-9 <= cx <= f.x1 - m + 1e-9 and f.y0 + m - 1e-9 <= cy <= f.y1 - m + 1e-9
               for cx, cy in corners)


# ---- validation ------------------------------------------------------------------------------
def validate_config(cfg: DatagenConfig) -> list:
    """Problems with the ranges (empty = OK): the region must lie in the spawn zone, the reach band and
    below the left arm's table reach; every label must route to a target an arm reaches; parking spots
    must hold every non-picked bag, on the table and out of reach."""
    p = []
    r = cfg.region
    if not (r.x0 < r.x1 and r.y0 < r.y1):
        p.append(f"empty region {r}")
    if not r.inside(cfg.spawn_zone):
        p.append(f"region {r} not inside the spawn zone {cfg.spawn_zone}")
    if r.x0 < cfg.reach_band_x[0] or r.x1 > cfg.reach_band_x[1]:
        p.append(f"region X {r.x0}..{r.x1} outside the reach band {cfg.reach_band_x}")
    if r.y1 > cfg.left_table_max_y:
        p.append(f"region Y up to {r.y1} beyond left_table_max_y {cfg.left_table_max_y}")
    try:
        tcfg = load_task_config(cfg.tasks_doc)
    except Exception as e:  # noqa: BLE001
        return p + [f"tasks config invalid: {e}"]
    known = {b.id for b in tcfg.bags.on_table()}
    for b in cfg.bag_ids:
        if b not in known:
            p.append(f"bag id {b} not in tasks.yaml bags {sorted(known)}")
    if not cfg.labels or min(cfg.labels.values()) <= 0:
        p.append("labels need positive weights")
    for label in cfg.labels:
        target, lane = tcfg.sort.target_for(label)
        for corner in ((r.x0, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x1, r.y0)):
            ch = select_arm(tcfg.reach, "table", corner, target, lane=lane)
            if ch.arm is None:
                p.append(f"label {label!r} -> {target} {lane}: {ch.reason}")
                break
    if len(cfg.park) < len(cfg.bag_ids) - 1:
        p.append(f"{len(cfg.park)} park spots for {len(cfg.bag_ids) - 1} non-picked bags")
    f = cfg.footprint
    table_poly = [(f.x0, f.y0), (f.x1, f.y0), (f.x1, f.y1), (f.x0, f.y1)]
    for x, y, yaw, z in cfg.park:
        poly = bag_corners(x, y, yaw, cfg.bag_size_xy)
        if abs(z - cfg.top_z) < 1e-6:
            if not _on_table(cfg, poly):
                p.append(f"park spot ({x}, {y}, {yaw}) on the table leaves the table footprint")
        elif abs(z) < 1e-6:
            if separation(poly, table_poly) < cfg.min_gap:
                p.append(f"park spot ({x}, {y}, {yaw}) on the floor is under the table footprint")
        else:
            p.append(f"park spot ({x}, {y}) z {z} is neither the floor (0) nor the table top ({cfg.top_z})")
        if y <= cfg.left_table_max_y:
            p.append(f"park spot ({x}, {y}) is reachable (Y <= {cfg.left_table_max_y})")
    for i, a in enumerate(cfg.park):
        for b in cfg.park[i + 1:]:
            if abs(a[3] - b[3]) < 1e-6 and separation(bag_corners(*a[:3], cfg.bag_size_xy),
                                                      bag_corners(*b[:3], cfg.bag_size_xy)) < cfg.min_gap:
                p.append(f"park spots {a} and {b} overlap")
    return p


# ---- sampling --------------------------------------------------------------------------------
def _weighted(rng: random.Random, weights: dict) -> str:
    keys = sorted(weights)
    u = rng.random() * sum(weights[k] for k in keys)
    for k in keys:
        u -= weights[k]
        if u < 0:
            return k
    return keys[-1]


def sample(cfg: DatagenConfig, seed: int) -> EpisodeSpec:
    rng = random.Random(int(seed))
    bag_id = rng.choice(sorted(cfg.bag_ids))
    label = _weighted(rng, cfg.labels)
    tcfg = load_task_config(cfg.tasks_doc)
    target, lane = tcfg.sort.target_for(label)
    others = [b for b in sorted(cfg.bag_ids) if b != bag_id]
    parked = {b: cfg.park[i] for i, b in enumerate(others)}
    park_polys = [bag_corners(*p[:3], cfg.bag_size_xy) for p in parked.values()
                  if abs(p[3] - cfg.top_z) < 1e-6]          # floor spots cannot touch a bag on the table
    r = cfg.region
    for tries in range(1, cfg.max_tries + 1):
        x, y = rng.uniform(r.x0, r.x1), rng.uniform(r.y0, r.y1)
        yaw = rng.uniform(*cfg.yaw_deg)
        if cfg.min_x_minus_y is not None and x - y < cfg.min_x_minus_y:
            continue
        poly = bag_corners(x, y, yaw, cfg.bag_size_xy)
        if not _on_table(cfg, poly):
            continue
        if any(separation(poly, q) < cfg.min_gap for q in park_polys):
            continue
        return EpisodeSpec(int(seed), bag_id, round(x, 4), round(y, 4), round(yaw, 2), label, target, lane,
                           parked, cfg.top_z, tries)
    raise SamplingError(f"seed {seed}: no valid bag pose in {cfg.max_tries} tries (region {r})")


def tasks_table_top(tasks_doc: dict, default: float = 0.87) -> float:
    """Table top the tasks.yaml bags rest on (top_z - bag height 0.07 for the Stage 2 bags)."""
    return float(tasks_doc.get("table_top_z", default))


def tasks_doc_for(tasks_doc: dict, spec: EpisodeSpec) -> dict:
    """tasks.yaml content for this episode's pick_bag_server: the picked bag at its sampled pose and
    label, the others at their parking spots."""
    out = copy.deepcopy(tasks_doc)
    poses = spec.poses()
    for b in out["bags"]:
        if b["id"] in poses:
            x, y, _, z = poses[b["id"]]
            height = float(b["top_z"]) - tasks_table_top(tasks_doc)
            b["xy"] = [x, y]
            b["top_z"] = round(z + height, 4)
        if b["id"] == spec.bag_id:
            b["label"] = spec.label
    return out


__all__ = ["DatagenConfig", "EpisodeSpec", "Rect", "SamplingError", "bag_corners", "load_config",
           "sample", "separation", "tasks_doc_for", "validate_config"]
