import copy
import math
from pathlib import Path

import pytest
import yaml

from g2cell_datagen import randomize as R
from g2cell_tasks.config import load_task_config
from g2cell_tasks.reachability import select_arm

PKG = Path(__file__).resolve().parents[1]
TASKS_YAML = PKG.parent / "g2cell_tasks" / "config" / "tasks.yaml"


@pytest.fixture(scope="module")
def docs():
    return (yaml.safe_load((PKG / "config" / "datagen.yaml").read_text()),
            yaml.safe_load(TASKS_YAML.read_text()))


@pytest.fixture(scope="module")
def cfg(docs):
    return R.load_config(*docs)


def test_shipped_config_is_valid(cfg):
    assert R.validate_config(cfg) == []


def test_region_inside_reachable_zone(cfg):
    # assets/cell/README.md Layout v2: spawn zone X -0.665..0.235 x Y 0.48..0.98, reach band X -0.4..0.35,
    # g2cell_tasks reachability: left arm only up to Y 0.85.
    r = cfg.region
    assert -0.665 <= r.x0 < r.x1 <= 0.235
    assert 0.48 <= r.y0 < r.y1 <= cfg.left_table_max_y == 0.85
    assert -0.40 <= r.x0 and r.x1 <= 0.35


def test_samples_in_ranges_and_reachable(cfg, docs):
    tcfg = load_task_config(docs[1])
    fp = cfg.footprint
    for seed in range(300):
        s = R.sample(cfg, seed)
        assert cfg.region.contains(s.x, s.y), s
        assert cfg.yaw_deg[0] <= s.yaw_deg <= cfg.yaw_deg[1]
        assert s.bag_id in cfg.bag_ids and s.label in cfg.labels
        for cx, cy in R.bag_corners(s.x, s.y, s.yaw_deg, cfg.bag_size_xy):
            assert fp.x0 + cfg.edge_margin - 1e-9 <= cx <= fp.x1 - cfg.edge_margin + 1e-9
            assert fp.y0 + cfg.edge_margin - 1e-9 <= cy <= fp.y1 - cfg.edge_margin + 1e-9
        # the sort rule decides the target; the reachability table must find an arm for it
        target, lane = tcfg.sort.target_for(s.label)
        assert (s.target, s.lane) == (target, lane)
        assert select_arm(tcfg.reach, "table", (s.x, s.y), target, lane=lane).arm is not None
        # parked bags: every other bag, out of reach, not overlapping the picked one
        assert set(s.parked) == set(cfg.bag_ids) - {s.bag_id}
        mine = R.bag_corners(s.x, s.y, s.yaw_deg, cfg.bag_size_xy)
        for px, py, pyaw, pz in s.parked.values():
            assert py > cfg.left_table_max_y
            assert R.separation(mine, R.bag_corners(px, py, pyaw, cfg.bag_size_xy)) >= cfg.min_gap - 1e-9


def test_deterministic_per_seed_and_order_independent(cfg):
    a = [R.sample(cfg, s) for s in (5, 6, 7)]
    b = [R.sample(cfg, s) for s in (7, 6, 5)][::-1]
    assert a == b
    assert R.sample(cfg, 5) != R.sample(cfg, 6)


def test_distribution_covers_choices(cfg):
    # the sampler's label weighting (the shipped config is good-only while the rack drop is unsolved)
    cfg = copy.copy(cfg)
    cfg.labels = {"good": 0.7, "reject": 0.3}
    ss = [R.sample(cfg, s) for s in range(400)]
    assert {s.bag_id for s in ss} == set(cfg.bag_ids)
    assert {s.label for s in ss} == {"good", "reject"}
    good = sum(s.label == "good" for s in ss) / len(ss)
    assert 0.6 < good < 0.8
    xs = [s.x for s in ss]
    ys = [s.y for s in ss]
    assert max(xs) - min(xs) > 0.3 and max(ys) - min(ys) > 0.15
    yaws = [s.yaw_deg for s in ss]
    assert min(yaws) < -60 and max(yaws) > 60


def test_spec_to_dict_roundtrip(cfg):
    s = R.sample(cfg, 3)
    d = s.to_dict()
    assert d["seed"] == 3 and d["bag"]["id"] == s.bag_id
    assert R.EpisodeSpec.from_dict(d) == s


def test_corners_and_separation():
    c = R.bag_corners(0.0, 0.0, 0.0, (0.2, 0.4))
    assert sorted(c) == sorted([(-0.1, -0.2), (0.1, -0.2), (0.1, 0.2), (-0.1, 0.2)])
    c90 = R.bag_corners(0.0, 0.0, 90.0, (0.2, 0.4))
    xs = [p[0] for p in c90]
    assert math.isclose(max(xs), 0.2, abs_tol=1e-9)
    other = R.bag_corners(0.5, 0.0, 0.0, (0.2, 0.4))
    assert math.isclose(R.separation(c, other), 0.3, abs_tol=1e-9)
    assert R.separation(c, R.bag_corners(0.05, 0.0, 30.0, (0.2, 0.4))) < 0


@pytest.mark.parametrize("path,value,needle", [
    (("randomise", "region_y"), [0.52, 0.95], "left_table_max_y"),
    (("randomise", "region_x"), [-0.60, 0.15], "reach band"),
    (("randomise", "region_x"), [-0.30, 0.30], "spawn zone"),
    (("randomise", "bag_ids"), ["Bag_01", "Bag_99"], "Bag_99"),
    (("randomise", "labels"), {"good": 1.0, "sample": 1.0}, "label"),
    (("randomise", "park"), [[-0.50, 0.90, 90.0]], "park"),
    (("randomise", "park"), [[-0.70, 0.90, 0.0], [-0.10, 0.90, 90.0]], "footprint"),
    (("randomise", "park"), [[-0.40, 0.90, 0.0, 0.0], [-0.10, 1.40, 0.0, 0.0]], "under the table"),
    (("randomise", "park"), [[-0.40, 1.40, 0.0, 0.5], [-0.10, 1.40, 0.0, 0.0]], "neither"),
    (("randomise", "park"), [[-0.40, 1.40, 0.0, 0.0], [-0.35, 1.40, 0.0, 0.0]], "overlap"),
])
def test_validate_rejects_bad_configs(docs, path, value, needle):
    d = copy.deepcopy(docs[0])
    d[path[0]][path[1]] = value
    probs = R.validate_config(R.load_config(d, docs[1]))
    assert any(needle in p for p in probs), probs


def test_unsatisfiable_region_raises(docs):
    d = copy.deepcopy(docs[0])
    d["randomise"]["park"] = [[-0.075, 0.66, 0.0], [-0.50, 0.90, 90.0]]   # parked bag (table) in the region
    d["randomise"]["region_x"] = [-0.10, -0.05]
    d["randomise"]["region_y"] = [0.62, 0.70]
    d["randomise"]["max_tries"] = 50
    with pytest.raises(R.SamplingError):
        R.sample(R.load_config(d, docs[1]), 0)


def test_tasks_doc_for_episode(cfg, docs):
    s = R.sample(cfg, 11)
    out = R.tasks_doc_for(docs[1], s)
    assert out is not docs[1] and docs[1]["bags"][0]["xy"] == [0.05, 0.62]   # input untouched
    by_id = {b["id"]: b for b in out["bags"]}
    assert by_id[s.bag_id]["xy"] == [s.x, s.y] and by_id[s.bag_id]["label"] == s.label
    for bid, (px, py, _, pz) in s.parked.items():
        assert by_id[bid]["xy"] == [px, py]
        assert by_id[bid]["top_z"] == pytest.approx(pz + 0.07)
    assert by_id[s.bag_id]["top_z"] == pytest.approx(0.94)
    tcfg = load_task_config(out)        # still a valid tasks.yaml
    assert tcfg.bags.get(s.bag_id).xy == (s.x, s.y)


def test_table_park_spots_still_supported(docs):
    d = copy.deepcopy(docs[0])
    d["randomise"]["park"] = [[-0.54, 0.90, 90.0], [-0.20, 0.90, 90.0, 0.87]]
    cfg = R.load_config(d, docs[1])
    assert R.validate_config(cfg) == []
    for seed in range(50):
        s = R.sample(cfg, seed)
        mine = R.bag_corners(s.x, s.y, s.yaw_deg, cfg.bag_size_xy)
        for px, py, pyaw, pz in s.parked.values():
            assert pz == 0.87
            assert R.separation(mine, R.bag_corners(px, py, pyaw, cfg.bag_size_xy)) >= cfg.min_gap - 1e-9


# ---- far-corner cut (measured: tools/perf/reach_sweep.py, output/perf/reach_sweep_pilot.json) ---------------
def test_far_corner_cut_is_respected(docs):
    d, t = docs
    d = copy.deepcopy(d)
    d["randomise"]["min_x_minus_y"] = -0.85
    c = R.load_config(d, t)
    assert c.min_x_minus_y == -0.85
    xs = [R.sample(c, s) for s in range(300)]
    assert all(s.x - s.y >= -0.85 - 1e-9 for s in xs)
    assert any(s.x - s.y < -0.75 for s in xs)              # the cut trims the corner, not the region


def test_shipped_config_avoids_the_measured_unreachable_corner(cfg):
    # pilot_v1 seeds 2 (-0.284, 0.711) and 3 (-0.241, 0.777) failed approach planning; both must be excluded
    assert cfg.region.x0 >= -0.26
    for x, y in ((-0.284, 0.711), (-0.241, 0.777), (-0.225, 0.73), (-0.15, 0.80)):
        assert not (cfg.region.contains(x, y) and x - y >= (cfg.min_x_minus_y or -9))
