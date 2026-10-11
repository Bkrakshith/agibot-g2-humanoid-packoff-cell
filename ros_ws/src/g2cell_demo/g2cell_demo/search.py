"""Pruned pick -> place plan search for packing_demo (no ROS imports, unit-tested).

The old loop solved the place chain again for every pick candidate and fixture-checked the WHOLE
12-waypoint path at the innermost level, so one bad pick segment was re-checked against every place
option (output/v2_work/take*/demo.log: ~22 identical rejections per run, each 4-9 s).

The path is  A = rest -> turn(wp) -> pick waypoints -> rest(wp)
             B = rest(wp) -> rest(wd)                       (the waist swing, arms tucked)
             C = rest(wd) -> place waypoints -> rest(wd) -> rest(0)
and path_check is a minimum over consecutive pairs, so check(A+B+C) == min(check(A), check(B), check(C))
exactly. Checking A before any place work, B per (wp, wd) and C per place chain, each memoised, returns
the same first plan as the brute-force order (wp, o1, wd, o2) while doing each check once.
"""
from __future__ import annotations

import json

INFO_MIN = (("fixture_clearance_m", "closest"), ("head_clearance_m", None), ("idle_arm_clearance_m", "idle_closest"))


class LazySeq:
    """Re-iterable, lazily evaluated view of an iterator (each item produced once)."""

    def __init__(self, it):
        self._it, self._items, self._done = iter(it), [], False

    def __iter__(self):
        i = 0
        while True:
            if i < len(self._items):
                yield self._items[i]
                i += 1
            elif self._done:
                return
            else:
                try:
                    self._items.append(next(self._it))
                except StopIteration:
                    self._done = True


def combine_info(infos):
    """Element-wise minimum of check_path info dicts (the closest collider follows its distance)."""
    out = {}
    for key, name in INFO_MIN:
        best = min(infos, key=lambda d: d[key])
        out[key] = best[key]
        if name:
            out[name] = best[name]
    return out


def _path_key(path):
    return json.dumps(path, sort_keys=True, default=str)


def memo_check(check):
    """Memoise check(path) -> (ok, info) by path content."""
    cache = {}

    def f(path):
        k = _path_key(path)
        if k not in cache:
            cache[k] = check(path)
        return cache[k]
    f.cache = cache
    return f


def pick_place_search(pick_waists, pick_chains, place_waists, place_chains, part_a, part_b, part_c, check,
                      log=None):
    """First (wp, o1, wd, o2) in brute-force order whose A, B and C all pass `check` (-> (ok, info)).
    pick_chains(wp) / place_chains(wd) -> iterable of (orientation, solutions); they are wrapped in
    LazySeq and memoised per waist, so each IK chain is solved at most once."""
    placed = {}

    def places(wd):
        if wd not in placed:
            placed[wd] = LazySeq(place_chains(wd))
        return placed[wd]

    for wp in pick_waists:
        for o1, pick in pick_chains(wp):
            ok_a, info_a = check(part_a(wp, pick))
            if not ok_a:
                if log:
                    log(f"pick waist {wp:+.0f} orient {o1}: rejected {info_a}")
                continue
            for wd in place_waists:
                ok_b, info_b = check(part_b(wp, wd))
                if not ok_b:
                    if log:
                        log(f"waist swing {wp:+.0f} -> {wd:+.0f}: rejected {info_b}")
                    continue
                for o2, plc in places(wd):
                    ok_c, info_c = check(part_c(wd, plc))
                    if not ok_c:
                        if log:
                            log(f"place waist {wd:+.0f} orient {o2}: rejected {info_c}")
                        continue
                    return {"wp": wp, "o1": o1, "pick": pick, "wd": wd, "o2": o2, "plc": plc,
                            "info": combine_info([info_a, info_b, info_c])}
    return None
