import math

from g2cell_motion.geometry import down_quat
from g2cell_motion.planner import orient_candidates, plan_chain

NAMES = ["j1"]


def test_orient_candidates_prefer_first_and_no_duplicates():
    c = orient_candidates(prefer=(90, 20))
    assert c[0][0] == (90, 20)
    labels = [l for l, _ in c]
    assert len(labels) == len(set(labels)) == 24


def test_orient_candidates_fixed_quaternion():
    q = (0.0, 1.0, 0.0, 0.0)
    assert orient_candidates(fixed=q) == [("fixed", q)]


def test_plan_chain_first_orientation_that_solves_everything():
    calls = []

    def solve(pos, quat, seed):
        calls.append(quat)
        # only yaw 90 tilt 0 solves
        return {"j1": pos[2]} if quat == down_quat(math.radians(90), 0.0) else None

    r = plan_chain([(0, 0, 1.0), (0, 0, 0.9)], solve, lambda path: 1.0, seed={"j1": 0.0}, joint_names=NAMES)
    assert r.ok and r.label == (90, 0) and len(r.solutions) == 2
    assert r.quat == down_quat(math.radians(90), 0.0)


def test_plan_chain_seeds_each_waypoint_with_previous_solution():
    seeds = []

    def solve(pos, quat, seed):
        seeds.append(dict(seed))
        return {"j1": pos[2]}

    plan_chain([(0, 0, 1.0), (0, 0, 0.5)], solve, lambda p: 1.0, seed={"j1": 7.0}, joint_names=NAMES,
               tool_frame=False)
    assert seeds[0]["j1"] == 7.0 and math.isclose(seeds[1]["j1"], 1.0)


def test_plan_chain_rejects_head_clearance_and_reports_reason():
    r = plan_chain([(0, 0, 1.0)], lambda p, q, s: {"j1": 0.0}, lambda path: 0.1,
                   seed={"j1": 0.0}, joint_names=NAMES, min_clearance=0.2)
    assert not r.ok and r.reason == "head_clearance" and math.isclose(r.clearance, 0.1)


def test_plan_chain_ik_failure_reason():
    r = plan_chain([(0, 0, 1.0)], lambda p, q, s: None, lambda path: 1.0, seed={}, joint_names=NAMES)
    assert not r.ok and r.reason == "ik_failed"


def test_plan_chain_clearance_path_includes_start_and_end():
    seen = []

    def clear(path):
        seen.append(path)
        return 1.0

    plan_chain([(0, 0, 1.0)], lambda p, q, s: {"j1": 0.3, "other": 9}, clear, seed={"j1": 0.0},
               joint_names=NAMES, start={"j1": -1.0, "x": 1}, end={"j1": 2.0})
    assert seen[0] == [{"j1": -1.0}, {"j1": 0.3}, {"j1": 2.0}]


def test_plan_chain_tool_frame_converts_to_end_link():
    got = []

    def solve(pos, quat, seed):
        got.append(pos)
        return {"j1": 0.0}

    plan_chain([(0.1, 0.2, 1.0)], solve, lambda p: 1.0, seed={}, joint_names=NAMES, orients=[((0, 0), down_quat(0))],
               tool_frame=True)
    assert all(math.isclose(a, b) for a, b in zip(got[0], (0.1, 0.2, 1.0 + 0.125 - 0.04)))
    got.clear()
    plan_chain([(0.1, 0.2, 1.0)], solve, lambda p: 1.0, seed={}, joint_names=NAMES, orients=[((0, 0), down_quat(0))],
               tool_frame=False)
    assert got[0] == (0.1, 0.2, 1.0)


# ---- waist-first planning -------------------------------------------------------------------
from g2cell_motion.planner import plan_chain_waist, waist_profiles  # noqa: E402
from g2cell_motion.waist_policy import WaistConfig  # noqa: E402

W = "idx05_body_joint5"
ARM = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
HOME7 = {n: 0.0 for n in ARM}
CFG0 = WaistConfig(pivot=(0.0, 0.0))
TABLE = (0.0, 0.8, 0.97)      # bearing 90 deg -> waist 60 deg


def test_waist_profiles_modes():
    prof, dec = waist_profiles([TABLE], 0.0, "waist_first", CFG0)
    assert math.isclose(prof[0][0], math.radians(60)) and dec is not None
    assert [0.0] in prof and prof[-1] == [0.0]                      # no-turn fallback last
    assert waist_profiles([TABLE, TABLE], 0.3, "fixed", CFG0)[0] == [[0.3, 0.3]]
    assert waist_profiles([TABLE], 0.0, "absolute", CFG0, yaw=5.0)[0] == [[math.radians(100)]]
    import pytest
    with pytest.raises(ValueError):
        waist_profiles([TABLE], 0.0, "spin", CFG0)


def test_plan_chain_waist_solves_in_turned_frame_and_syncs_waist():
    seen = []

    def solve(pos, q, seed):
        seen.append(seed[W])
        return {n: 0.1 for n in ARM} if abs(seed[W] - math.radians(60)) < 1e-6 else None

    r = plan_chain_waist([TABLE], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                         waist_now=0.0, cfg=CFG0, start={**HOME7, W: 0.0})
    assert r.ok and math.isclose(r.waist_yaws[0], math.radians(60))
    assert math.isclose(r.solutions[0][W], math.radians(60)) and r.solutions[0]["a1"] == 0.1
    assert math.isclose(seen[0], math.radians(60))
    assert not r.needs_base


def test_plan_chain_waist_orientations_follow_torso():
    quats = []

    def solve(pos, q, seed):
        quats.append(q)
        return {n: 0.0 for n in ARM}

    plan_chain_waist([TABLE], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                     waist_now=0.0, cfg=CFG0)
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(quats[0], down_quat(math.radians(60), 0.0)))  # label (0, 0) = along the turned torso


def test_plan_chain_waist_fixed_keeps_waist():
    r = plan_chain_waist([TABLE], lambda p, q, s: {n: 0.0 for n in ARM}, lambda p: 1.0, seed=dict(HOME7),
                         joint_names=ARM, home=HOME7, waist_now=0.2, mode="fixed", cfg=CFG0)
    assert r.ok and r.waist_yaws == [0.2] and r.solutions[0][W] == 0.2


def test_plan_chain_waist_tries_home_seed_and_keeps_best():
    def solve(pos, q, seed):
        # seeded from the current (twisted-away) pose -> far solution; from home -> near-home solution
        return {n: (0.05 if seed["a2"] == 0.0 else 1.0) for n in ARM}

    cur = {n: 1.2 for n in ARM}
    r = plan_chain_waist([(0.6, 0.0, 1.1)], solve, lambda p: 1.0, seed=cur, joint_names=ARM, home=HOME7,
                         waist_now=0.0, cfg=CFG0)
    assert r.ok and r.solutions[0]["a2"] == 0.05


def test_plan_chain_waist_rejects_backward_twist():
    def solve(pos, q, seed):
        return {**{n: 0.0 for n in ARM}, "a1": math.radians(150)}

    r = plan_chain_waist([(0.6, 0.0, 1.1)], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM,
                         home=HOME7, waist_now=0.0, cfg=CFG0)
    assert not r.ok and r.reason == "twist" and r.twist_rejected > 0


def test_plan_chain_waist_rechecks_head_clearance_with_waist_in_path():
    paths = []

    def clear(path):
        paths.append(path)
        return 0.1 if path[-1][W] > 0.5 else 0.5          # turned paths too close to the head

    r = plan_chain_waist([TABLE], lambda p, q, s: {n: 0.0 for n in ARM}, clear, seed=dict(HOME7),
                         joint_names=ARM, home=HOME7, waist_now=0.0, cfg=CFG0, start={**HOME7, W: 0.0})
    assert all(W in c for c in paths[0])
    assert paths[0][0][W] == 0.0                          # path starts at the current waist
    assert r.ok and r.waist_yaws[0] < 0.5                 # a fallback waist profile was used


def test_plan_chain_waist_behind_flags_base_turn():
    r = plan_chain_waist([(-1.0, 0.0, 1.0)], lambda p, q, s: {n: 0.0 for n in ARM}, lambda p: 1.0,
                         seed=dict(HOME7), joint_names=ARM, home=HOME7, waist_now=0.0, cfg=CFG0)
    assert r.ok and r.needs_base and math.isclose(abs(r.base_turn), math.radians(50))
    assert math.isclose(abs(r.waist_yaws[0]), math.radians(100))


# ---- PickBag transfer fix: waist sequencing in absolute mode, lookahead points -------------------
def test_waist_profiles_absolute_interpolates_via_points_first():
    """absolute + via points: the via points get yaws on the way from the current waist to the target
    (transfer from a table pick at waist 78 deg to the carton at 0: the via is at ~39 deg, where it is
    reachable; at waist 0 it is too close to the torso). Then the old all-target profile, then the via
    at the current waist."""
    now, tgt = math.radians(78.0), 0.0
    prof, dec = waist_profiles([TABLE, TABLE], now, "absolute", CFG0, yaw=tgt)
    assert dec is None
    assert len(prof[0]) == 2 and math.isclose(prof[0][0], now / 2) and prof[0][1] == tgt
    assert [tgt, tgt] in prof and [now, tgt] in prof
    assert all(p[-1] == tgt for p in prof)                   # the goal's waist yaw is always honoured
    three = waist_profiles([TABLE] * 3, now, "absolute", CFG0, yaw=tgt)[0][0]
    assert all(math.isclose(a, b) for a, b in zip(three, [now * 2 / 3, now / 3, 0.0]))


def test_waist_profiles_absolute_single_point_and_no_turn_unchanged():
    assert waist_profiles([TABLE], 0.7, "absolute", CFG0, yaw=0.0)[0] == [[0.0]]
    assert waist_profiles([TABLE, TABLE], 0.0, "absolute", CFG0, yaw=0.0)[0] == [[0.0, 0.0]]


def test_plan_chain_waist_absolute_uses_interpolated_via_yaw():
    via, above = (0.3, 0.3, 1.2), (0.6, 0.0, 1.2)
    now = math.radians(78.0)

    def solve(pos, q, seed):     # the via is only solvable with the torso turned at least 30 deg
        if abs(pos[0] - 0.3) < 0.05 and seed[W] < math.radians(30):
            return None
        return {n: 0.0 for n in ARM}

    r = plan_chain_waist([via, above], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                         waist_now=now, mode="absolute", waist_yaw=0.0, cfg=CFG0, tool_frame=False)
    assert r.ok and math.isclose(r.waist_yaws[0], now / 2) and r.waist_yaws[1] == 0.0


def test_plan_chain_waist_lookahead_rejects_orientation_that_cannot_continue():
    """lookahead points (e.g. the grasp and lift below an approach) must solve with the same orientation and
    the final waist yaw, but are not part of the returned chain."""
    good = down_quat(math.radians(90), 0.0)
    seen = []

    def solve(pos, q, seed):
        seen.append((pos[2], seed[W]))
        if pos[2] < 0.95 and q != good:          # the low lookahead point only solves at yaw 90
            return None
        return {n: pos[2] for n in ARM}

    r = plan_chain_waist([(0.6, 0.0, 1.1)], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                         waist_now=0.2, mode="fixed", cfg=CFG0, tool_frame=False, lookahead=[(0.6, 0.0, 0.9)])
    assert r.ok and r.quat == good and len(r.solutions) == 1 and r.solutions[0]["a1"] == 1.1
    assert all(w == 0.2 for z, w in seen if z < 0.95)     # lookahead solved at the final waist yaw


def test_plan_chain_waist_lookahead_in_clearance_path_and_failure_reason():
    paths = []

    def clear(path):
        paths.append(path)
        return 1.0

    r = plan_chain_waist([(0.6, 0.0, 1.1)], lambda p, q, s: {n: p[2] for n in ARM}, clear, seed=dict(HOME7),
                         joint_names=ARM, home=HOME7, waist_now=0.0, mode="fixed", cfg=CFG0, tool_frame=False,
                         fixed=down_quat(0.0), lookahead=[(0.6, 0.0, 0.9)], start={**HOME7, W: 0.0})
    assert r.ok and [c["a1"] for c in paths[0]] == [0.0, 1.1, 0.9]
    bad = plan_chain_waist([(0.6, 0.0, 1.1)], lambda p, q, s: None if p[2] < 1.0 else {n: 0.0 for n in ARM},
                           lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7, waist_now=0.0,
                           mode="fixed", cfg=CFG0, tool_frame=False, lookahead=[(0.6, 0.0, 0.9)])
    assert not bad.ok and bad.reason == "lookahead_ik_failed"


def test_plan_chain_waist_fallback_profiles_can_search_every_orientation():
    """Bag_02's pick column only solves tilted (yaw 90, tilt 20) at waist 15-50 deg; fallback profiles that try
    only the 8 untilted orientations never find it. fallback_orients=0 means all 24."""
    good = down_quat(math.radians(90) + 0.0, math.radians(20))

    def solve(pos, q, seed):
        return {n: 0.0 for n in ARM} if seed[W] == 0.0 and q == good else None

    kw = dict(seed=dict(HOME7), joint_names=ARM, home=HOME7, waist_now=0.0, cfg=CFG0, tool_frame=False)
    assert not plan_chain_waist([TABLE], solve, lambda p: 1.0, fallback_orients=8, **kw).ok
    r = plan_chain_waist([TABLE], solve, lambda p: 1.0, fallback_orients=0, **kw)
    assert r.ok and r.label == (90, 20) and r.waist_yaws == [0.0]
