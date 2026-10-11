"""Determinism + bounded attempt budgets (docs/OPTIMISATION.md "Motion planning")."""
import math

from g2cell_motion.planner import PlanBudget, ik_seed_sequence, plan_chain_waist
from g2cell_motion.waist_policy import WaistConfig

W = "idx05_body_joint5"
ARM = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
HOME7 = {n: 0.0 for n in ARM}
CFG0 = WaistConfig(pivot=(0.0, 0.0))
TABLE = (0.0, 0.8, 0.97)


def test_ik_seed_sequence_is_deterministic_and_starts_with_the_seed():
    base = {n: 0.1 * i for i, n in enumerate(ARM)}
    a = ik_seed_sequence(base, ARM, attempts=6, jitter=0.4)
    b = ik_seed_sequence(base, ARM, attempts=6, jitter=0.4)
    assert a == b and len(a) == 6 and a[0] == base
    assert all(abs(s[n] - base[n]) <= 0.4 for s in a for n in ARM)
    assert a[1] != a[2]
    assert ik_seed_sequence(base, ARM, attempts=6, jitter=0.4, rng_seed=1) != a


def test_ik_seed_sequence_does_not_touch_global_random():
    import random
    random.seed(5)
    x = random.random()
    random.seed(5)
    ik_seed_sequence(HOME7, ARM, attempts=4, jitter=0.4)
    assert random.random() == x


def test_budget_counts_solves_and_time():
    t = [0.0]
    b = PlanBudget(max_solves=3, max_seconds=10.0, clock=lambda: t[0])
    assert b.spend() and b.spend() and b.spend()
    assert not b.spend() and b.exhausted and b.solves == 4
    b2 = PlanBudget(max_solves=None, max_seconds=1.0, clock=lambda: t[0])
    assert b2.spend()
    t[0] = 2.0
    assert not b2.spend() and b2.exhausted


def test_plan_chain_waist_stops_at_the_budget():
    calls = []

    def solve(pos, q, seed):
        calls.append(1)
        return None

    r = plan_chain_waist([TABLE], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                         waist_now=0.0, cfg=CFG0, budget=PlanBudget(max_solves=5))
    assert not r.ok and r.reason == "budget" and len(calls) == 5


def test_plan_chain_waist_without_budget_unchanged():
    r = plan_chain_waist([TABLE], lambda p, q, s: None, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM,
                         home=HOME7, waist_now=0.0, cfg=CFG0)
    assert not r.ok and r.reason == "ik_failed"


def test_plan_chain_waist_early_exit_on_first_valid_plan():
    calls = []

    def solve(pos, q, seed):
        calls.append(1)
        return {n: 0.0 for n in ARM}

    r = plan_chain_waist([TABLE, TABLE], solve, lambda p: 1.0, seed=dict(HOME7), joint_names=ARM, home=HOME7,
                         waist_now=0.0, cfg=CFG0)
    assert r.ok and len(calls) <= 4             # one orientation x 2 waypoints x <= 2 seeds


def test_same_inputs_same_plan():
    def solve(pos, q, seed):        # deterministic but orientation-dependent fake IK
        if q[2] < 0.3:
            return None
        return {n: round(pos[0] + q[2] + seed.get(n, 0.0) * 0.5, 9) for n in ARM}

    kw = dict(seed=dict(HOME7), joint_names=ARM, home=HOME7, waist_now=0.0, cfg=CFG0, start={**HOME7, W: 0.0})
    r1 = plan_chain_waist([TABLE, (0.1, 0.7, 1.0)], solve, lambda p: 1.0, **kw)
    r2 = plan_chain_waist([TABLE, (0.1, 0.7, 1.0)], solve, lambda p: 1.0, **kw)
    assert r1.ok and r1 == r2
    assert math.isclose(r1.waist_yaws[0], r2.waist_yaws[0])


def test_zero_attempts_means_no_seeds():
    """ik_fallback_attempts 0 must not make any /compute_ik call."""
    assert ik_seed_sequence(HOME7, ARM, attempts=0, jitter=0.4) == []
    assert len(ik_seed_sequence(HOME7, ARM, attempts=1, jitter=0.4)) == 1
