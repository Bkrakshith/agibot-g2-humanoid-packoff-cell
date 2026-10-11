import math

import pytest

from g2cell_motion.waist_policy import (WaistConfig, TwistLimits, bearing, best_solution, choose_waist_yaw,
                                        clamp_waist, solution_score, synced_chain, twist_violations,
                                        waist_candidates, wrap)

D = math.radians
PIVOT = (0.0, 0.0)
CFG = WaistConfig(pivot=PIVOT)
NAMES = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
HOME = dict(zip(NAMES, [0.0, D(-45), 0.0, D(-75), 0.0, D(-55), 0.0]))


# ---- bearing / yaw selection ------------------------------------------------------------
def test_defaults():
    c = WaistConfig()
    assert c.cone_deg == 30.0 and c.max_turn_deg == 100.0
    assert c.lower == pytest.approx(-3.0456) and c.upper == pytest.approx(3.0456)
    assert c.joint == "idx05_body_joint5"
    t = TwistLimits()
    assert t.j1_max_dev_deg == 90.0 and t.jump_max == 1.5 and t.jump_joints == (3, 5, 7)


def test_wrap():
    assert wrap(D(190)) == pytest.approx(D(-170))
    assert wrap(D(-190)) == pytest.approx(D(170))
    assert wrap(0.3) == pytest.approx(0.3)


def test_bearing_from_pivot():
    assert bearing((1.0, 0.0), PIVOT) == pytest.approx(0.0)
    assert bearing((0.0, 1.0), PIVOT) == pytest.approx(D(90))
    assert bearing((0.5, 0.5), (0.5, 0.0)) == pytest.approx(D(90))


def test_straight_ahead_no_turn():
    d = choose_waist_yaw((0.6, 0.0, 1.1), 0.0, CFG)
    assert d.yaw == 0.0 and d.turn == 0.0 and not d.needs_base and d.base_turn == 0.0


def test_inside_cone_keeps_current_yaw():
    # target 20 deg left, torso already at 10 deg: relative bearing 10 deg, no turn
    d = choose_waist_yaw((math.cos(D(20)), math.sin(D(20)), 1.0), D(10), CFG)
    assert d.yaw == pytest.approx(D(10)) and d.turn == 0.0


def test_left_table_turns_left_minimally():
    # table bag at bearing 95 deg: turn left until it sits on the 30 deg cone edge
    p = (math.cos(D(95)), math.sin(D(95)), 0.97)
    d = choose_waist_yaw(p, 0.0, CFG)
    assert d.yaw == pytest.approx(D(65))
    assert d.rel_bearing == pytest.approx(D(30))
    assert not d.needs_base


def test_right_rack_turns_right_minimally():
    p = (math.cos(D(-110)), math.sin(D(-110)), 1.2)
    d = choose_waist_yaw(p, 0.0, CFG)
    assert d.yaw == pytest.approx(D(-80))
    assert d.rel_bearing == pytest.approx(D(-30))
    assert not d.needs_base


def test_turn_back_from_table_to_crate_is_minimal():
    d = choose_waist_yaw((0.6, 0.0, 1.1), D(65), CFG)
    assert d.yaw == pytest.approx(D(30))   # crate now on the right cone edge


def test_behind_is_flagged_and_needs_base():
    d = choose_waist_yaw((-1.0, 0.0, 1.0), 0.0, CFG)
    assert d.needs_base
    assert abs(d.yaw) == pytest.approx(D(100))            # max turn
    assert abs(d.base_turn) == pytest.approx(D(50))       # 180 - 100 - 30
    assert math.copysign(1, d.base_turn) == math.copysign(1, d.yaw)   # same direction


def test_max_turn_clamp_and_base_remainder():
    p = (math.cos(D(150)), math.sin(D(150)), 1.0)
    d = choose_waist_yaw(p, 0.0, CFG)
    assert d.yaw == pytest.approx(D(100)) and d.needs_base
    assert d.base_turn == pytest.approx(D(20))


def test_joint_limits_clamp_when_max_turn_disabled():
    cfg = WaistConfig(pivot=PIVOT, max_turn_deg=None, cone_deg=0.0)
    d = choose_waist_yaw((-1.0, 0.001, 1.0), 0.0, cfg)
    assert d.yaw == pytest.approx(3.0456)
    assert d.needs_base and d.base_turn == pytest.approx(math.pi - 3.0456, abs=2e-3)


def test_clamp_waist():
    assert clamp_waist(D(150), CFG) == pytest.approx(D(100))
    assert clamp_waist(D(-150), CFG) == pytest.approx(D(-100))
    assert clamp_waist(D(40), CFG) == pytest.approx(D(40))
    assert clamp_waist(5.0, WaistConfig(max_turn_deg=None)) == pytest.approx(3.0456)


def test_current_yaw_outside_max_turn_is_pulled_in():
    d = choose_waist_yaw((math.cos(D(120)), math.sin(D(120)), 1.0), D(110), CFG)
    assert d.yaw == pytest.approx(D(100)) and not d.needs_base


def test_target_on_pivot_never_turns():
    d = choose_waist_yaw((0.01, 0.0, 1.0), D(5), CFG)
    assert d.yaw == pytest.approx(D(5)) and not d.needs_base


def test_waist_candidates_start_with_choice_and_end_with_current():
    c = waist_candidates(D(60), 0.0, CFG, step_deg=15, count=2)
    assert c[0] == pytest.approx(D(60))
    assert c[1:4] == pytest.approx([D(45), D(30), D(15)])     # back toward the current yaw first
    assert c[4:6] == pytest.approx([D(75), D(90)])            # then past the choice
    assert c[-1] == pytest.approx(0.0)
    assert len(set(round(v, 6) for v in c)) == len(c)
    assert all(abs(v) <= D(100) + 1e-9 for v in c)


# ---- twist filter / scoring ----------------------------------------------------------------
def test_home_like_solution_is_valid():
    assert twist_violations(dict(HOME), HOME, HOME, NAMES) == []


def test_shoulder_swung_backwards_is_rejected():
    s = dict(HOME)
    s[NAMES[0]] = D(120)
    v = twist_violations(s, HOME, HOME, NAMES)
    assert v and "joint1" in v[0]


def test_wrist_flip_jump_is_rejected():
    for k in (3, 5, 7):
        s = dict(HOME)
        s[NAMES[k - 1]] += 1.6
        v = twist_violations(s, HOME, HOME, NAMES)
        assert v and f"joint{k}" in v[0]
    # joint 2 jumps are not "twists"
    s = dict(HOME)
    s[NAMES[1]] += 1.6
    assert twist_violations(s, HOME, HOME, NAMES) == []


def test_custom_limits():
    s = dict(HOME)
    s[NAMES[0]] = D(50)
    assert twist_violations(s, HOME, HOME, NAMES, TwistLimits(j1_max_dev_deg=45)) != []


def test_score_prefers_forward_home_posture():
    near = dict(HOME); near[NAMES[3]] += 0.2
    far = dict(HOME); far[NAMES[3]] += 1.0
    assert solution_score(near, HOME, HOME, NAMES) < solution_score(far, HOME, HOME, NAMES)
    assert solution_score(dict(HOME), HOME, HOME, NAMES) == 0.0


def test_best_solution_filters_then_scores():
    twisted = dict(HOME); twisted[NAMES[0]] = D(150)
    ok_far = dict(HOME); ok_far[NAMES[1]] += 0.8
    ok_near = dict(HOME); ok_near[NAMES[1]] += 0.1
    best, rejected = best_solution([twisted, ok_far, None, ok_near], HOME, HOME, NAMES)
    assert best == ok_near and len(rejected) == 1
    best, rejected = best_solution([twisted], HOME, HOME, NAMES)
    assert best is None and len(rejected) == 1


# ---- synchronised interpolation --------------------------------------------------------------
def test_synced_chain_adds_waist_to_every_waypoint():
    sols = [{"a": 1.0}, {"a": 2.0}]
    out = synced_chain(sols, [0.5, 0.7], "w")
    assert out == [{"a": 1.0, "w": 0.5}, {"a": 2.0, "w": 0.7}]
    with pytest.raises(ValueError):
        synced_chain(sols, [0.5], "w")


def test_waist_and_arm_interpolate_together():
    from g2cell_motion.geometry import interpolate
    a = {"arm": 0.0, "w": 0.0}
    b = synced_chain([{"arm": 1.0}], [D(60)], "w")[0]
    names = list(a)
    samples = interpolate([a[n] for n in names], [b[n] for n in names], 20)
    for q in samples:
        arm_frac = q[0] / 1.0
        waist_frac = q[1] / D(60)
        assert arm_frac == pytest.approx(waist_frac)   # same quintic phase: synchronised
    assert samples[-1] == pytest.approx([1.0, D(60)])
    # smooth: zero velocity at both ends (first/last steps are the smallest)
    steps = [samples[0][1]] + [s2[1] - s1[1] for s1, s2 in zip(samples, samples[1:])]
    assert steps[0] < steps[10] and steps[-1] < steps[10]
