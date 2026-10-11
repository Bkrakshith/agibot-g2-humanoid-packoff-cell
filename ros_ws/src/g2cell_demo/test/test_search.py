"""Pruned pick->place search (g2cell_demo.search) must find exactly what the old brute-force loop found."""
import itertools
import random

from g2cell_demo.search import LazySeq, combine_info, memo_check, pick_place_search


def _info(env, name="b", head=0.3, idle=0.1, idle_name="i"):
    return {"fixture_clearance_m": env, "closest": name, "head_clearance_m": head,
            "idle_arm_clearance_m": idle, "idle_closest": idle_name}


def test_lazy_seq_evaluates_once_and_lazily():
    made = []

    def gen():
        for i in range(5):
            made.append(i)
            yield i
    s = LazySeq(gen())
    assert list(itertools.islice(iter(s), 2)) == [0, 1] and made == [0, 1]
    assert list(s) == [0, 1, 2, 3, 4] and list(s) == [0, 1, 2, 3, 4] and made == [0, 1, 2, 3, 4]


def test_combine_info_takes_the_minimum_with_its_name():
    c = combine_info([_info(0.05, "a", 0.3, 0.2, "x"), _info(0.02, "b", 0.25, 0.3, "y"), _info(0.04, "c", 0.4, 0.1, "z")])
    assert c == _info(0.02, "b", 0.25, 0.1, "z")


def test_memo_check_caches_by_path_content():
    calls = []

    def check(path):
        calls.append(path)
        return True, _info(0.1)
    m = memo_check(check)
    m([{"a": 1.0}, {"a": 2.0}])
    m([{"a": 1.0}, {"a": 2.0}])
    m([{"a": 1.0}, {"a": 2.5}])
    assert len(calls) == 2


def _brute(pick_waists, pick_chains, place_waists, place_chains, parts, seg_ok):
    """The old nested loop: full path = A + B + C checked at the innermost level."""
    for wp in pick_waists:
        for o1, pick in pick_chains(wp):
            for wd in place_waists:
                for o2, plc in place_chains(wd):
                    a, b, c = parts(wp, pick, wd, plc)
                    if all(seg_ok(x) for x in (a, b, c)):
                        return wp, o1, wd, o2
    return None


def test_pruned_search_matches_brute_force_on_random_problems():
    for trial in range(200):
        rng = random.Random(trial)
        pw, dw = [10, 20, 30], [0, -15, 15]
        good = {k: rng.random() < 0.5 for k in [("A", w, o) for w in pw for o in range(3)] +
                [("B", w, d) for w in pw for d in dw] + [("C", d, o) for d in dw for o in range(3)]}
        avail = {("p", w, o): rng.random() < 0.7 for w in pw for o in range(3)}
        avail.update({("d", d, o): rng.random() < 0.7 for d in dw for o in range(3)})

        def pick_chains(w):
            return [(o, ("pick", w, o)) for o in range(3) if avail[("p", w, o)]]

        def place_chains(d):
            return [(o, ("plc", d, o)) for o in range(3) if avail[("d", d, o)]]

        def parts(wp, pick, wd, plc):
            return ("A", wp, pick[2]), ("B", wp, wd), ("C", wd, plc[2])

        exp = _brute(pw, pick_chains, dw, place_chains, parts, lambda x: good[x])
        got = pick_place_search(pw, pick_chains, dw, place_chains,
                                part_a=lambda wp, pick: ("A", wp, pick[2]), part_b=lambda wp, wd: ("B", wp, wd),
                                part_c=lambda wd, plc: ("C", wd, plc[2]),
                                check=lambda x: (good[x], _info(0.1 if good[x] else -0.1)))
        assert (got and (got["wp"], got["o1"], got["wd"], got["o2"])) == (exp or None), trial


def test_failed_pick_part_skips_all_place_work():
    place_calls = []

    def place_chains(d):
        place_calls.append(d)
        return [(0, "plc")]
    r = pick_place_search([1], lambda w: [(0, "pick")], [5, 6], place_chains,
                          part_a=lambda wp, p: "A", part_b=lambda wp, wd: "B", part_c=lambda wd, p: "C",
                          check=lambda x: (x != "A", _info(-0.1 if x == "A" else 0.1)))
    assert r is None and place_calls == []


def test_result_carries_the_combined_info():
    infos = {"A": _info(0.05, "a"), "B": _info(0.03, "b"), "C": _info(0.04, "c")}
    r = pick_place_search([1], lambda w: [("o1", "pick")], [2], lambda d: [("o2", "plc")],
                          part_a=lambda wp, p: "A", part_b=lambda wp, wd: "B", part_c=lambda wd, p: "C",
                          check=lambda x: (True, infos[x]))
    assert r["pick"] == "pick" and r["plc"] == "plc" and r["info"]["fixture_clearance_m"] == 0.03
    assert r["info"]["closest"] == "b"
