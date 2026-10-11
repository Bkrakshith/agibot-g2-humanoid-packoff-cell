import json

from g2cell_motion.plan_cache import PlanCache, cache_key, layout_constants_hash, layout_key

BUILD = '''"""doc"""
import math
BELT_TOP = 0.820            # comment
LAYOUT = {"conveyor": (0.60, 0.0), "bags": [("Bag_01", 0.05, 0.62, 0.0)]}
helper = 3

def build():
    return LAYOUT
'''


def test_cache_key_is_stable_and_rounds_float_noise():
    a = cache_key({"arm": "left", "pts": [(0.1, 0.2, 0.3)], "base": (0.0, 0.0, 0.0)})
    b = cache_key({"base": (0.0, 0.0, 1e-9), "pts": [[0.1, 0.2, 0.3]], "arm": "left"})
    c = cache_key({"arm": "left", "pts": [(0.1, 0.2, 0.31)], "base": (0.0, 0.0, 0.0)})
    assert a == b and a != c


def test_layout_hash_ignores_comments_and_functions(tmp_path):
    p = tmp_path / "build.py"
    p.write_text(BUILD)
    h0 = layout_constants_hash(p)
    p.write_text(BUILD.replace("# comment", "# other words").replace("return LAYOUT", "return None")
                 .replace("helper = 3", "helper = 4"))
    assert layout_constants_hash(p) == h0


def test_layout_hash_changes_with_layout_constants(tmp_path):
    p = tmp_path / "build.py"
    p.write_text(BUILD)
    h0 = layout_constants_hash(p)
    p.write_text(BUILD.replace("0.05, 0.62", "0.05, 0.64"))
    h1 = layout_constants_hash(p)
    p.write_text(BUILD.replace("0.820", "0.83"))
    assert len({h0, h1, layout_constants_hash(p)}) == 3


def test_layout_key_uses_whatever_is_available(tmp_path):
    b = tmp_path / "build.py"
    b.write_text(BUILD)
    c = tmp_path / "cell_colliders.json"
    c.write_text("[]")
    k_both = layout_key(build_py=b, colliders=c)
    k_missing = layout_key(build_py=tmp_path / "nope.py", colliders=c)
    assert k_both != k_missing and k_missing == layout_key(colliders=c)
    c.write_text('[{"name": "x"}]')
    assert layout_key(build_py=b, colliders=c) != k_both


def test_round_trip_and_invalidation_on_layout_change(tmp_path):
    f = tmp_path / "cache.json"
    c = PlanCache(f, "layoutA")
    assert c.get({"t": 1}) is None
    c.put({"t": 1}, {"path": [1, 2]})
    c.save()
    c2 = PlanCache(f, "layoutA")
    assert c2.get({"t": 1}) == {"path": [1, 2]} and not c2.invalidated
    c3 = PlanCache(f, "layoutB")             # build.py layout constants changed
    assert c3.invalidated and c3.get({"t": 1}) is None
    c3.save()
    assert json.loads(f.read_text())["layout"] == "layoutB"
    assert PlanCache(f, "layoutA").get({"t": 1}) is None   # old entries are gone


def test_version_bump_invalidates(tmp_path):
    f = tmp_path / "cache.json"
    c = PlanCache(f, "L", version=1)
    c.put({"t": 1}, 5)
    c.save()
    assert PlanCache(f, "L", version=2).get({"t": 1}) is None


def test_disabled_and_unwritable_paths_never_raise(tmp_path):
    c = PlanCache(None, "L")
    c.put({"t": 1}, 5)
    assert c.get({"t": 1}) == 5
    c.save()                                  # in-memory only
    bad = tmp_path / "file"
    bad.write_text("x")
    c2 = PlanCache(bad / "sub" / "cache.json", "L")   # parent is a file -> cannot write
    c2.put({"t": 1}, 5)
    c2.save()
    assert c2.get({"t": 1}) == 5


def test_corrupt_file_is_ignored(tmp_path):
    f = tmp_path / "cache.json"
    f.write_text("{not json")
    c = PlanCache(f, "L")
    assert c.get({"t": 1}) is None and c.invalidated


def test_goal_key_rounds_the_start_state_and_separates_goals():
    from g2cell_motion.plan_cache import goal_key
    goal = {"named_pose": "", "frame": "cell_tool", "pose": [0.6, 0.0, 1.1, 0, 0, 0, 0], "via": [],
            "waist_mode": "waist_first", "waist_yaw": 0.0, "auto_orientation": True}
    cur = {"j1": 0.1, "j2": -0.5}
    k = goal_key("left", goal, cur, {"head": 0.2})
    assert k == goal_key("left", goal, {"j1": 0.1 + 2e-6, "j2": -0.5}, {"head": 0.2})   # sensor noise
    assert k != goal_key("left", goal, {"j1": 0.11, "j2": -0.5}, {"head": 0.2})
    assert k != goal_key("right", goal, cur, {"head": 0.2})
    assert k != goal_key("left", {**goal, "pose": [0.6, 0.0, 1.2, 0, 0, 0, 0]}, cur, {"head": 0.2})
    assert k != goal_key("left", goal, cur, {"head": 0.25})
