import json

import pytest

from g2cell_datagen import store as S


@pytest.mark.parametrize("text,seeds", [
    ("0-4", [0, 1, 2, 3, 4]),
    ("3", [3]),
    ("0-2,10,7-8", [0, 1, 2, 10, 7, 8]),
    (" 1 , 1, 2 ", [1, 2]),
])
def test_parse_seeds(text, seeds):
    assert S.parse_seeds(text) == seeds


@pytest.mark.parametrize("bad", ["", "a", "5-2", "-3", "1,,2"])
def test_parse_seeds_rejects(bad):
    with pytest.raises(ValueError):
        S.parse_seeds(bad)


def test_episode_name():
    assert S.episode_name(7) == "ep_s00007"
    assert S.seed_of("ep_s00042") == 42
    assert S.seed_of("teleop_x") is None


def test_write_is_atomic_and_readable(tmp_path):
    st = S.EpisodeStore(tmp_path / "b")
    st.write("ep_s00001", {"status": "running", "seed": 1})
    p = tmp_path / "b" / "ep_s00001" / "episode.json"
    assert json.loads(p.read_text())["status"] == "running"
    assert not list(p.parent.glob("*.tmp"))
    assert st.status("ep_s00001") == "running"
    assert st.status("ep_s00002") is None


def test_corrupt_json_counts_as_not_done(tmp_path):
    st = S.EpisodeStore(tmp_path)
    (tmp_path / "ep_s00003").mkdir()
    (tmp_path / "ep_s00003" / "episode.json").write_text("{trunc")
    assert st.status("ep_s00003") == "corrupt"
    assert st.pending([3]) == [3]


def test_pending_skips_done_only(tmp_path):
    st = S.EpisodeStore(tmp_path)
    st.write(S.episode_name(0), {"status": "done"})
    st.write(S.episode_name(1), {"status": "running"})
    st.write(S.episode_name(2), {"status": "error"})
    st.write(S.episode_name(3), {"status": "discarded"})
    assert st.pending([0, 1, 2, 3, 4]) == [1, 2, 3, 4]
    assert st.pending([0, 1, 2, 3, 4], retry_errors=False) == [1, 3, 4]


def test_quarantine_moves_leftovers(tmp_path):
    st = S.EpisodeStore(tmp_path)
    name = S.episode_name(5)
    st.write(name, {"status": "running"})
    (tmp_path / name / "bag").mkdir()
    (tmp_path / name / "bag" / "x.mcap").write_text("partial")
    dest = st.quarantine(name)
    assert not (tmp_path / name).exists()
    assert dest.parent == tmp_path / "_crashed" and (dest / "bag" / "x.mcap").exists()
    # again: a second crash of the same seed gets its own folder
    st.write(name, {"status": "running"})
    dest2 = st.quarantine(name)
    assert dest2 != dest and dest2.exists()
    assert st.quarantine("ep_s09999") is None


def test_summary(tmp_path):
    st = S.EpisodeStore(tmp_path)
    st.write("ep_s00000", {"status": "done", "outcome": "unverified", "success": "unknown"})
    st.write("ep_s00001", {"status": "done", "outcome": "plan_failed", "success": "false"})
    st.write("ep_s00002", {"status": "error", "error": "boom"})
    s = st.summary()
    assert s["status"] == {"done": 2, "error": 1}
    assert s["outcome"] == {"plan_failed": 1, "unverified": 1}
    assert s["success"] == {"false": 1, "unknown": 1}
    assert [r["name"] for r in st.records()] == ["ep_s00000", "ep_s00001", "ep_s00002"]
