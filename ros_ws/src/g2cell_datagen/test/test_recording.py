"""EpisodeRecorder against a fake record_episode.sh (same CLI: <name> --scene S --seed N, EPISODES_DIR,
stops on SIGINT, writes bag/metadata.yaml and <name>.hdf5) and a fake mark_episode."""
import stat
import textwrap

import pytest

from g2cell_datagen.recording import EpisodeRecorder, judge


def _script(path, body):
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fakes(tmp_path):
    rec = _script(tmp_path / "record.sh", """
        name=$1; shift; echo "$@" > "$EPISODES_DIR/$name.args"
        d=$EPISODES_DIR/$name; mkdir -p "$d/bag"
        trap 'echo "x: 1" > "$d/bag/metadata.yaml"; [ -n "${NO_H5:-}" ] && exit 3; echo h5 > "$d/$name.hdf5"; exit 0' INT
        while true; do sleep 0.05; done
        """)
    mark = _script(tmp_path / "mark.sh", 'echo "$2 $3" > "$1.mark"\n')
    return rec, mark


def test_record_stop_convert_and_mark(tmp_path, fakes, monkeypatch):
    rec, mark = fakes
    out = tmp_path / "eps"
    out.mkdir()
    r = EpisodeRecorder(out, scene="scene_g2cell", record_cmd=[str(rec)], mark_cmd=[str(mark)], warmup_s=0.2,
                        convert_timeout_s=10, log=lambda *_: None)
    r.start("ep_s00001", 1)
    assert (out / "ep_s00001.args").read_text().split() == ["--scene", "scene_g2cell", "--seed", "1"]
    res = r.stop("ep_s00001", "true")
    assert res.ok and res.returncode == 0 and res.hdf5.endswith("ep_s00001/ep_s00001.hdf5")
    assert (out / "ep_s00001" / "ep_s00001.hdf5.mark").read_text().split() == ["--success", "true"]


def test_unknown_outcome_is_not_marked(tmp_path, fakes):
    rec, mark = fakes
    r = EpisodeRecorder(tmp_path, record_cmd=[str(rec)], mark_cmd=[str(mark)], warmup_s=0.2, log=lambda *_: None)
    r.start("ep_s00002", 2)
    assert r.stop("ep_s00002", "unknown").ok
    assert not (tmp_path / "ep_s00002" / "ep_s00002.hdf5.mark").exists()


def test_deferred_conversion(tmp_path, fakes, monkeypatch):
    rec, mark = fakes
    monkeypatch.setenv("NO_H5", "1")
    r = EpisodeRecorder(tmp_path, record_cmd=[str(rec)], mark_cmd=[str(mark)], warmup_s=0.2, log=lambda *_: None)
    r.start("ep_s00003", 3)
    res = r.stop("ep_s00003", "false")
    assert res.ok and res.returncode == 3 and res.hdf5 == "" and "finalize" in res.error
    r2 = EpisodeRecorder(tmp_path, record_cmd=[str(rec)], mark_cmd=[str(mark)], warmup_s=0.2,
                         allow_deferred=False, log=lambda *_: None)
    r2.start("ep_s00004", 4)
    assert not r2.stop("ep_s00004", "false").ok


def test_recorder_dying_at_start_raises(tmp_path):
    dead = _script(tmp_path / "dead.sh", "exit 1\n")
    r = EpisodeRecorder(tmp_path, record_cmd=[str(dead)], warmup_s=0.3, log=lambda *_: None)
    with pytest.raises(RuntimeError):
        r.start("ep_s00005", 5)


def test_judge(tmp_path):
    bag, h5 = tmp_path / "bag", tmp_path / "e.hdf5"
    assert not judge(1, bag, h5).ok
    assert not judge(None, bag, h5).ok
    bag.mkdir()
    (bag / "metadata.yaml").write_text("x")
    assert judge(3, bag, h5).ok and not judge(3, bag, h5, allow_deferred=False).ok
    assert not judge(0, bag, h5).ok
    h5.write_text("x")
    assert judge(0, bag, h5).ok
