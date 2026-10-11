import json
import signal
import sys
import time
from pathlib import Path

import pytest

from g2cell_teleop.recorder_control import RecorderController, default_episode_name

# Stands in for `ros2 run g2cell_recorder record_episode.sh NAME --scene S --seed N`:
# records its argv, waits for SIGINT like `ros2 bag record`, then "converts" (writes the HDF5).
FAKE_RECORDER = r'''
import json, os, signal, sys, time
name = sys.argv[1]
d = os.path.join(os.environ["EPISODES_DIR"], name)
os.makedirs(d, exist_ok=True)
stop = []
signal.signal(signal.SIGINT, lambda *a: stop.append(1))
json.dump({"argv": sys.argv[1:]}, open(os.path.join(d, "argv.json"), "w"))
open(os.path.join(d, "ready"), "w").close()
while not stop:
    time.sleep(0.01)
if os.environ.get("FAKE_FAIL"):
    sys.exit(3)
time.sleep(float(os.environ.get("FAKE_CONVERT_S", "0")))
open(os.path.join(d, name + ".hdf5"), "w").write("converted")
'''

# Stands in for `ros2 run g2cell_recorder mark_episode H5 --success X`.
FAKE_MARK = r'''
import sys
open(sys.argv[1] + ".mark", "a").write(sys.argv[3] + "\n")
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    rec = tmp_path / "fake_record.py"
    rec.write_text(FAKE_RECORDER)
    mark = tmp_path / "fake_mark.py"
    mark.write_text(FAKE_MARK)
    eps = tmp_path / "episodes"
    names = iter(f"ep{i}" for i in range(10))

    def make(**kw):
        return RecorderController(
            scene="scene_pnp_g2_op", seed=7, episodes_dir=str(eps),
            record_cmd=(sys.executable, str(rec)), mark_cmd=(sys.executable, str(mark)),
            name_fn=lambda: next(names), log=lambda *_: None, **kw)
    return make, eps, monkeypatch


def wait_ready(eps: Path, name: str, timeout=5.0):
    end = time.time() + timeout
    while not (eps / name / "ready").exists():
        assert time.time() < end, "fake recorder did not start"
        time.sleep(0.01)


def marks(eps: Path, name: str):
    p = eps / name / f"{name}.hdf5.mark"
    return p.read_text().split() if p.exists() else []


def test_start_launches_recorder_with_name_scene_and_seed(env):
    make, eps, _ = env
    rc = make()
    name = rc.start()
    assert name == "ep0" and rc.running
    wait_ready(eps, name)
    argv = json.loads((eps / name / "argv.json").read_text())["argv"]
    assert argv == ["ep0", "--scene", "scene_pnp_g2_op", "--seed", "7"]
    rc.stop()
    rc.wait(5)


def test_stop_sends_sigint_so_the_recorder_converts(env):
    make, eps, _ = env
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    assert rc.stop() is True
    assert rc.wait(5)
    assert not rc.running
    assert (eps / name / f"{name}.hdf5").read_text() == "converted"
    assert rc.last_returncode == 0


def test_outcome_marked_during_recording_is_written_after_conversion(env):
    make, eps, mp = env
    mp.setenv("FAKE_CONVERT_S", "0.3")
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    rc.mark("true")
    rc.stop()
    assert marks(eps, name) == []  # conversion still running
    assert rc.wait(5)
    assert marks(eps, name) == ["true"]


def test_outcome_passed_to_stop_is_written(env):
    make, eps, _ = env
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    rc.stop("false")
    rc.wait(5)
    assert marks(eps, name) == ["false"]


def test_mark_after_conversion_finished_is_written_immediately(env):
    make, eps, _ = env
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    rc.stop()
    rc.wait(5)
    assert marks(eps, name) == []
    rc.mark("false")
    assert marks(eps, name) == ["false"]
    rc.mark("true")  # changing your mind overwrites the attr
    assert marks(eps, name) == ["false", "true"]


def test_unknown_outcome_writes_nothing(env):
    make, eps, _ = env
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    rc.stop("unknown")
    rc.wait(5)
    assert marks(eps, name) == []


def test_new_recording_starts_with_unknown_outcome(env):
    make, eps, _ = env
    rc = make()
    a = rc.start()
    wait_ready(eps, a)
    rc.mark("true")
    rc.stop()
    rc.wait(5)
    b = rc.start()
    wait_ready(eps, b)
    rc.stop()
    rc.wait(5)
    assert marks(eps, a) == ["true"] and marks(eps, b) == []


def test_failed_conversion_is_not_marked(env):
    make, eps, mp = env
    mp.setenv("FAKE_FAIL", "1")
    rc = make()
    name = rc.start()
    wait_ready(eps, name)
    rc.stop("true")
    rc.wait(5)
    assert rc.last_returncode == 3
    assert marks(eps, name) == []
    assert "hdf5" in rc.last_error


def test_start_twice_raises_and_stop_when_idle_is_a_noop(env):
    make, eps, _ = env
    rc = make()
    assert rc.stop() is False
    name = rc.start()
    wait_ready(eps, name)
    with pytest.raises(RuntimeError):
        rc.start()
    rc.stop()
    rc.wait(5)


def test_mark_rejects_bad_outcome(env):
    make, _, _ = env
    with pytest.raises(ValueError):
        make().mark("maybe")


def test_default_episode_name_is_filesystem_safe_and_timestamped():
    name = default_episode_name("scene pnp/g2", now=time.struct_time((2026, 10, 11, 9, 5, 3, 0, 0, 0)))
    assert name == "teleop_scene_pnp_g2_20261011_090503"
    assert default_episode_name("", now=time.struct_time((2026, 1, 2, 3, 4, 5, 0, 0, 0))) == "teleop_20260102_030405"


def test_controller_uses_sigint_not_sigterm():
    assert RecorderController.STOP_SIGNAL == signal.SIGINT
