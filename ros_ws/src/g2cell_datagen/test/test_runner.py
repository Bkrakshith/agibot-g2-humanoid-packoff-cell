from pathlib import Path

import pytest
import yaml

from g2cell_datagen import randomize as R
from g2cell_datagen import runner as RU
from g2cell_datagen.store import EpisodeStore, episode_name

PKG = Path(__file__).resolve().parents[1]
TASKS_YAML = PKG.parent / "g2cell_tasks" / "config" / "tasks.yaml"


@pytest.fixture(scope="module")
def cfg():
    return R.load_config(yaml.safe_load((PKG / "config" / "datagen.yaml").read_text()),
                         yaml.safe_load(TASKS_YAML.read_text()))


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeBackend:
    """Records the call order; outcomes / failures injected per seed."""

    def __init__(self, clock, outcomes=None, fail=None, rec_ok=True):
        self.clock, self.calls = clock, []
        self.outcomes = outcomes or {}
        self.fail = fail or {}           # (phase, seed) -> exception
        self.rec_ok = rec_ok
        self.recording = None
        self.closed = False

    def _maybe_fail(self, phase, spec):
        exc = self.fail.get((phase, spec.seed))
        if exc is not None:
            raise exc

    def reset(self, name, spec):
        self.calls.append(("reset", name))
        self.clock.t += 2.0
        self._maybe_fail("reset", spec)
        return {"mode": "fake"}

    def start_recording(self, name, spec):
        self.calls.append(("start", name))
        self._maybe_fail("start", spec)
        self.recording = name
        self.clock.t += 0.5

    def run_task(self, name, spec):
        self.calls.append(("task", name))
        self.clock.t += 30.0
        self._maybe_fail("task", spec)
        return RU.TaskResult(self.outcomes.get(spec.seed, "unverified"), message="ok", arm="left")

    def stop_recording(self, name, success):
        self.calls.append(("stop", name, success))
        assert self.recording == name
        self.recording = None
        self.clock.t += 4.0
        if not self.rec_ok:
            return RU.RecordingResult(False, returncode=3, error="no h5py")
        return RU.RecordingResult(True, hdf5=f"/x/{name}/{name}.hdf5", returncode=0)

    def close(self):
        self.closed = True


def make(tmp_path, cfg, **kw):
    clock = Clock()
    be = FakeBackend(clock, **{k: kw.pop(k) for k in ("outcomes", "fail", "rec_ok") if k in kw})
    store = EpisodeStore(tmp_path / "batch")
    run = RU.EpisodeRunner(cfg, be, store, clock=clock, now=lambda: 1_700_000_000.0, log=lambda *_: None,
                           batch="batch", **kw)
    return run, be, store


@pytest.mark.parametrize("outcome,unverified_as,success", [
    ("success", "unknown", "true"), ("regrasp", "unknown", "true"),
    ("unverified", "unknown", "unknown"), ("unverified", "true", "true"),
    ("plan_failed", "unknown", "false"), ("grasp_failed", "true", "false"), ("drop", "true", "false"),
    ("wrong_target", "unknown", "false"), ("unreachable", "unknown", "false"),
])
def test_success_of(outcome, unverified_as, success):
    assert RU.success_of(outcome, unverified_as) == success


def test_one_episode_happy_path(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg)
    rec = run.run_episode(4)
    name = episode_name(4)
    assert [c[0] for c in be.calls] == ["reset", "start", "task", "stop"]
    assert be.calls[-1] == ("stop", name, "unknown")
    assert rec["status"] == "done" and rec["outcome"] == "unverified" and rec["success"] == "unknown"
    assert rec["randomisation"] == R.sample(cfg, 4).to_dict()
    assert rec["timings"] == {"reset_s": 2.0, "record_start_s": 0.5, "task_s": 30.0, "stop_s": 4.0,
                              "total_s": 36.5}
    assert rec["recording"]["hdf5"].endswith(f"{name}.hdf5")
    assert store.load(name) == rec


def test_success_mark_goes_to_the_recorder(tmp_path, cfg):
    run, be, _ = make(tmp_path, cfg, outcomes={1: "success", 2: "plan_failed"})
    run.run([1, 2])
    stops = [c for c in be.calls if c[0] == "stop"]
    assert stops == [("stop", episode_name(1), "true"), ("stop", episode_name(2), "false")]


def test_batch_resume_skips_done_and_quarantines_crashed(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg)
    run.run([0, 1])
    # a crash mid-episode 2 left a "running" json and a partial bag
    store.write(episode_name(2), {"status": "running"})
    (store.episode_dir(episode_name(2)) / "bag").mkdir()
    be.calls.clear()
    summary = run.run([0, 1, 2, 3])
    assert [c[1] for c in be.calls if c[0] == "reset"] == [episode_name(2), episode_name(3)]
    assert summary["ran"] == 2 and summary["skipped"] == 2
    assert (store.dir / "_crashed" / f"{episode_name(2)}.1" / "bag").is_dir()
    assert store.load(episode_name(2))["attempt"] == 2
    assert store.status(episode_name(2)) == "done"


def test_task_exception_stops_recording_and_marks_error(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg, fail={("task", 1): RuntimeError("action server gone")})
    summary = run.run([0, 1, 2])
    assert ("stop", episode_name(1), "unknown") in be.calls
    rec = store.load(episode_name(1))
    assert rec["status"] == "error" and "action server gone" in rec["error"] and rec["phase"] == "task"
    assert summary["status"] == {"done": 2, "error": 1}
    assert store.pending([0, 1, 2]) == [1]                       # retried on the next run


def test_reset_failure_never_records(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg, fail={("reset", 0): RuntimeError("sim did not come up")})
    run.run([0])
    assert [c[0] for c in be.calls] == ["reset"]
    assert store.load(episode_name(0))["phase"] == "reset"


def test_infra_outcomes_are_errors(tmp_path, cfg):
    run, _, store = make(tmp_path, cfg, outcomes={0: "aborted"})
    run.run([0])
    rec = store.load(episode_name(0))
    assert rec["status"] == "error" and rec["outcome"] == "aborted"


def test_recording_failure_is_error_but_keeps_outcome(tmp_path, cfg):
    run, _, store = make(tmp_path, cfg, outcomes={0: "success"}, rec_ok=False)
    run.run([0])
    rec = store.load(episode_name(0))
    assert rec["status"] == "error" and rec["phase"] == "recording"
    assert rec["outcome"] == "success" and rec["success"] == "true"


def test_consecutive_errors_abort_the_batch(tmp_path, cfg):
    fail = {("reset", s): RuntimeError("down") for s in range(10)}
    run, be, _ = make(tmp_path, cfg, fail=fail, max_consecutive_errors=3)
    summary = run.run(list(range(10)))
    assert summary["ran"] == 3 and summary["aborted"]


def test_keyboard_interrupt_closes_the_bag_and_reraises(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg, fail={("task", 0): KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run.run([0, 1])
    assert be.calls[-1] == ("stop", episode_name(0), "unknown")
    assert store.status(episode_name(0)) == "interrupted"
    assert store.pending([0, 1]) == [0, 1]


def test_limit(tmp_path, cfg):
    run, be, _ = make(tmp_path, cfg)
    s = run.run(list(range(10)), limit=3)
    assert s["ran"] == 3


def test_dry_plan_touches_nothing(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg)
    plan = run.plan([3, 4])
    assert [p["seed"] for p in plan] == [3, 4] and be.calls == []
    assert not store.dir.exists()


# ---- operator in the loop ----------------------------------------------------------------------
def test_operator_mode_wraps_teleop(tmp_path, cfg):
    answers = iter(["success", "failure", "discard"])
    asked = []

    def operator(name, spec):
        asked.append(name)
        return next(answers)

    run, be, store = make(tmp_path, cfg, mode="operator", operator=operator)
    summary = run.run([0, 1, 2])
    assert asked == [episode_name(i) for i in range(3)]
    assert ("task", episode_name(0)) not in be.calls             # the backend's scripted task is not used
    assert [c for c in be.calls if c[0] == "stop"] == [
        ("stop", episode_name(0), "true"), ("stop", episode_name(1), "false"), ("stop", episode_name(2), "unknown")]
    assert store.load(episode_name(0))["outcome"] == "operator_success"
    assert store.status(episode_name(2)) == "discarded"
    assert summary["status"] == {"discarded": 1, "done": 2}
    assert store.pending([0, 1, 2]) == [2]


def test_operator_quit_stops_after_closing(tmp_path, cfg):
    run, be, store = make(tmp_path, cfg, mode="operator", operator=lambda n, s: "quit")
    summary = run.run([0, 1])
    assert summary["ran"] == 1 and summary["quit"]
    assert be.calls[-1] == ("stop", episode_name(0), "unknown")
    assert store.status(episode_name(0)) == "discarded"


def test_operator_mode_needs_operator(cfg, tmp_path):
    with pytest.raises(ValueError):
        RU.EpisodeRunner(cfg, FakeBackend(Clock()), EpisodeStore(tmp_path), mode="operator")
