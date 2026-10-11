"""Episode state machine (pure; the sim side is an injected Backend).

Per seed:  sample randomisation -> RESET (backend.reset) -> RECORD (backend.start_recording)
           -> TASK (backend.run_task, or the operator in operator mode) -> STOP + MARK
           (backend.stop_recording(success)) -> episode.json status done | error | discarded.
episode.json is written as "running" before anything moves, so a crash leaves a visible trace; on the
next run that attempt is moved to _crashed/ and the seed runs again. Done seeds are skipped.
"""
from __future__ import annotations

import datetime as _dt
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from g2cell_datagen.randomize import DatagenConfig, EpisodeSpec, sample
from g2cell_datagen.store import EpisodeStore, episode_name

SUCCESS_OUTCOMES = ("success", "regrasp")
INFRA_OUTCOMES = ("aborted", "invalid")    # the task never really ran: redo the seed
OPERATOR_VERDICTS = ("success", "failure", "discard", "quit")


@dataclass
class TaskResult:
    outcome: str                 # g2cell_tasks outcome name (success, unverified, plan_failed, ...)
    message: str = ""
    arm: str = ""
    verified: bool = False
    grasp_attempts: int = 0
    failed_stage: str = ""
    duration: float = 0.0
    task_outcome: str = ""       # what pick_bag_server reported, when the scorer re-labelled it
    placement: dict | None = None   # scene_io.placement_of after the task (teleport/score mode)


@dataclass
class RecordingResult:
    ok: bool
    hdf5: str = ""
    returncode: int | None = None
    error: str = ""


class Backend(Protocol):
    def reset(self, name: str, spec: EpisodeSpec) -> dict: ...
    def start_recording(self, name: str, spec: EpisodeSpec) -> None: ...
    def run_task(self, name: str, spec: EpisodeSpec) -> TaskResult: ...
    def stop_recording(self, name: str, success: str) -> RecordingResult: ...
    def close(self) -> None: ...


def success_of(outcome: str, unverified_as: str = "unknown") -> str:
    """HDF5 `success` attr from a PickBag outcome. `unverified` = the motion finished but nothing
    observed the placement (no Stage 5 scorer yet): `unverified_as` decides (default unknown)."""
    if outcome in SUCCESS_OUTCOMES:
        return "true"
    if outcome == "unverified":
        return unverified_as
    if outcome in INFRA_OUTCOMES:
        return "unknown"
    return "false"


class _Quit(Exception):
    pass


class EpisodeRunner:
    def __init__(self, cfg: DatagenConfig, backend: Backend, store: EpisodeStore, mode: str = "scripted",
                 operator: Callable[[str, EpisodeSpec], str] | None = None, unverified_as: str = "unknown",
                 batch: str = "", retry_errors: bool = True, max_consecutive_errors: int = 3,
                 clock=time.monotonic, now=time.time, log=print):
        if mode not in ("scripted", "operator"):
            raise ValueError(f"mode must be scripted or operator, got {mode!r}")
        if mode == "operator" and operator is None:
            raise ValueError("operator mode needs an operator callback")
        if unverified_as not in ("true", "false", "unknown"):
            raise ValueError("unverified_as must be true, false or unknown")
        self.cfg, self.backend, self.store = cfg, backend, store
        self.mode, self.operator, self.unverified_as = mode, operator, unverified_as
        self.batch, self.retry_errors = batch, retry_errors
        self.max_consecutive_errors = max_consecutive_errors
        self.clock, self.now, self.log = clock, now, log

    def _iso(self) -> str:
        return _dt.datetime.fromtimestamp(self.now(), _dt.timezone.utc).isoformat(timespec="seconds")

    def plan(self, seeds) -> list:
        return [sample(self.cfg, s).to_dict() for s in seeds]

    # ---- one episode ----------------------------------------------------------------------------
    def run_episode(self, seed: int) -> dict:
        name = episode_name(seed)
        prev = self.store.load(name)
        attempt = 1
        if prev is not None:
            attempt = int(prev.get("attempt", 1)) + 1
            moved = self.store.quarantine(name)
            self.log(f"{name}: previous attempt ({prev.get('status')}) moved to {moved}")
        spec = sample(self.cfg, seed)
        rec = {"name": name, "seed": int(seed), "batch": self.batch, "mode": self.mode, "status": "running",
               "phase": "reset", "attempt": attempt, "randomisation": spec.to_dict(), "reset": {},
               "outcome": "", "success": "", "task": {}, "recording": {}, "timings": {},
               "started_at": self._iso(), "finished_at": "", "error": ""}
        self.store.write(name, rec)
        t0 = self.clock()
        tm = rec["timings"]
        recording = False

        def lap(key, since):
            tm[key] = round(self.clock() - since, 3)
            return self.clock()

        def stop(success):
            nonlocal recording
            if not recording:
                return None
            recording = False
            t = self.clock()
            r = self.backend.stop_recording(name, success)
            lap("stop_s", t)
            rec["recording"] = asdict(r)
            return r

        try:
            t = self.clock()
            rec["reset"] = self.backend.reset(name, spec) or {}
            t = lap("reset_s", t)
            rec["phase"] = "record"
            self.store.write(name, rec)
            self.backend.start_recording(name, spec)
            recording = True
            t = lap("record_start_s", t)
            rec["phase"] = "task"
            self.store.write(name, rec)
            if self.mode == "operator":
                verdict = self.operator(name, spec)
                if verdict not in OPERATOR_VERDICTS:
                    raise ValueError(f"operator verdict must be one of {OPERATOR_VERDICTS}, got {verdict!r}")
                lap("task_s", t)
                if verdict in ("discard", "quit"):
                    stop("unknown")
                    rec.update(status="discarded", phase="done", outcome=f"operator_{verdict}", success="unknown")
                    if verdict == "quit":
                        raise _Quit()
                    return self._finish(name, rec, t0)
                rec["outcome"] = f"operator_{verdict}"
                rec["success"] = "true" if verdict == "success" else "false"
            else:
                res = self.backend.run_task(name, spec)
                lap("task_s", t)
                rec["task"] = asdict(res)
                rec["outcome"] = res.outcome
                rec["success"] = success_of(res.outcome, self.unverified_as)
            rec["phase"] = "recording"
            r = stop(rec["success"])
            if r is not None and not r.ok:
                rec.update(status="error", error=f"recording failed: {r.error or r.returncode}")
            elif rec["outcome"] in INFRA_OUTCOMES:
                rec.update(status="error", error=f"task {rec['outcome']}: {rec['task'].get('message', '')}")
            else:
                rec.update(status="done", phase="done")
        except _Quit:
            self._finish(name, rec, t0)
            raise
        except KeyboardInterrupt:
            stop("unknown")
            rec.update(status="interrupted", error="KeyboardInterrupt")
            self._finish(name, rec, t0)
            raise
        except Exception as e:  # noqa: BLE001 - any backend failure ends this episode, not the batch
            try:
                stop("unknown")
            except Exception as e2:  # noqa: BLE001
                rec["recording"] = {"ok": False, "error": f"stop failed: {e2}"}
            rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=5))
        return self._finish(name, rec, t0)

    def _finish(self, name, rec, t0):
        rec["timings"]["total_s"] = round(self.clock() - t0, 3)
        rec["finished_at"] = self._iso()
        self.store.write(name, rec)
        self.log(f"{name}: {rec['status']} outcome={rec['outcome'] or '-'} success={rec['success'] or '-'}"
                 f"{' error=' + rec['error'] if rec['error'] else ''} ({rec['timings']['total_s']:.1f} s)")
        return rec

    # ---- batch --------------------------------------------------------------------------------
    def run(self, seeds, limit: int | None = None) -> dict:
        todo = self.store.pending(seeds, self.retry_errors)
        skipped = len(list(seeds)) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        self.log(f"batch {self.batch or self.store.dir}: {len(todo)} to run, {skipped} already done")
        ran, consecutive, aborted, quit_ = 0, 0, False, False
        for s in todo:
            try:
                rec = self.run_episode(s)
            except _Quit:
                ran += 1
                quit_ = True
                break
            ran += 1
            consecutive = consecutive + 1 if rec["status"] == "error" else 0
            if self.max_consecutive_errors and consecutive >= self.max_consecutive_errors:
                self.log(f"{consecutive} errors in a row: stopping the batch (check the sim)")
                aborted = True
                break
        return {"ran": ran, "skipped": skipped, "aborted": aborted, "quit": quit_, **self.store.summary()}
