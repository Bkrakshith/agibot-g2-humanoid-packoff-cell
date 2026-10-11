"""Pick-and-place task loop as a pure state machine (motion and sensing are injected).

    approach -> grasp -> lift -> transfer -> place -> retract  (-> recover on failure)

runner: object with approach/grasp/lift/transfer/place/retract/release/recover(ctx) -> StepResult.
  release = open the gripper and back off after a missed grasp; recover = open + go home.
hooks:  grasp_detected(ctx) -> bool | None    after grasp (None = unknown, carry on)
        holding(ctx, stage) -> bool | None    after lift and after transfer
        observe_placement(ctx) -> Placement | None   after retract (Stage 5 scorer)
        classify(target, lane, placement, attempts, dropped=...) -> (outcome, verified)
Failure handling: a missed grasp or a slip during lift retries up to max_grasp_attempts; losing
the bag during transfer is a drop; any motion failure -> plan_failed + recover; cancel -> aborted
(no further motion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from g2cell_tasks.outcome import classify as default_classify


class Stage(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    RETRACT = "retract"
    RECOVER = "recover"


STAGE_INDEX = {s: i for i, s in enumerate(Stage)}   # = PickBag feedback STAGE_* constants


@dataclass
class StepResult:
    ok: bool
    message: str = ""
    canceled: bool = False


@dataclass
class TaskResult:
    outcome: str
    verified: bool = False
    grasp_attempts: int = 0
    failed_stage: str = ""
    message: str = ""
    placement: object = None


@dataclass
class Hooks:
    grasp_detected: Callable = lambda ctx: None
    holding: Callable = lambda ctx, stage: None
    observe_placement: Callable = lambda ctx: None
    classify: Callable = field(default=default_classify)


class PickPlaceMachine:
    def __init__(self, runner, hooks: Hooks | None = None, max_grasp_attempts: int = 2, on_stage=None):
        self.runner = runner
        self.hooks = hooks or Hooks()
        self.max_attempts = max(1, int(max_grasp_attempts))
        self.on_stage = on_stage or (lambda stage, attempt: None)

    def _do(self, stage: Stage, ctx, attempt: int) -> StepResult:
        self.on_stage(stage, attempt)
        return getattr(self.runner, stage.value)(ctx)

    def _fail(self, stage: Stage, r: StepResult, ctx, attempts: int) -> TaskResult:
        if r.canceled:
            return TaskResult("aborted", False, attempts, stage.value, r.message or "canceled")
        self.on_stage(Stage.RECOVER, attempts)
        rec = self.runner.recover(ctx)
        msg = f"{stage.value}: {r.message}" + ("" if rec.ok else f"; recovery failed: {rec.message}")
        return TaskResult("plan_failed", False, attempts, stage.value, msg)

    def run(self, ctx) -> TaskResult:
        attempt = 0
        while True:
            attempt += 1
            for stage in (Stage.APPROACH, Stage.GRASP):
                r = self._do(stage, ctx, attempt)
                if not r.ok:
                    return self._fail(stage, r, ctx, attempt)
            missed = self.hooks.grasp_detected(ctx) is False
            if not missed:
                r = self._do(Stage.LIFT, ctx, attempt)
                if not r.ok:
                    return self._fail(Stage.LIFT, r, ctx, attempt)
                missed = self.hooks.holding(ctx, Stage.LIFT) is False   # slipped back onto the table
                failed_at = Stage.LIFT
            else:
                failed_at = Stage.GRASP
            if not missed:
                break
            if attempt >= self.max_attempts:
                self.on_stage(Stage.RECOVER, attempt)
                self.runner.recover(ctx)
                return TaskResult("grasp_failed", True, attempt, failed_at.value,
                                  f"no grasp after {attempt} attempt(s)")
            r = self.runner.release(ctx)
            if not r.ok:
                return self._fail(failed_at, r, ctx, attempt)

        r = self._do(Stage.TRANSFER, ctx, attempt)
        if not r.ok:
            return self._fail(Stage.TRANSFER, r, ctx, attempt)
        if self.hooks.holding(ctx, Stage.TRANSFER) is False:
            self.on_stage(Stage.RECOVER, attempt)
            self.runner.recover(ctx)
            name, verified = self.hooks.classify(ctx["target"], ctx["lane"], None, attempt, dropped=True)
            return TaskResult(name, verified, attempt, Stage.TRANSFER.value, "bag lost during transfer")
        r = self._do(Stage.PLACE, ctx, attempt)
        if not r.ok:
            return self._fail(Stage.PLACE, r, ctx, attempt)
        failed, msg = "", ""
        r = self._do(Stage.RETRACT, ctx, attempt)
        if not r.ok:
            failed, msg = Stage.RETRACT.value, f"retract: {r.message}"
            if r.canceled:
                return TaskResult("aborted", False, attempt, failed, msg)
            self.on_stage(Stage.RECOVER, attempt)
            self.runner.recover(ctx)
        placement = self.hooks.observe_placement(ctx)
        name, verified = self.hooks.classify(ctx["target"], ctx["lane"], placement, attempt)
        if name == "unverified" and not msg:
            msg = "motion completed; placement not observed (Stage 5 scorer not implemented)"
        return TaskResult(name, verified, attempt, failed, msg, placement)
