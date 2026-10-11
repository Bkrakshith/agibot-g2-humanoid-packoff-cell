from g2cell_tasks.outcome import Placement
from g2cell_tasks.state_machine import Hooks, PickPlaceMachine, Stage, StepResult

OK = StepResult(True)


class FakeRunner:
    def __init__(self, fail=None, cancel=None):
        self.calls = []
        self.fail, self.cancel = fail or set(), cancel or set()

    def _step(self, name):
        self.calls.append(name)
        if name in self.cancel:
            return StepResult(False, "canceled", canceled=True)
        return StepResult(False, f"{name} failed") if name in self.fail else OK

    def __getattr__(self, name):
        if name in ("approach", "grasp", "lift", "transfer", "place", "retract", "release", "recover"):
            return lambda ctx: self._step(name)
        raise AttributeError(name)


def run(runner=None, hooks=None, max_attempts=2):
    stages = []
    r = runner or FakeRunner()
    m = PickPlaceMachine(r, hooks or Hooks(), max_grasp_attempts=max_attempts,
                         on_stage=lambda st, attempt: stages.append((st, attempt)))
    return m.run({"target": "crate", "lane": ""}), r, stages


def test_happy_path_without_observation_is_unverified():
    res, r, stages = run()
    assert r.calls == ["approach", "grasp", "lift", "transfer", "place", "retract"]
    assert [s for s, _ in stages] == [Stage.APPROACH, Stage.GRASP, Stage.LIFT, Stage.TRANSFER,
                                      Stage.PLACE, Stage.RETRACT]
    assert res.outcome == "unverified" and not res.verified and res.grasp_attempts == 1


def test_observed_placement_success():
    res, _, _ = run(hooks=Hooks(observe_placement=lambda ctx: Placement("crate")))
    assert res.outcome == "success" and res.verified


def test_missed_grasp_retries_then_regrasp_outcome():
    seq = iter([False, True])
    res, r, stages = run(hooks=Hooks(grasp_detected=lambda ctx: next(seq),
                                     observe_placement=lambda ctx: Placement("crate")))
    assert r.calls[:5] == ["approach", "grasp", "release", "approach", "grasp"]
    assert (Stage.APPROACH, 2) in stages
    assert res.outcome == "regrasp" and res.grasp_attempts == 2


def test_all_grasps_missed():
    res, r, _ = run(hooks=Hooks(grasp_detected=lambda ctx: False), max_attempts=2)
    assert res.outcome == "grasp_failed" and res.failed_stage == "grasp"
    assert r.calls.count("grasp") == 2 and r.calls[-1] == "recover"


def test_slip_during_lift_counts_as_missed_grasp():
    seq = iter([False, True])
    res, r, _ = run(hooks=Hooks(holding=lambda ctx, stage: next(seq) if stage == Stage.LIFT else True))
    assert r.calls.count("grasp") == 2 and res.grasp_attempts == 2


def test_lost_during_transfer_is_drop():
    res, r, _ = run(hooks=Hooks(holding=lambda ctx, stage: stage != Stage.TRANSFER))
    assert res.outcome == "drop" and res.verified and res.failed_stage == "transfer"
    assert "place" not in r.calls and r.calls[-1] == "recover"


def test_motion_failure_is_plan_failed_with_recovery():
    res, r, _ = run(runner=FakeRunner(fail={"transfer"}))
    assert res.outcome == "plan_failed" and res.failed_stage == "transfer"
    assert r.calls[-1] == "recover" and "transfer failed" in res.message


def test_cancel_aborts_without_recovery():
    res, r, _ = run(runner=FakeRunner(cancel={"lift"}))
    assert res.outcome == "aborted" and "recover" not in r.calls


def test_retract_failure_still_classifies_placement():
    res, r, _ = run(runner=FakeRunner(fail={"retract"}),
                    hooks=Hooks(observe_placement=lambda ctx: Placement("crate")))
    assert res.outcome == "success" and res.failed_stage == "retract" and r.calls[-1] == "recover"


def test_custom_classifier_hook():
    res, _, _ = run(hooks=Hooks(classify=lambda *a, **k: ("overflow", True)))
    assert res.outcome == "overflow"


def test_stage_index_matches_pickbag_feedback_constants():
    import pathlib
    import re
    from g2cell_tasks.state_machine import STAGE_INDEX
    action = pathlib.Path(__file__).parents[2] / "g2cell_interfaces" / "action" / "PickBag.action"
    consts = {m[0].lower(): int(m[1]) for m in re.findall(r"uint8 STAGE_(\w+)=(\d+)", action.read_text())}
    assert {s.value: i for s, i in STAGE_INDEX.items()} == consts


def test_outcome_codes_match_msg_constants():
    import pathlib
    import re
    from g2cell_tasks.outcome import OUTCOME_CODES
    msg = pathlib.Path(__file__).parents[2] / "g2cell_interfaces" / "msg" / "PickPlaceResult.msg"
    consts = {m[0].lower(): int(m[1]) for m in re.findall(r"uint8 OUTCOME_(\w+)=(\d+)", msg.read_text())}
    assert consts == OUTCOME_CODES
