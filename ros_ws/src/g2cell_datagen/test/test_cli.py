import json

from g2cell_datagen import episode_runner as ER
from g2cell_datagen.randomize import sample
from g2cell_datagen.runner import RecordingResult, TaskResult


class Fake:
    def __init__(self):
        self.closed, self.calls = False, []

    def reset(self, name, spec):
        self.calls.append(name)
        return {}

    def start_recording(self, name, spec):
        pass

    def run_task(self, name, spec):
        return TaskResult("unverified")

    def stop_recording(self, name, success):
        return RecordingResult(True)

    def close(self):
        self.closed = True


def test_dry_run_prints_plan_and_writes_nothing(tmp_path, capsys):
    rc = ER.main(["--batch", "b", "--seeds", "0-2", "--episodes-root", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and len(out) == 3 and all("pending" in line for line in out)
    assert not (tmp_path / "b").exists()


def test_run_resume_and_batch_json(tmp_path, capsys):
    fake = Fake()
    argv = ["--batch", "b", "--n", "3", "--seed-start", "10", "--episodes-root", str(tmp_path)]
    assert ER.main(argv, backend_factory=lambda: fake) == 0
    assert fake.calls == ["ep_s00010", "ep_s00011", "ep_s00012"] and fake.closed
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == {"done": 3}
    bj = json.loads((tmp_path / "b" / "batch.json").read_text())
    assert bj["batch"] == "b" and bj["reset"] == "teleport" and len(bj["runs"]) == 1   # datagen.yaml default
    rec = json.loads((tmp_path / "b" / "ep_s00011" / "episode.json").read_text())
    _, _, cfg, _ = ER.load_configs(ER.build_parser().parse_args(argv))
    assert rec["randomisation"] == sample(cfg, 11).to_dict()
    # resume: nothing left; batch.json gains a run entry
    fake2 = Fake()
    assert ER.main(argv, backend_factory=lambda: fake2) == 0
    assert fake2.calls == []
    assert len(json.loads((tmp_path / "b" / "batch.json").read_text())["runs"]) == 2


def test_invalid_config_refuses(tmp_path, capsys):
    bad = tmp_path / "dg.yaml"
    src = (ER._share("g2cell_datagen", "config/datagen.yaml"))
    bad.write_text(open(src).read().replace("region_y: [0.52, 0.80]", "region_y: [0.52, 0.95]"))
    rc = ER.main(["--batch", "b", "--seeds", "0", "--episodes-root", str(tmp_path), "--config", str(bad)],
                 backend_factory=Fake)
    assert rc == 2 and "left_table_max_y" in capsys.readouterr().err


def test_operator_cli(tmp_path):
    fake = Fake()
    rc = ER.main(["--batch", "op", "--seeds", "0-1", "--episodes-root", str(tmp_path), "--mode", "operator"],
                 backend_factory=lambda: fake, operator=lambda n, s: "success")
    assert rc == 0
    rec = json.loads((tmp_path / "op" / "ep_s00000" / "episode.json").read_text())
    assert rec["mode"] == "operator" and rec["success"] == "true"


def test_console_operator_reprompts():
    answers = iter(["", "x", "F"])
    printed = []
    ask = ER.console_operator(input_fn=lambda _p: next(answers), print_fn=printed.append)
    _, _, cfg, _ = ER.load_configs(ER.build_parser().parse_args(["--batch", "b", "--n", "1"]))
    assert ask("ep_s00000", sample(cfg, 0)) == "failure"
    assert any("RECORDING" in p for p in printed) and sum("type s, f" in p for p in printed) == 2
