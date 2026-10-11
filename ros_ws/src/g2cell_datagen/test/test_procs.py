import time

import pytest

from g2cell_datagen.procs import ManagedProcess, run_shell


def test_start_stop_process_group(tmp_path):
    log = tmp_path / "p.log"
    p = ManagedProcess(["bash", "-c", "echo hi; sleep 30 & wait"], log_path=log, log=lambda *_: None).start()
    time.sleep(0.3)
    assert p.running
    t = time.monotonic()
    p.stop(timeout=2.0)
    assert not p.running and time.monotonic() - t < 5
    assert "hi" in log.read_text()


def test_stop_never_started_is_noop():
    assert ManagedProcess(["true"], log=lambda *_: None).stop() is None


def test_run_shell_rc_and_timeout(tmp_path):
    assert run_shell("exit 3", timeout=5) == 3
    assert run_shell("echo ok > " + str(tmp_path / "o"), timeout=5) == 0
    assert (tmp_path / "o").read_text().strip() == "ok"
    with pytest.raises(TimeoutError):
        run_shell("sleep 30", timeout=0.5)
