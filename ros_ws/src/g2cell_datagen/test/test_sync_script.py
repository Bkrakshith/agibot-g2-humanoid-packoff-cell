"""scripts/sync_to_server.sh --push-dataset, run against fake ssh/rsync on PATH (no network)."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "sync_to_server.sh"


@pytest.fixture
def env(tmp_path):
    proj = tmp_path / "g2_packoff_cell"
    (proj / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, proj / "scripts" / "sync_to_server.sh")
    ds = proj / "output" / "episodes" / "pick_v1" / "ep_s00000"
    ds.mkdir(parents=True)
    (ds / "ep_s00000.hdf5").write_text("h5")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "calls.log"
    for tool in ("ssh", "rsync"):
        p = bin_ / tool
        p.write_text(f'#!/usr/bin/env bash\necho "{tool} $*" >> {log}\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    e = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    return proj, e, log


def run(proj, e, *args):
    return subprocess.run(["bash", str(proj / "scripts" / "sync_to_server.sh"), *args], env=e,
                          capture_output=True, text=True)


def test_push_dataset(env):
    proj, e, log = env
    r = run(proj, e, "--push-dataset", "pick_v1", "--host", "box")
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == "ssh box mkdir -p g2/g2_packoff_cell/output/episodes/pick_v1"
    rs = calls[1]
    assert rs.startswith("rsync ") and f"{proj}/output/episodes/pick_v1/ box:g2/g2_packoff_cell/output/episodes/pick_v1/" in rs
    assert "--exclude bag/" in rs and "--exclude _crashed/" in rs and "--exclude *.mcap" in rs
    assert len(calls) == 2           # nothing else (no project / assets sync)


def test_push_dataset_with_bags_and_dry_run(env):
    proj, e, log = env
    r = run(proj, e, "--push-dataset", "pick_v1", "--with-bags", "--dry-run")
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith("rsync ") and "--dry-run" in calls[0]
    assert "--exclude bag/" not in calls[0]


@pytest.mark.parametrize("name,code", [("nope", 1), ("../x", 2), ("a/b", 2)])
def test_push_dataset_rejects(env, name, code):
    proj, e, log = env
    assert run(proj, e, "--push-dataset", name).returncode == code
    assert not log.exists()
