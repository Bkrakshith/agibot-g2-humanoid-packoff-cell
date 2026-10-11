import json
import sys

import h5py
import pytest

from g2cell_datagen import batch as B
from g2cell_datagen.store import EpisodeStore


def _h5(path, seed, success="unknown", duration=10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update({"scene": "scene_g2cell", "seed": seed, "success": success, "duration_s": duration,
                        "aligned_rate_hz": 30.0, "schema_version": 1})
    return path


def _rec(seed, status="done", outcome="unverified", success="unknown", label="good", total=40.0):
    return {"seed": seed, "status": status, "outcome": outcome, "success": success, "mode": "scripted",
            "randomisation": {"seed": seed, "bag": {"id": "Bag_01", "x": 0.0, "y": 0.6, "z": 0.87, "yaw_deg": 10.0,
                                                    "label": label},
                              "target": "crate" if label == "good" else "rack_lane",
                              "lane": "" if label == "good" else "L1_1", "parked": {}, "tries": 1},
            "timings": {"total_s": total, "task_s": total - 10}}


@pytest.fixture
def batch(tmp_path):
    st = EpisodeStore(tmp_path / "b1")
    st.write("ep_s00000", _rec(0, outcome="success", success="true"))
    _h5(st.hdf5_path("ep_s00000"), 0, "true", 12.0)
    st.write("ep_s00001", _rec(1, outcome="plan_failed", success="false", label="reject"))
    _h5(st.hdf5_path("ep_s00001"), 1, "false", 8.0)
    st.write("ep_s00002", _rec(2, status="error"))
    return st


def test_card_merges_hdf5_card_and_datagen(batch, capsys):
    assert B.main(["card", str(batch.dir), "--name", "pick_v1"]) == 0
    card = json.loads((batch.dir / "dataset_card.json").read_text())
    assert card["batch_name"] == "pick_v1" and card["num_episodes"] == 2
    assert card["success"]["rate"] == 0.5
    dg = card["datagen"]
    assert dg["status"] == {"done": 2, "error": 1}
    assert dg["outcome"]["success"] == 1 and dg["label"] == {"good": 1, "reject": 1}
    assert dg["target"] == {"crate": 1, "rack_lane": 1}
    assert dg["randomisation_ranges"]["x"] == [0.0, 0.0]
    assert dg["mean_total_s"] == pytest.approx(40.0)
    md = (batch.dir / "dataset_card.md").read_text()
    assert "Dataset card: pick_v1" in md and "## Scripted generation" in md and "plan_failed" in md


def test_card_without_hdf5(tmp_path):
    st = EpisodeStore(tmp_path / "b2")
    st.write("ep_s00000", _rec(0))
    assert B.main(["card", str(st.dir)]) == 0
    card = json.loads((st.dir / "dataset_card.json").read_text())
    assert card["num_episodes"] == 0 and card["datagen"]["episodes"] == 1


def test_status(batch, capsys):
    assert B.main(["status", str(batch.dir)]) == 0
    out = capsys.readouterr().out
    assert "ep_s00002" in out and "error" in out and '"done": 2' in out


def test_finalize_converts_and_marks(tmp_path):
    st = EpisodeStore(tmp_path / "b3")
    # done, bag only (container had no h5py) -> convert with the JSON's success
    st.write("ep_s00004", _rec(4, outcome="success", success="true"))
    (st.episode_dir("ep_s00004") / "bag").mkdir()
    (st.episode_dir("ep_s00004") / "bag" / "metadata.yaml").write_text("x")
    # done, hdf5 exists but unmarked -> mark only
    st.write("ep_s00005", _rec(5, outcome="drop", success="false"))
    _h5(st.hdf5_path("ep_s00005"), 5, "unknown")
    # error episodes are left alone
    st.write("ep_s00006", _rec(6, status="error"))
    (st.episode_dir("ep_s00006") / "bag").mkdir()
    fake_conv = tmp_path / "conv.py"
    fake_conv.write_text(
        "import sys, h5py\n"
        "a = sys.argv[1:]\n"
        "out = a[a.index('-o') + 1]\n"
        "with h5py.File(out, 'w') as f:\n"
        "    f.attrs['success'] = a[a.index('--success') + 1]\n"
        "    f.attrs['seed'] = int(a[a.index('--seed') + 1])\n")
    rc = B.main(["finalize", str(st.dir), "--converter", f"{sys.executable} {fake_conv}"])
    assert rc == 0
    with h5py.File(st.hdf5_path("ep_s00004")) as f:
        assert f.attrs["success"] == "true" and f.attrs["seed"] == 4
    with h5py.File(st.hdf5_path("ep_s00005")) as f:
        assert f.attrs["success"] == "false"
    assert not st.hdf5_path("ep_s00006").exists()
    assert st.load("ep_s00004")["recording"]["hdf5"].endswith("ep_s00004.hdf5")


def test_finalize_dry_run(tmp_path, capsys):
    st = EpisodeStore(tmp_path / "b4")
    st.write("ep_s00001", _rec(1, success="true"))
    (st.episode_dir("ep_s00001") / "bag").mkdir()
    assert B.main(["finalize", str(st.dir), "--dry-run"]) == 0
    assert "convert" in capsys.readouterr().out and not st.hdf5_path("ep_s00001").exists()
