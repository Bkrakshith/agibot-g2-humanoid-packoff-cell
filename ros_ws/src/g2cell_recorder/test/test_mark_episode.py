import h5py
import pytest

from g2cell_recorder.mark_episode import main, mark


@pytest.fixture
def h5(tmp_path):
    p = tmp_path / "ep.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["success"] = "unknown"
        f.attrs["scene"] = "s"
        f.create_dataset("x", data=[1, 2, 3])
    return p


def test_mark_sets_root_success_attr_and_keeps_the_rest(h5):
    mark(h5, "true")
    with h5py.File(h5, "r") as f:
        assert f.attrs["success"] == "true"
        assert f.attrs["scene"] == "s"
        assert list(f["x"][:]) == [1, 2, 3]


def test_mark_rejects_other_values(h5):
    with pytest.raises(ValueError):
        mark(h5, "yes")


def test_cli_marks_failure(h5):
    assert main([str(h5), "--success", "false"]) == 0
    with h5py.File(h5, "r") as f:
        assert f.attrs["success"] == "false"


def test_cli_missing_file_returns_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.hdf5"), "--success", "true"]) == 1
    assert "nope.hdf5" in capsys.readouterr().err
