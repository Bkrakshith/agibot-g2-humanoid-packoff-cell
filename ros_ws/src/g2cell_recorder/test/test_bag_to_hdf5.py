"""CLI + bag reader tests. The MCAP round trip uses the pure-python `mcap` reader (host venv)."""
import h5py
import numpy as np
import pytest

from g2cell_recorder import bag_to_hdf5
from g2cell_recorder.bag_reader import find_mcap_files, read_bag

JOINT_STATE_DEF = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _write_mcap_bag(bag_dir, n=10):
    pytest.importorskip("mcap_ros2")
    from mcap_ros2.writer import Writer

    bag_dir.mkdir()
    with open(bag_dir / "ep_0.mcap", "wb") as fh:
        w = Writer(fh)
        schema = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE_DEF)
        for i in range(n):
            t = 1_000_000_000 + i * 10_000_000
            w.write_message("/joint_states", schema, {
                "header": {"stamp": {"sec": 1, "nanosec": i * 10_000_000}, "frame_id": ""},
                "name": ["a", "b"], "position": [float(i), 2.0 * i], "velocity": [], "effort": []},
                log_time=t, publish_time=t)
        w.finish()


def test_find_mcap_files_accepts_dir_or_file(tmp_path):
    _write_mcap_bag(tmp_path / "bag")
    files = find_mcap_files(tmp_path / "bag")
    assert [p.name for p in files] == ["ep_0.mcap"]
    assert find_mcap_files(files[0]) == files
    with pytest.raises(FileNotFoundError):
        find_mcap_files(tmp_path / "missing")


def test_mcap_backend_yields_topic_time_msg(tmp_path):
    _write_mcap_bag(tmp_path / "bag", n=5)
    msgs = list(read_bag(tmp_path / "bag", topics=["/joint_states"], backend="mcap"))
    assert [m[0] for m in msgs] == ["/joint_states"] * 5
    assert msgs[1][1] == 1_010_000_000
    assert list(msgs[3][2].position) == [3.0, 6.0]
    assert list(read_bag(tmp_path / "bag", topics=["/other"], backend="mcap")) == []


def test_cli_converts_bag_to_hdf5(tmp_path):
    _write_mcap_bag(tmp_path / "bag")
    out = tmp_path / "ep.hdf5"
    rc = bag_to_hdf5.main([str(tmp_path / "bag"), "-o", str(out), "--scene", "s1", "--seed", "3",
                           "--rate", "50", "--backend", "mcap"])
    assert rc == 0
    with h5py.File(out, "r") as f:
        assert f.attrs["scene"] == "s1" and f.attrs["seed"] == 3
        assert f.attrs["success"] == "unknown"
        assert f["joint_states/positions"].shape == (10, 2)
        assert f["aligned/t_ns"].shape == (5,)  # 90 ms at 50 Hz
        np.testing.assert_array_equal(f["aligned/joint_states/positions"][:, 0], [0, 2, 4, 6, 8])


def test_cli_default_output_is_bag_path_plus_suffix(tmp_path):
    _write_mcap_bag(tmp_path / "bag")
    assert bag_to_hdf5.main([str(tmp_path / "bag"), "--backend", "mcap"]) == 0
    assert (tmp_path / "bag.hdf5").is_file()


def test_cli_success_flag(tmp_path):
    _write_mcap_bag(tmp_path / "bag")
    out = tmp_path / "e.hdf5"
    bag_to_hdf5.main([str(tmp_path / "bag"), "-o", str(out), "--success", "true", "--backend", "mcap"])
    with h5py.File(out, "r") as f:
        assert f.attrs["success"] == "true"


def test_cli_compression_and_workers_flags(tmp_path):
    _write_mcap_bag(tmp_path / "bag")
    out = tmp_path / "c.hdf5"
    assert bag_to_hdf5.main([str(tmp_path / "bag"), "-o", str(out), "--backend", "mcap",
                             "--compression", "lzf", "--workers", "2"]) == 0
    with h5py.File(out, "r") as f:
        assert f.attrs["image_compression"] == "lzf"


def test_cli_defaults_gzip4_and_bounded_workers():
    a = bag_to_hdf5.parse_args(["bag"])
    assert a.compression == "gzip" and a.level == 4
    assert 1 <= a.workers <= 8
