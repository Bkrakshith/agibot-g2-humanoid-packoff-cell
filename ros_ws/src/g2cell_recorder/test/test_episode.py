"""Unit tests for the ROS-free episode logic. Run on the host:
    ../.venv-tools/bin/python -m pytest ros_ws/src/g2cell_recorder/test
"""
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from g2cell_recorder.episode import (
    DEFAULT_TOPICS,
    aligned_grid,
    convert_messages,
    depth_to_float32,
    image_to_rgb,
    joint_matrix,
    previous_index,
)

SEC = 1_000_000_000


# ---------- image conversion ----------

def test_rgba8_drops_alpha_and_keeps_layout():
    h, w = 2, 3
    rgba = np.arange(h * w * 4, dtype=np.uint8).reshape(h, w, 4)
    out = image_to_rgb(rgba.tobytes(), h, w, "rgba8", step=w * 4)
    assert out.dtype == np.uint8 and out.shape == (h, w, 3)
    np.testing.assert_array_equal(out, rgba[..., :3])


def test_bgra8_and_bgr8_are_swapped_to_rgb():
    px = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)  # B G R A
    assert image_to_rgb(px.tobytes(), 1, 1, "bgra8", 4).tolist() == [[[30, 20, 10]]]
    assert image_to_rgb(px[..., :3].tobytes(), 1, 1, "bgr8", 3).tolist() == [[[30, 20, 10]]]


def test_row_padding_in_step_is_ignored():
    h, w = 2, 2
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    padded = np.zeros((h, w * 3 + 2), dtype=np.uint8)
    padded[:, : w * 3] = rgb.reshape(h, -1)
    out = image_to_rgb(padded.tobytes(), h, w, "rgb8", step=w * 3 + 2)
    np.testing.assert_array_equal(out, rgb)


def test_mono8_is_replicated_to_three_channels():
    out = image_to_rgb(bytes([7, 9]), 1, 2, "mono8", 2)
    assert out.tolist() == [[[7, 7, 7], [9, 9, 9]]]


def test_unknown_encoding_raises():
    with pytest.raises(ValueError):
        image_to_rgb(b"\0" * 4, 1, 1, "yuv422", 4)


def test_depth_32fc1_and_16uc1():
    d = np.array([[0.5, 1.25]], dtype=np.float32)
    out = depth_to_float32(d.tobytes(), 1, 2, "32FC1", 8)
    assert out.dtype == np.float32 and out.shape == (1, 2)
    np.testing.assert_array_equal(out, d)
    mm = np.array([[500, 1250]], dtype=np.uint16)
    np.testing.assert_allclose(depth_to_float32(mm.tobytes(), 1, 2, "16UC1", 4), d)


# ---------- joint tables ----------

def test_joint_matrix_unions_names_and_fills_missing_with_nan():
    names, mat = joint_matrix([["a", "b"], ["b", "c"]], [[1.0, 2.0], [3.0, 4.0]])
    assert names == ["a", "b", "c"]
    assert mat.dtype == np.float64 and mat.shape == (2, 3)
    np.testing.assert_array_equal(mat[0, :2], [1.0, 2.0])
    assert np.isnan(mat[0, 2]) and np.isnan(mat[1, 0])
    np.testing.assert_array_equal(mat[1, 1:], [3.0, 4.0])


def test_joint_matrix_handles_empty_value_lists():
    names, mat = joint_matrix([["a", "b"]], [[]])
    assert names == ["a", "b"] and np.isnan(mat).all()


# ---------- alignment ----------

def test_aligned_grid_spans_common_window_at_rate():
    times = {"x": np.array([0, 1 * SEC, 2 * SEC]), "y": np.array([SEC // 2, 3 * SEC])}
    grid = aligned_grid(times, rate_hz=10.0)
    assert grid[0] == SEC // 2           # latest first sample
    assert grid[-1] <= 2 * SEC           # earliest last sample
    assert np.all(np.diff(grid) == SEC // 10)
    assert len(grid) == 16


def test_aligned_grid_ignores_empty_streams_and_rejects_bad_rate():
    times = {"x": np.array([0, SEC]), "empty": np.array([], dtype=np.int64)}
    assert len(aligned_grid(times, 30.0)) == 31
    with pytest.raises(ValueError):
        aligned_grid(times, 0.0)


def test_previous_index_picks_latest_sample_at_or_before():
    t = np.array([10, 20, 30])
    grid = np.array([10, 15, 20, 29, 30, 100])
    np.testing.assert_array_equal(previous_index(t, grid), [0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError):
        previous_index(t, np.array([5]))


# ---------- end-to-end conversion with fake messages ----------

def _stamp(ns):
    return NS(sec=ns // SEC, nanosec=ns % SEC)


def _img(ns, h=4, w=5, value=0):
    data = np.full((h, w, 4), value, dtype=np.uint8)
    data[..., 3] = 255
    return NS(header=NS(stamp=_stamp(ns)), height=h, width=w, encoding="rgba8",
              step=w * 4, data=data.tobytes())


def _depth(ns, h=4, w=5):
    data = np.full((h, w), 1.5, dtype=np.float32)
    return NS(header=NS(stamp=_stamp(ns)), height=h, width=w, encoding="32FC1",
              step=w * 4, data=data.tobytes())


def _js(ns, names, pos):
    return NS(header=NS(stamp=_stamp(ns)), name=list(names), position=list(pos),
              velocity=[0.1] * len(pos), effort=[])


def _fake_bag():
    msgs = []
    for i in range(21):  # joint_states 100 Hz -> 0..0.2 s
        t = i * SEC // 100
        msgs.append(("/joint_states", t, _js(t, ["j1", "j2"], [i, -i])))
    for i in range(3):  # head camera 10 Hz
        t = i * SEC // 10
        msgs.append(("/genie_sim/head_front_camera_rgb/image_raw", t, _img(t, value=i + 1)))
        msgs.append(("/genie_sim/head_front_camera_depth/image_raw", t, _depth(t)))
    for i in range(11):  # joint_command 50 Hz, arm subsets alternate
        t = i * SEC // 50
        names = ["j1"] if i % 2 == 0 else ["j2"]
        msgs.append(("/joint_command", t, _js(t, names, [100 + i])))
    msgs.append(("/tf", 0, NS()))  # not converted, must not break
    msgs.sort(key=lambda m: m[1])
    return msgs


def test_convert_writes_expected_schema(tmp_path):
    out = tmp_path / "ep.hdf5"
    convert_messages(iter(_fake_bag()), out, source="/bags/ep", scene="scene_pnp_g2_op",
                     seed=7, aligned_rate_hz=30.0)
    with h5py.File(out, "r") as f:
        a = f.attrs
        assert a["source_bag"] == "/bags/ep"
        assert a["scene"] == "scene_pnp_g2_op" and a["seed"] == 7
        assert a["success"] == "unknown"
        assert a["start_time_ns"] == 0 and a["end_time_ns"] == 2 * SEC // 10
        assert a["aligned_rate_hz"] == 30.0

        rgb = f["images/head_front_rgb"]
        assert rgb["data"].shape == (3, 4, 5, 3) and rgb["data"].dtype == np.uint8
        assert rgb["data"].compression == "gzip"
        assert rgb["data"][2, 0, 0].tolist() == [3, 3, 3]
        assert rgb["t_ns"].shape == (3,)
        assert rgb.attrs["rate_hz"] == pytest.approx(10.0)
        assert "free_rgb" not in f["images"]  # topic absent -> no group

        depth = f["depth/head_front"]["data"]
        assert depth.shape == (3, 4, 5) and depth.dtype == np.float32

        js = f["joint_states"]
        assert js["positions"].shape == (21, 2)
        assert js["velocities"].shape == (21, 2) and js["efforts"].shape == (21, 2)
        assert list(js.attrs["names"]) == ["j1", "j2"]
        assert np.all(np.diff(js["t_ns"][:]) > 0)
        assert js["header_t_ns"][5] == 5 * SEC // 100

        jc = f["joint_command"]
        assert jc["actions"].shape == (11, 2)
        assert list(jc.attrs["names"]) == ["j1", "j2"]

        al = f["aligned"]
        t = al["t_ns"][:]
        assert al.attrs["rate_hz"] == 30.0
        assert t[0] == 0 and t[-1] <= 2 * SEC // 10 and len(t) == 7
        assert al["joint_states/positions"].shape == (7, 2)
        # at t = 1/30 s the previous joint_states sample is index 3 (0.03 s)
        assert al["joint_states/positions"][1].tolist() == [3.0, -3.0]
        assert al["images/head_front_rgb/index"][:].tolist() == [0, 0, 0, 1, 1, 1, 2]
        assert al["joint_command/actions"].shape == (7, 2)
        # aligned actions are forward-filled per joint, so no NaN once each joint was commanded
        assert not np.isnan(al["joint_command/actions"][2:]).any()


def test_convert_can_copy_aligned_images(tmp_path):
    out = tmp_path / "ep.hdf5"
    convert_messages(iter(_fake_bag()), out, source="b", aligned_rate_hz=10.0,
                     aligned_images=True)
    with h5py.File(out, "r") as f:
        imgs = f["aligned/images/head_front_rgb/data"]
        assert imgs.shape == (3, 4, 5, 3)
        assert imgs[2, 0, 0].tolist() == [3, 3, 3]


def test_convert_rejects_bag_without_known_topics(tmp_path):
    with pytest.raises(ValueError):
        convert_messages(iter([("/tf", 0, NS())]), tmp_path / "x.hdf5", source="b")


def test_default_topics_cover_required_streams():
    assert set(DEFAULT_TOPICS) >= {
        "/joint_states", "/joint_command",
        "/genie_sim/head_front_camera_rgb/image_raw",
        "/genie_sim/head_front_camera_depth/image_raw",
        "/genie_sim/free_camera_rgb/image_raw",
    }
