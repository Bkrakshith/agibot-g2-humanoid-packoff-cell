"""record_episode.sh topic/flag selection, checked through --dry-run (no ROS needed)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from g2cell_recorder.episode import DEFAULT_TOPICS
from g2cell_recorder.throttle_relay import RateGate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_episode.sh"
HEAD = "/genie_sim/head_front_camera_rgb/image_raw"
FREE = "/genie_sim/free_camera_rgb/image_raw"
DEPTH = "/genie_sim/head_front_camera_depth/image_raw"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def dry(tmp_path, *args, env=None):
    """Run the script with --dry-run; return {key: value} of its KEY=value lines."""
    r = subprocess.run(["bash", str(SCRIPT), "ep1", "--dry-run", *args], capture_output=True,
                       text=True, env={"EPISODES_DIR": str(tmp_path), "PATH": "/usr/bin:/bin",
                                       **(env or {})})
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    out["topics"] = out.get("TOPICS", "").split()
    return out


def test_defaults_are_the_cheap_profile(tmp_path):
    o = dry(tmp_path)
    t = o["topics"]
    assert {"/joint_states", "/joint_command", "/tf", "/tf_static"} <= set(t)
    assert HEAD + "/compressed" in t and HEAD not in t
    assert not any(x.startswith(FREE) for x in t)                     # free camera off
    assert "/g2cell/rec/head_front_camera_depth/compressedDepth" in t  # throttled depth
    assert not any(x.startswith(DEPTH) for x in t)
    assert o["RELAY"] == f"{DEPTH}/compressedDepth -> " \
        "/g2cell/rec/head_front_camera_depth/compressedDepth @ 5 Hz (sensor_msgs/msg/CompressedImage)"
    assert "--storage-preset-profile zstd_fast" in o["CMD"]
    assert "--storage mcap" in o["CMD"]
    assert not (tmp_path / "ep1").exists()       # dry run creates nothing


def test_every_recorded_image_topic_is_converted(tmp_path):
    for args in ([], ["--free-cam"], ["--full-fidelity"], ["--rgb", "zstd", "--depth", "zstd"],
                 ["--depth", "raw"], ["--depth-rate", "full"]):
        for topic in dry(tmp_path, *args)["topics"]:
            if "camera" in topic:
                assert topic in DEFAULT_TOPICS, (args, topic)


def test_free_cam_flag(tmp_path):
    assert FREE + "/compressed" in dry(tmp_path, "--free-cam")["topics"]
    assert FREE + "/zstd" in dry(tmp_path, "--free-cam", "--rgb", "zstd")["topics"]


def test_full_fidelity_restores_raw_streams(tmp_path):
    o = dry(tmp_path, "--full-fidelity")
    assert {HEAD, FREE, DEPTH} <= set(o["topics"])
    assert o["RELAY"] == "none"
    assert "--storage-preset-profile zstd_fast" in o["CMD"]   # lossless, still on


def test_depth_rate_and_off(tmp_path):
    o = dry(tmp_path, "--depth-rate", "full")
    assert DEPTH + "/compressedDepth" in o["topics"] and o["RELAY"] == "none"
    o = dry(tmp_path, "--depth-rate", "2.5")
    assert "@ 2.5 Hz" in o["RELAY"]
    o = dry(tmp_path, "--depth", "raw")
    assert "/g2cell/rec/head_front_camera_depth/image_raw" in o["topics"]
    assert "(sensor_msgs/msg/Image)" in o["RELAY"]
    o = dry(tmp_path, "--depth", "off")
    assert not any("depth" in x for x in o["topics"]) and o["RELAY"] == "none"


def test_bag_compression_flag(tmp_path):
    assert "--storage-preset-profile zstd_small" in dry(tmp_path, "--compression",
                                                        "zstd_small")["CMD"]
    assert "storage-preset-profile" not in dry(tmp_path, "--compression", "none")["CMD"]


def test_extra_topics_env_and_bad_flags(tmp_path):
    assert "/odom" in dry(tmp_path, env={"EXTRA_TOPICS": "/odom"})["topics"]
    for bad in (["--rgb", "theora"], ["--depth", "jpeg"], ["--compression", "lz4"],
                ["--depth-rate", "fast"]):
        r = subprocess.run(["bash", str(SCRIPT), "ep1", "--dry-run", *bad], capture_output=True,
                           text=True, env={"EPISODES_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"})
        assert r.returncode != 0, bad


# ---------------------------------------------------------------- RateGate (throttle_relay)

def test_rate_gate_limits_to_rate_with_jitter_tolerance():
    g = RateGate(5.0)                      # 200 ms period
    ms = 1_000_000
    stamps = [0, 74, 148, 222, 296, 370, 444, 518, 592, 666]   # 13.5 Hz source
    kept = [t for t in stamps if g.accept(t * ms)]
    assert kept == [0, 222, 444, 666]
    g = RateGate(5.0)
    assert g.accept(0) and g.accept(190 * ms)   # 10 % jitter tolerance
    assert not g.accept(300 * ms)


def test_rate_gate_rejects_bad_rate():
    with pytest.raises(ValueError):
        RateGate(0)


def test_relay_rate_param_is_a_double(tmp_path):
    # throttle_relay declares rate_hz as a double; ROS 2 rejects an integer override ("rate_hz:=5"),
    # the relay exits at start and the episode has no depth (seen in the first live datagen run)
    o = dry(tmp_path)
    assert "-p rate_hz:=5.0 " in o["RELAY_CMD"] + " "
    assert "-p rate_hz:=2.5 " in dry(tmp_path, "--depth-rate", "2.5")["RELAY_CMD"] + " "
    assert "-p rate_hz:=10.0 " in dry(tmp_path, "--depth-rate", "10")["RELAY_CMD"] + " "
