"""Shared fixtures: tiny synthetic episode HDF5 files written through the real recorder
(g2cell_recorder.episode.convert_messages) so the schema can never drift from the converter.

Run (CPU only):
    CUDA_VISIBLE_DEVICES="" ../../../../.venv-learn/bin/python -m pytest ros_ws/src/g2cell_learning/test
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SRC = PKG.parent
for p in (PKG, SRC / "g2cell_recorder"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEC = 1_000_000_000

# joint_states publishes more joints, in a different order, than joint_command commands
STATE_NAMES = ["idx01_body", "right_j1", "left_j1", "left_j2", "head_j1",
               "left_gripper", "right_gripper", "right_j2"]
CMD_NAMES = ["left_j1", "left_j2", "left_gripper", "right_j1", "right_j2", "right_gripper"]

HEAD_HW = (24, 32)
FREE_HW = (30, 40)


def _stamp(t_ns):
    return NS(stamp=NS(sec=t_ns // SEC, nanosec=t_ns % SEC))


def cmd_value(name: str, t: float) -> float:
    """Ground-truth commanded trajectory: smooth and joint-specific."""
    k = CMD_NAMES.index(name)
    return 0.1 * (k + 1) * np.sin(t + k)


def _rgb_msg(t_ns, hw, shade):
    h, w = hw
    img = np.full((h, w, 3), shade % 256, dtype=np.uint8)
    img[: h // 2, :, 0] = 255 - shade % 256
    return NS(header=_stamp(t_ns), height=h, width=w, encoding="rgb8", step=w * 3,
              data=img.tobytes())


def _depth_msg(t_ns, hw):
    h, w = hw
    d = np.full((h, w), 1.5, dtype=np.float32)
    return NS(header=_stamp(t_ns), height=h, width=w, encoding="32FC1", step=w * 4,
              data=d.tobytes())


def make_messages(duration_s: float = 2.0, with_commands: bool = True, phase: float = 0.0):
    """(topic, t_ns, msg) in time order: joint_states 100 Hz, commands 20 Hz, cams 30/15 Hz."""
    msgs = []
    t0 = 1 * SEC
    end = t0 + int(duration_s * SEC)
    # joint_states: track commands with a small lag; joints not commanded hold still
    for t_ns in range(t0, end, SEC // 100):
        t = (t_ns - t0) / SEC + phase
        pos = []
        for n in STATE_NAMES:
            pos.append(cmd_value(n, t - 0.02) if n in CMD_NAMES else 0.3)
        msgs.append(("/joint_states", t_ns, NS(header=_stamp(t_ns), name=list(STATE_NAMES),
                                                position=pos, velocity=[0.0] * len(pos),
                                                effort=[0.0] * len(pos))))
    if with_commands:
        for i, t_ns in enumerate(range(t0 + SEC // 200, end, SEC // 20)):
            t = (t_ns - t0) / SEC + phase
            # alternate arms the way the keyboard teleop does: left arm msgs and right arm msgs
            names = CMD_NAMES[:3] if i % 2 == 0 else CMD_NAMES[3:]
            if i == 0:
                names = CMD_NAMES[:3]  # right arm uncommanded at first -> leading NaN
            msgs.append(("/joint_command", t_ns, NS(header=_stamp(t_ns), name=list(names),
                                                     position=[cmd_value(n, t) for n in names],
                                                     velocity=[], effort=[])))
    for i, t_ns in enumerate(range(t0 + SEC // 300, end, SEC // 30)):
        msgs.append(("/genie_sim/head_front_camera_rgb/image_raw", t_ns,
                     _rgb_msg(t_ns, HEAD_HW, i * 4)))
        msgs.append(("/genie_sim/head_front_camera_depth/image_raw", t_ns,
                     _depth_msg(t_ns, HEAD_HW)))
    for i, t_ns in enumerate(range(t0 + SEC // 250, end, SEC // 15)):
        msgs.append(("/genie_sim/free_camera_rgb/image_raw", t_ns, _rgb_msg(t_ns, FREE_HW, i * 8)))
    msgs.sort(key=lambda m: m[1])
    return msgs


def write_episode(path: Path, duration_s=2.0, with_commands=True, phase=0.0, scene="scene_a",
                  seed=0, success="unknown"):
    from g2cell_recorder.episode import DEFAULT_TOPICS, convert_messages
    topics = dict(DEFAULT_TOPICS)
    if not with_commands:
        topics.pop("/joint_command")
    convert_messages(make_messages(duration_s, with_commands, phase), path, source="synthetic",
                     scene=scene, seed=seed, success=success, topics=topics)
    return path


@pytest.fixture
def episode_path(tmp_path) -> Path:
    return write_episode(tmp_path / "ep000.hdf5")


@pytest.fixture
def episode_paths(tmp_path) -> list[Path]:
    return [
        write_episode(tmp_path / "ep000.hdf5", 2.0, phase=0.0, scene="scene_a", seed=0,
                      success="true"),
        write_episode(tmp_path / "ep001.hdf5", 1.5, phase=0.7, scene="scene_a", seed=1,
                      success="false"),
        write_episode(tmp_path / "ep002.hdf5", 2.5, phase=1.3, scene="scene_b", seed=2,
                      success="true"),
    ]


@pytest.fixture
def no_action_episode(tmp_path) -> Path:
    return write_episode(tmp_path / "noact.hdf5", with_commands=False)
