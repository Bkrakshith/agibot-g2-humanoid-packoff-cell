import math
from pathlib import Path

import pytest
import yaml

from g2cell_teleop.devices import SpaceMouseConfig
from g2cell_teleop.motion import MotionLimits
from g2cell_teleop.params import DEFAULTS, home_from, limits_from, spacemouse_config_from, workspace_from
from g2cell_teleop.teleop_core import DEFAULT_HOME_DEG

PKG = Path(__file__).resolve().parents[1]
YAML = PKG / "config" / "teleop_g2.yaml"


@pytest.fixture(scope="module")
def cfg():
    doc = yaml.safe_load(YAML.read_text())
    return doc["/**"]["ros__parameters"]


def _kind(v):
    if isinstance(v, bool):
        return bool
    if isinstance(v, list):
        return (list, _kind(v[0]))
    return type(v)


def test_yaml_has_exactly_the_declared_params(cfg):
    assert set(cfg) == set(DEFAULTS)


def test_yaml_types_match_declared_defaults(cfg):
    # rclpy rejects a yaml override whose type differs from the declared default (e.g. 0 vs 0.0)
    for k, v in cfg.items():
        assert _kind(v) == _kind(DEFAULTS[k]), k


def test_yaml_values_match_defaults(cfg):
    assert cfg == DEFAULTS


def test_spec_values(cfg):
    assert cfg["max_lin_speed"] == 0.1 and cfg["max_ang_speed"] == 0.5
    assert cfg["home_left_deg"] == cfg["home_right_deg"] == DEFAULT_HOME_DEG
    assert cfg["left_gripper_joint"] == "idx31_gripper_l_inner_joint1"
    assert cfg["right_gripper_joint"] == "idx71_gripper_r_inner_joint1"
    assert cfg["gripper_lower"] == -0.785 and cfg["gripper_upper"] == 0.0
    assert cfg["device"] == "keyboard"


def test_workspace_boxes_are_well_formed(cfg):
    for arm in ("left", "right"):
        lo, hi = cfg[f"workspace_{arm}_min"], cfg[f"workspace_{arm}_max"]
        assert len(lo) == len(hi) == 3
        assert all(a < b for a, b in zip(lo, hi))


def test_builders_turn_params_into_objects():
    p = dict(DEFAULTS, max_lin_speed=0.05, spacemouse_axis_sign=[1, -1, 1, 1, 1, -1])
    assert limits_from(p) == MotionLimits(max_lin=0.05, max_ang=0.5, max_lin_acc=p["max_lin_acc"],
                                          max_ang_acc=p["max_ang_acc"], lin_deadband=p["lin_deadband"],
                                          ang_deadband=p["ang_deadband"])
    sm = spacemouse_config_from(p)
    assert isinstance(sm, SpaceMouseConfig) and sm.axis_sign == (1, -1, 1, 1, 1, -1)
    ws = workspace_from(p)
    assert ws["left"] == (tuple(p["workspace_left_min"]), tuple(p["workspace_left_max"]))
    home = home_from(p)
    assert home["right"]["idx64_arm_r_joint4"] == pytest.approx(math.radians(-75))


def test_workspace_can_be_disabled():
    assert workspace_from(dict(DEFAULTS, use_workspace=False)) is None


def test_setup_installs_launch_config_and_readme():
    setup = (PKG / "setup.py").read_text()
    assert 'glob("launch/*.launch.py")' in setup and 'glob("config/*.yaml")' in setup
    assert (PKG / "launch" / "teleop.launch.py").is_file()
    assert (PKG / "README.md").is_file()
    assert "keyboard_teleop = g2cell_teleop.keyboard_teleop_node:main" in setup  # name unchanged
