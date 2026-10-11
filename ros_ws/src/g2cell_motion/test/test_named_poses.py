import math

import pytest
import yaml

from g2cell_motion.named_poses import NamedTarget, load_named_poses, resolve

DOC = """
named_poses:
  home:
    joints_deg: [0, -45, 0, -75, 0, -55, 0]
  above_crate:
    tool: {xyz: [0.60, 0.0, 1.194], yaw_deg: 0, tilt_deg: 0}
  above_table:
    left: {tool: {xyz: [0.0, 0.60, 1.10]}}
"""


def poses():
    return load_named_poses(yaml.safe_load(DOC))


def test_joint_pose_for_both_arms():
    t = resolve(poses(), "home", "right")
    assert t.kind == "joints" and math.isclose(t.joints_deg[1], -45)


def test_tool_pose_with_orientation():
    t = resolve(poses(), "above_crate", "left")
    assert t == NamedTarget("tool", xyz=(0.60, 0.0, 1.194), yaw_deg=0.0, tilt_deg=0.0)


def test_per_arm_pose_missing_for_other_arm():
    t = resolve(poses(), "above_table", "left")
    assert t.kind == "tool" and t.yaw_deg is None  # orientation searched
    with pytest.raises(KeyError):
        resolve(poses(), "above_table", "right")


def test_unknown_name():
    with pytest.raises(KeyError):
        resolve(poses(), "nope", "left")


@pytest.mark.parametrize("bad", [
    {"named_poses": {"x": {}}},
    {"named_poses": {"x": {"joints_deg": [0, 1]}}},
    {"named_poses": {"x": {"tool": {"xyz": [0, 1]}}}},
    {"named_poses": {"x": {"joints_deg": [0] * 7, "tool": {"xyz": [0, 0, 1]}}}},
    {"nothing": 1},
    {"named_poses": {"x": {"tool": {"xyz": [0, 0, 1], "tilt_deg": 20}}}},
])
def test_invalid(bad):
    with pytest.raises(ValueError):
        load_named_poses(bad)


def test_shipped_config_is_valid():
    import pathlib
    cfg = pathlib.Path(__file__).parents[1] / "config" / "named_poses.yaml"
    p = load_named_poses(yaml.safe_load(cfg.read_text()))
    for name in ("home", "above_table", "above_crate"):
        assert name in p
    assert resolve(p, "home", "left").kind == "joints"


def test_waist_deg_on_joint_pose():
    p = load_named_poses({"named_poses": {"h": {"joints_deg": [0] * 7, "waist_deg": 0},
                                          "k": {"joints_deg": [0] * 7}}})
    assert resolve(p, "h", "left").waist_deg == 0.0
    assert resolve(p, "k", "left").waist_deg is None


def test_shipped_home_turns_waist_back():
    import pathlib
    import yaml
    doc = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config" / "named_poses.yaml").read_text())
    assert resolve(load_named_poses(doc), "home", "left").waist_deg == 0.0
