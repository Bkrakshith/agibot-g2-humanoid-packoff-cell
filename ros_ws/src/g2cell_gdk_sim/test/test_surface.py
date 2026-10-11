"""The public surface of agibot_gdk must match the 3.3.8 surface used by the training solutions."""
import json
import pathlib

import agibot_gdk
import surface

SNAPSHOT = pathlib.Path(__file__).with_name("fake_surface_3.3.8.json")


def test_snapshot_matches_the_cloned_training_fake(training_dir):
    live = surface.from_source((training_dir / "fake" / "agibot_gdk" / "__init__.py").read_text())
    assert live == json.loads(SNAPSHOT.read_text()), "training fake changed: refresh the snapshot and review"


def test_module_has_every_name_signature_and_enum_member_of_the_snapshot():
    assert surface.check_module(agibot_gdk, json.loads(SNAPSHOT.read_text())) == []


def test_module_has_every_name_of_the_live_training_fake(training_dir):
    live = surface.from_source((training_dir / "fake" / "agibot_gdk" / "__init__.py").read_text())
    assert surface.check_module(agibot_gdk, live) == []


def test_version_marks_the_sim_build():
    assert agibot_gdk.__version__ == "3.3.8-sim"
    assert "sim" in agibot_gdk.get_app_version()


def test_check_module_reports_a_changed_signature():
    s = json.loads(SNAPSHOT.read_text())
    s["classes"]["Robot"]["methods"]["move_head_joint"] = [{"name": "positions", "kind": "positional", "default": False}]
    assert any("move_head_joint" in p for p in surface.check_module(agibot_gdk, s))


def test_structs_construct_without_gdk_init():
    pose = agibot_gdk.EndEffectorPose()
    assert pose.left_end_effector_pose.orientation.w == 1.0
    req = agibot_gdk.JointControlReq()
    assert req.joint_names == [] and req.life_time == 0.0
    assert agibot_gdk.CameraIntrinsic([1.0, 2.0, 3.0, 4.0]).intrinsic == [1.0, 2.0, 3.0, 4.0]


def test_import_does_not_import_rclpy():
    import sys
    # importing agibot_gdk (done above) must not pull in ROS; only gdk_init() does
    assert "agibot_gdk._sim_backend" not in sys.modules
