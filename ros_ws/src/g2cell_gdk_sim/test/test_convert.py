"""Conversions between the sim's ROS data and the GDK return shapes."""
import math

import numpy as np
import pytest
import yaml

import agibot_gdk
from agibot_gdk import _convert as cv
from agibot_gdk import _geometry as g
from agibot_gdk import _limits as lim
from agibot_gdk._convert import JointSample, RawImage
from conftest import SCENE

# /joint_states name order as the engine publishes it (from a recorded bag): arms interleaved,
# gripper joints and wheels mixed in.
SIM_ORDER = ["idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3", "idx04_body_joint4", "idx05_body_joint5",
             "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3", "idx21_arm_l_joint1", "idx61_arm_r_joint1",
             "idx22_arm_l_joint2", "idx23_arm_l_joint3", "idx24_arm_l_joint4", "idx25_arm_l_joint5",
             "idx26_arm_l_joint6", "idx27_arm_l_joint7", "idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1",
             "idx62_arm_r_joint2", "idx63_arm_r_joint3", "idx64_arm_r_joint4", "idx65_arm_r_joint5",
             "idx66_arm_r_joint6", "idx67_arm_r_joint7", "idx71_gripper_r_inner_joint1",
             "idx111_chassis_lwheel_front_joint1"]


def sample(effort=True, velocity=True):
    pos = [0.01 * i for i in range(len(SIM_ORDER))]
    return JointSample(names=list(SIM_ORDER), position=pos,
                       velocity=[0.1 * i for i in range(len(SIM_ORDER))] if velocity else [],
                       effort=[1.0 * i for i in range(len(SIM_ORDER))] if effort else [],
                       stamp_ns=1_791_650_044_914_984_696)


# ---- joint states ------------------------------------------------------------------------------
def test_joint_states_have_the_22_body_joints_in_gdk_order():
    out = cv.joint_states_dict(sample())
    assert set(out) == {"timestamp", "nums", "states"}
    assert out["nums"] == 22 and [s["name"] for s in out["states"]] == lim.ALL
    assert out["timestamp"] == 1_791_650_044_914_984_696


def test_joint_state_entry_has_every_gdk_key_and_position_mirrors_motor_position():
    out = cv.joint_states_dict(sample())
    entry = next(s for s in out["states"] if s["name"] == "idx61_arm_r_joint1")
    i = SIM_ORDER.index("idx61_arm_r_joint1")
    assert set(entry) == {"name", "mode", "position", "velocity", "effort", "motor_position", "motor_velocity",
                          "motor_current", "error_code"}
    assert entry["motor_position"] == pytest.approx(0.01 * i)
    assert entry["position"] == entry["motor_position"]
    assert entry["motor_velocity"] == pytest.approx(0.1 * i) and entry["velocity"] == entry["motor_velocity"]
    assert entry["effort"] == pytest.approx(1.0 * i)
    assert math.isnan(entry["motor_current"])  # not simulated: NaN, never a made-up number
    assert entry["error_code"] == 0


def test_missing_effort_and_velocity_are_nan():
    entry = cv.joint_states_dict(sample(effort=False, velocity=False))["states"][0]
    assert math.isnan(entry["effort"]) and math.isnan(entry["motor_velocity"])


def test_joint_missing_from_the_sim_is_left_out():
    s = sample()
    i = s.names.index("idx13_head_joint3")
    for field in (s.names, s.position, s.velocity, s.effort):
        del field[i]
    out = cv.joint_states_dict(s)
    assert out["nums"] == 21 and "idx13_head_joint3" not in [e["name"] for e in out["states"]]


# ---- gripper -------------------------------------------------------------------------------------
def test_end_state_reports_both_omnipickers_in_rad():
    s = sample()
    s.position[s.names.index("idx31_gripper_l_inner_joint1")] = -0.785
    s.position[s.names.index("idx71_gripper_r_inner_joint1")] = 0.0
    out = cv.end_state_dict(s)
    left, right = out["left_end_state"], out["right_end_state"]
    assert left["type"] == "omnipicker" and left["controlled"] is True
    assert left["names"] == ["idx31_gripper_l_inner_joint1"]
    st = left["end_states"][0]
    assert set(st) >= {"position", "velocity", "current", "temperature", "effort", "voltage", "enable", "status",
                       "err_code"}
    assert st["position"] == pytest.approx(-0.785) and right["end_states"][0]["position"] == 0.0
    assert math.isnan(st["current"]) and math.isnan(st["temperature"]) and math.isnan(st["voltage"])


def _command(group, positions, target_type="omnipicker", nums=None):
    c = agibot_gdk.JointStates()
    c.group, c.target_type = group, target_type
    for p in positions:
        s = agibot_gdk.JointState()
        s.position = p
        c.states.append(s)
    c.nums = len(positions) if nums is None else nums
    return c


def test_dual_tool_maps_left_then_right_to_leader_joints():
    assert cv.gripper_targets(_command("dual_tool", [-0.785, 0.0])) == {
        "idx31_gripper_l_inner_joint1": pytest.approx(-0.785), "idx71_gripper_r_inner_joint1": 0.0}


def test_single_tool_groups_and_clamping():
    assert cv.gripper_targets(_command("right_tool", [0.5])) == {"idx71_gripper_r_inner_joint1": 0.0}
    left = cv.gripper_targets(_command("left_tool", [-3.0]))
    assert left["idx31_gripper_l_inner_joint1"] == pytest.approx(-0.785, abs=1e-3)


@pytest.mark.parametrize("command", [
    _command("dual_tool", [0.0]),                     # one state for two tools
    _command("left_tool", [0.0], nums=2),             # nums disagrees
    _command("both", [0.0, 0.0]),                     # unknown group
    _command("left_tool", [0.0], target_type="hand"),  # not an omnipicker
])
def test_invalid_gripper_command_raises_invalid_input(command):
    with pytest.raises(cv.InvalidInput):
        cv.gripper_targets(command)


# ---- images --------------------------------------------------------------------------------------
def test_rgba8_becomes_uncompressed_bgr_that_the_solution_decodes():
    h, w = 4, 5
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 20, 30, 255
    img = cv.ros_image_to_gdk(RawImage("rgba8", w, h, w * 4, rgba.tobytes(), 123), agibot_gdk)
    assert isinstance(img, agibot_gdk.Image)
    assert (img.width, img.height, img.timestamp_ns, img.bit_depth) == (w, h, 123, 8)
    assert img.encoding == agibot_gdk.Encoding.UNCOMPRESSED and img.color_format == agibot_gdk.ColorFormat.BGR
    pixels = np.frombuffer(img.data, np.uint8).reshape(img.height, img.width, 3)  # what solutions 09/10 do
    assert pixels[0, 0].tolist() == [30, 20, 10]
    assert len(img.data) == h * w * 3


def test_row_padding_is_removed():
    h, w, step = 2, 3, 16
    buf = np.zeros((h, step), np.uint8)
    buf[:, : w * 4] = np.tile([1, 2, 3, 4], w)
    img = cv.ros_image_to_gdk(RawImage("rgba8", w, h, step, buf.tobytes(), 0), agibot_gdk)
    assert np.frombuffer(img.data, np.uint8).reshape(h, w, 3)[1, 2].tolist() == [3, 2, 1]


def test_rgb8_and_bgr8_inputs():
    px = np.array([[[1, 2, 3]]], np.uint8)
    rgb = cv.ros_image_to_gdk(RawImage("rgb8", 1, 1, 3, px.tobytes(), 0), agibot_gdk)
    bgr = cv.ros_image_to_gdk(RawImage("bgr8", 1, 1, 3, px.tobytes(), 0), agibot_gdk)
    assert list(np.frombuffer(rgb.data, np.uint8)) == [3, 2, 1]
    assert list(np.frombuffer(bgr.data, np.uint8)) == [1, 2, 3]


def test_depth_32fc1_metres_becomes_z16_millimetres():
    d = np.array([[0.5, 1.2344, np.nan], [np.inf, -1.0, 70.0]], np.float32)
    img = cv.ros_image_to_gdk(RawImage("32FC1", 3, 2, 12, d.tobytes(), 7), agibot_gdk)
    assert img.encoding == agibot_gdk.Encoding.UNCOMPRESSED
    assert img.color_format == agibot_gdk.ColorFormat.RS2_FORMAT_Z16 and img.bit_depth == 16
    mm = np.frombuffer(img.data, np.uint16).reshape(img.height, img.width)  # what solution 09 does
    assert mm.tolist() == [[500, 1234, 0], [0, 0, 0]]  # invalid and out-of-range are 0, as on a RealSense


def test_depth_16uc1_passes_through():
    d = np.array([[100, 2000]], np.uint16)
    img = cv.ros_image_to_gdk(RawImage("16UC1", 2, 1, 4, d.tobytes(), 0), agibot_gdk)
    assert np.frombuffer(img.data, np.uint16).tolist() == [100, 2000]


def test_unknown_encoding_raises():
    with pytest.raises(ValueError):
        cv.ros_image_to_gdk(RawImage("yuv422", 1, 1, 2, b"\0\0", 0), agibot_gdk)


# ---- scene camera --------------------------------------------------------------------------------
def test_head_camera_from_scene_yaml():
    cam = cv.head_camera_from_scene(yaml.safe_load(SCENE.read_text()))
    assert cam.shape == (640, 400)
    assert cam.intrinsic == pytest.approx([305.6425170898, 305.6007080078, 318.7979431152, 202.4943847656])
    assert cam.fps == 10.0
    assert cam.color_topic == "/genie_sim/head_front_camera_rgb/image_raw"
    assert cam.depth_topic == "/genie_sim/head_front_camera_depth/image_raw"
    t = cam.head_link3_from_optical
    assert np.allclose(t[:3, 3], (0.10237, 0.02375, 0.10256))
    # optical frame: z forward (+x of head_link3, tilted down), x right (-y), y down
    assert t[:3, 2] @ (1, 0, 0) > 0.85 and t[:3, 2][2] < 0
    assert t[:3, 0] @ (0, -1, 0) > 0.99
    assert t[:3, 1][2] < -0.85


def test_default_head_camera_equals_the_scene():
    from_yaml = cv.head_camera_from_scene(yaml.safe_load(SCENE.read_text()))
    default = cv.DEFAULT_HEAD_CAMERA
    assert np.allclose(default.head_link3_from_optical, from_yaml.head_link3_from_optical)
    assert default.intrinsic == pytest.approx(from_yaml.intrinsic) and default.shape == from_yaml.shape


def test_scene_without_cameras_falls_back_to_default():
    assert cv.head_camera_from_scene({"robot": {}}) is cv.DEFAULT_HEAD_CAMERA


def test_tool_frame_offset_is_along_end_link_z():
    m = cv.VIRTUAL_FRAMES["gripper_l_center_link"]
    assert m[0] == "arm_l_end_link"
    assert np.allclose(m[1], g.matrix((0, 0, 0.125), (0, 0, 0, 1)))
