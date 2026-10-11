"""The public Robot / TF / Camera calls on top of the Controller and a fake sim port."""
import math

import numpy as np
import pytest

import agibot_gdk
from agibot_gdk import _geometry as g
from agibot_gdk import _runtime
from agibot_gdk._controller import Controller
from agibot_gdk._convert import RawImage
from fakes import HEAD3, FakePort


@pytest.fixture
def port():
    p = FakePort()
    _runtime.install(Controller(p, now=p.now, sleep=p.sleep))
    yield p
    _runtime.uninstall()


@pytest.fixture
def robot(port):
    return agibot_gdk.Robot()


def _request(names, positions, velocities=None):
    r = agibot_gdk.JointControlReq()
    r.life_time = 5.0
    r.joint_names, r.joint_positions = list(names), list(positions)
    r.joint_velocities = list(velocities or [])
    return r


def _servo(names, positions):
    r = agibot_gdk.JointServoControlReq()
    r.control_period = 0.035
    r.joint_names, r.joint_positions = list(names), list(positions)
    return r


# ---- without gdk_init ----------------------------------------------------------------------------
def test_calls_before_gdk_init_raise_a_clear_error():
    _runtime.uninstall()
    with pytest.raises(RuntimeError, match="gdk_init"):
        agibot_gdk.Robot().get_joint_states()


# ---- reading -------------------------------------------------------------------------------------
def test_get_joint_states_shape(robot):
    out = robot.get_joint_states()
    assert out["nums"] == 22 and len(out["states"]) == 22
    head = next(s for s in out["states"] if s["name"] == "idx13_head_joint3")
    assert head["motor_position"] == pytest.approx(0.1) == head["position"]


def test_whole_body_status_is_healthy_and_complete(robot):
    s = robot.get_whole_body_status()
    for part in ("right_arm", "left_arm", "right_end", "left_end", "waist", "lift", "neck", "chassis"):
        assert s[f"{part}_error"] == 0
    assert s["left_arm_estop"] is False and s["right_arm_estop"] is False
    assert s["left_arm_control"] is True and s["left_end_model"] == "omnipicker"
    assert isinstance(s["timestamp"], int)


def test_motion_control_status_frames_from_tf_and_unavailable_wrenches_are_nan(robot):
    s = robot.get_motion_control_status()
    assert (s.mode, s.control_mode, s.error_code, s.error_msg) == (1, 1, 0, "")
    assert len(s.frame_names) == len(s.frame_poses) == len(s.wrenches) == len(s.twists)
    i = list(s.frame_names).index("arm_l_end_link")
    assert np.allclose(g.from_pose(s.frame_poses[i]), FakePort().frames["arm_l_end_link"])
    assert "gripper_l_center_link" in s.frame_names
    assert math.isnan(s.wrenches[i].force.z) and len(s.arm_disturbance_force) == 14
    assert all(math.isnan(v) for v in s.arm_disturbance_force)
    assert s.collision_pairs_1 == [] and s.collision_pairs_2 == []


def test_chassis_power_is_synthetic_full(robot):
    p = robot.get_chassis_power_state()
    assert len(p.battery_states) == 2 and p.battery_states[0].battery_soc == 100.0
    assert math.isnan(p.battery_states[0].battery_output_voltage)
    assert p.emergency_stop_pedal_state == 0


def test_strict_mode_refuses_synthetic_status(port, monkeypatch):
    monkeypatch.setenv("GDK_SIM_STRICT", "1")
    with pytest.raises(RuntimeError, match="kRuntimeError"):
        agibot_gdk.Robot().get_chassis_power_state()


# ---- control mode ---------------------------------------------------------------------------------
def test_control_mode_is_validated_and_persists_like_on_the_robot(port):
    robot = agibot_gdk.Robot()
    with pytest.raises(RuntimeError):
        robot.set_control_mode(7)
    assert robot.set_control_mode(2) == 0
    # a new process (new controller) finds the mode the previous one set
    _runtime.install(Controller(port, now=port.now, sleep=port.sleep))
    assert agibot_gdk.Robot().get_motion_control_status().control_mode == 2


def test_joint_moves_are_refused_in_cartesian_impedance_mode(robot):
    robot.set_control_mode(2)
    with pytest.raises(RuntimeError):
        robot.joint_control_request(_request(["idx11_head_joint1"], [0.1], [0.2]))
    robot.set_control_mode(1)
    assert robot.joint_control_request(_request(["idx11_head_joint1"], [0.1], [0.2])) == 0


# ---- blocking joint moves ----------------------------------------------------------------------
def test_blocking_move_streams_a_quintic_and_returns_on_arrival(robot, port):
    t0 = port.t
    assert robot.joint_control_request(_request(["idx13_head_joint3"], [0.3], [0.2])) == 0
    assert port.position["idx13_head_joint3"] == pytest.approx(0.3)
    assert port.t - t0 == pytest.approx(1.875 * 0.2 / 0.2, abs=0.05)
    assert port.started == 1


def test_target_outside_the_limits_is_rejected_with_outside(robot, port):
    with pytest.raises(RuntimeError, match="outside"):
        robot.joint_control_request(_request(["idx13_head_joint3"], [9.0], [0.2]))
    assert port.started == 0  # nothing streamed


def test_mismatched_lengths_and_unknown_joints_are_invalid_input(robot):
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.joint_control_request(_request(["idx11_head_joint1", "idx12_head_joint2"], [0.1], [0.2]))
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.joint_control_request(_request(["gripper"], [0.1], [0.2]))
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.joint_control_request(_request([], [], []))


def test_target_beyond_the_sim_limit_is_clamped(robot, port):
    assert robot.move_head_joint([1.2, 0.0, 0.1], [0.5] * 3) == 0
    assert port.position["idx11_head_joint1"] == pytest.approx(0.781)


def test_move_that_never_arrives_times_out_with_runtime_error_code(robot, port):
    port.stuck.add("idx11_head_joint1")
    assert robot.move_head_joint([0.3, 0.0, 0.1], [0.5] * 3) == int(agibot_gdk.GDKRes.kRuntimeError)


def test_group_moves_map_to_their_joints(robot, port):
    assert robot.move_waist_joint([-0.1, 0.1, 0.2, 0.0, 0.0], [0.3] * 5) == 0
    assert port.position["idx03_body_joint3"] == pytest.approx(0.2)
    assert robot.move_arm_joint([0.1] * 7, [0.3] * 7, 1) == 0  # 1: right arm
    assert port.position["idx61_arm_r_joint1"] == pytest.approx(0.1) and port.position["idx21_arm_l_joint1"] == 0.0
    assert robot.move_arm_joint([0.2] * 14, [0.3] * 14, 2) == 0
    assert port.position["idx21_arm_l_joint1"] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.move_head_joint([0.1, 0.1], [0.2, 0.2])


def test_arm_move_through_the_head_is_refused(robot, port):
    port.arm_near_head = True
    with pytest.raises(RuntimeError, match="head"):
        robot.joint_control_request(_request(["idx21_arm_l_joint1"], [0.5], [0.3]))
    assert port.position["idx21_arm_l_joint1"] == 0.0


def test_head_only_move_needs_no_fk(robot, port):
    port.fk_available = False
    assert robot.move_head_joint([0.1, 0.0, 0.1], [0.5] * 3) == 0


def test_arm_move_without_fk_is_refused_unless_the_check_is_off(robot, port, monkeypatch):
    port.fk_available = False
    with pytest.raises(RuntimeError, match="compute_fk"):
        robot.joint_control_request(_request(["idx21_arm_l_joint1"], [0.5], [0.3]))
    monkeypatch.setenv("GDK_SIM_HEAD_CLEARANCE", "0")
    _runtime.install(Controller(port, now=port.now, sleep=port.sleep))
    assert agibot_gdk.Robot().joint_control_request(_request(["idx21_arm_l_joint1"], [0.5], [0.3])) == 0


def test_second_publisher_on_joint_command_refuses_motion(robot, port):
    port.conflicts = ["/keyboard_teleop"]
    with pytest.raises(RuntimeError, match="kRuntimeError.*keyboard_teleop"):
        robot.move_head_joint([0.1, 0.0, 0.1], [0.5] * 3)


# ---- servo -------------------------------------------------------------------------------------------
def test_servo_sets_the_target_without_waiting(robot, port):
    t0 = port.t
    assert robot.joint_servo_control(_servo(["idx27_arm_l_joint7"], [0.05]), enable_low_latency=False) == 0
    assert port.t == t0  # returned at once
    assert port.buffer.snapshot()["idx27_arm_l_joint7"] == pytest.approx(0.05)


def test_servo_rejects_a_jump(robot):
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.joint_servo_control(_servo(["idx27_arm_l_joint7"], [1.0]))


def test_servo_preempts_a_running_blocking_move_on_the_same_joint(robot, port):
    # a blocking move is interrupted when a servo request claims its joint mid-way
    calls = []

    def sleep(dt, original=port.sleep):
        original(dt)
        if not calls and port.t > 0.3:
            calls.append(1)
            robot.joint_servo_control(_servo(["idx11_head_joint1"], [port.position["idx11_head_joint1"]]))
    _runtime.install(Controller(port, now=port.now, sleep=sleep))
    r = agibot_gdk.Robot().move_head_joint([0.5, 0.0, 0.1], [0.3] * 3)
    assert r == int(agibot_gdk.GDKRes.kRuntimeError)
    assert port.position["idx11_head_joint1"] < 0.5


# ---- end-effector pose -------------------------------------------------------------------------
def _ee(group, left, right):
    p = agibot_gdk.EndEffectorPose()
    p.life_time, p.group = 0.02, int(group)
    g.fill_pose(p.left_end_effector_pose, left)
    g.fill_pose(p.right_end_effector_pose, right)
    return p


def test_ee_pose_goes_to_ik_for_arm_end_link_in_base_link(robot, port):
    left, right = port.frames["arm_l_end_link"].copy(), port.frames["arm_r_end_link"]
    left[2, 3] += 0.002
    G = agibot_gdk.EndEffectorControlGroup
    assert robot.end_effector_pose_control(_ee(G.kLeftArm, left, right)) == 0
    side, target, seed = port.ik_calls[-1]
    assert side == "left" and np.allclose(target, left)
    assert port.buffer.snapshot()["idx21_arm_l_joint1"] == pytest.approx(0.01)
    assert len(port.ik_calls) == 1  # the right arm stays where it is


def test_ee_pose_both_arms(robot, port):
    G = agibot_gdk.EndEffectorControlGroup
    assert robot.end_effector_pose_control(_ee(G.kBothArms, port.frames["arm_l_end_link"],
                                               port.frames["arm_r_end_link"])) == 0
    assert [c[0] for c in port.ik_calls] == ["left", "right"]


def test_ee_pose_jump_and_bad_group_are_rejected(robot, port):
    far = port.frames["arm_l_end_link"].copy()
    far[0, 3] += 0.2
    G = agibot_gdk.EndEffectorControlGroup
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.end_effector_pose_control(_ee(G.kLeftArm, far, port.frames["arm_r_end_link"]))
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.end_effector_pose_control(_ee(0, far, far))


def test_ee_pose_without_ik_solution_returns_runtime_error(robot, port):
    port.ik_answer = lambda *a: None
    G = agibot_gdk.EndEffectorControlGroup
    r = robot.end_effector_pose_control(_ee(G.kRightArm, port.frames["arm_l_end_link"], port.frames["arm_r_end_link"]))
    assert r == int(agibot_gdk.GDKRes.kRuntimeError)


def test_ee_pose_rejects_an_ik_flip(robot, port):
    port.ik_answer = lambda side, target, seed: {n: seed[n] + 2.0 for n in seed if "arm_l" in n}
    G = agibot_gdk.EndEffectorControlGroup
    r = robot.end_effector_pose_control(_ee(G.kLeftArm, port.frames["arm_l_end_link"], port.frames["arm_r_end_link"]))
    assert r == int(agibot_gdk.GDKRes.kRuntimeError)


# ---- gripper -----------------------------------------------------------------------------------------
def test_move_ee_pos_drives_the_leader_joints(robot, port):
    c = agibot_gdk.JointStates()
    c.group, c.target_type, c.nums = "dual_tool", "omnipicker", 2
    for p in (-0.785, 0.0):
        s = agibot_gdk.JointState()
        s.position = p
        c.states.append(s)
    assert robot.move_ee_pos(c) == 0
    snap = port.buffer.snapshot()
    assert snap["idx31_gripper_l_inner_joint1"] == pytest.approx(-0.785) and snap["idx71_gripper_r_inner_joint1"] == 0
    port.sleep(0.1)
    end = robot.get_end_state()
    assert end["left_end_state"]["end_states"][0]["position"] == pytest.approx(-0.785)


def test_bad_gripper_command_raises(robot):
    c = agibot_gdk.JointStates()
    c.group, c.target_type, c.nums = "dual_tool", "omnipicker", 0
    with pytest.raises(RuntimeError, match="kInvalidInput"):
        robot.move_ee_pos(c)


# ---- TF ------------------------------------------------------------------------------------------------
def test_lookup_transform_latest_returns_pose_of_target_in_source(port):
    tf = agibot_gdk.TF()
    transform, stamp = tf.lookup_transform_latest("base_link", "head_link3")
    # solution 10 calls this head_from_base: it maps base_link points into head_link3
    assert np.allclose(g.from_transform(transform), g.inverse(HEAD3)) and stamp is None
    _, stamp = tf.lookup_transform_latest("base_link", "head_link3", return_timestamp=True)
    assert stamp == port.wall_ns()


def test_get_tf_from_base_link_is_the_pose_in_base_link(port):
    tf = agibot_gdk.TF()
    assert np.allclose(g.from_transform(tf.get_tf_from_base_link("head_link3")), HEAD3)


def test_virtual_gripper_centre_frame(port):
    tf = agibot_gdk.TF()
    assert "gripper_l_center_link" in tf.get_all_frame_names()
    assert tf.can_transform("arm_l_end_link", "gripper_l_center_link")
    offset, _ = tf.lookup_transform_latest("arm_l_end_link", "gripper_l_center_link")
    assert np.allclose(g.from_transform(offset)[:3, 3], (0, 0, -0.125))  # end link seen from the centre
    centre = g.from_transform(tf.get_tf_from_base_link("gripper_l_center_link"))
    assert np.allclose(centre, port.frames["arm_l_end_link"] @ g.matrix((0, 0, 0.125), (0, 0, 0, 1)))


def test_unknown_frame_raises(port):
    tf = agibot_gdk.TF()
    assert tf.can_transform("base_link", "nope") is False
    with pytest.raises(RuntimeError):
        tf.get_tf_from_base_link("nope")


def test_sensor_extrinsics(port):
    tf = agibot_gdk.TF()
    S = agibot_gdk.SensorExtrinsicType
    cam = g.from_transform(tf.get_tf_from_sensor(S.kHeadRGBDToHeadLink3))
    assert np.allclose(cam[:3, 3], (0.10237, 0.02375, 0.10256)) and cam[:3, 2][0] > 0.85
    assert np.allclose(g.from_transform(tf.get_tf_from_sensor(S.kHeadDepthToHeadColor)), np.eye(4))
    with pytest.raises(RuntimeError, match="kRuntimeError"):
        tf.get_tf_from_sensor(S.kLeftHandRGBDToArmLEndLink)


# ---- camera --------------------------------------------------------------------------------------------
def test_head_camera_info_and_images(port):
    C = agibot_gdk.CameraType
    cam = agibot_gdk.Camera([C.kHeadColor, C.kHeadDepth])
    assert cam.get_image_shape(C.kHeadColor) == (640, 400)
    assert cam.get_image_fps(C.kHeadColor) == 10.0  # nominal until frames arrive
    port.rates["/genie_sim/head_front_camera_rgb/image_raw"] = 9.5
    assert cam.get_image_fps(C.kHeadColor) == 9.5
    assert cam.get_camera_intrinsic(C.kHeadDepth).intrinsic == pytest.approx([305.6425, 305.6007, 318.798, 202.494],
                                                                              abs=1e-3)
    assert cam.get_latest_image(C.kHeadColor, 100.0) is None  # no frame yet
    rgba = np.zeros((400, 640, 4), np.uint8)
    port.images["/genie_sim/head_front_camera_rgb/image_raw"] = RawImage("rgba8", 640, 400, 2560, rgba.tobytes(), 5)
    img = cam.get_latest_image(C.kHeadColor, 100.0)
    assert (img.width, img.height, img.timestamp_ns) == (640, 400, 5)
    assert "/genie_sim/head_front_camera_depth/image_raw" in port.opened
    assert cam.close_camera() == agibot_gdk.GDKRes.kSuccess


def test_wrist_cameras_are_not_simulated(port):
    C = agibot_gdk.CameraType
    cam = agibot_gdk.Camera([C.kHandLeftColor, C.kHeadColor])
    assert cam.get_latest_image(C.kHandLeftColor, 10.0) is None
    with pytest.raises(RuntimeError, match="kRuntimeError"):
        cam.get_image_shape(C.kHandLeftColor)
    with pytest.raises(RuntimeError, match="kRuntimeError"):
        cam.get_camera_intrinsic(C.kHandLeftColor)


def test_camera_type_not_requested_raises(port):
    C = agibot_gdk.CameraType
    with pytest.raises(RuntimeError):
        agibot_gdk.Camera([C.kHeadColor]).get_latest_image(C.kHeadDepth, 10.0)


def test_saved_pose_with_sim_drift_can_be_commanded_back(robot, port):
    port.position["idx01_body_joint1"] = 0.005  # sim rests slightly past the real limit 0.0
    assert robot.joint_control_request(_request(["idx01_body_joint1"], [0.005], [0.2])) in (0, 3)
