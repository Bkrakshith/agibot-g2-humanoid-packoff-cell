"""The training solutions' functions, run against this agibot_gdk on a fake sim port (no ROS).

The live acceptance (each solution's main() against Genie Sim) is run_training_solutions.sh.
"""
import importlib.util
import sys

import numpy as np
import pytest

import agibot_gdk
from agibot_gdk import _geometry as g
from agibot_gdk import _runtime
from agibot_gdk._controller import Controller
from fakes import FakePort


@pytest.fixture
def port(training_dir):
    if str(training_dir) not in sys.path:
        sys.path.append(str(training_dir))  # gdk_training; agibot_gdk still resolves to this package
    p = FakePort()
    _runtime.install(Controller(p, now=p.now, sleep=p.sleep))
    yield p
    _runtime.uninstall()


def load(training_dir, script):
    spec = importlib.util.spec_from_file_location("sol_" + script.split("/")[0], training_dir / "solutions" / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.agibot_gdk is agibot_gdk
    return module


def test_01_joints(training_dir, port):
    joints = load(training_dir, "01_joints/joints.py")
    rows = joints.read_joints(agibot_gdk.Robot())
    assert len(rows) == 22 and rows[0].name == "idx01_body_joint1"


def test_02_status(training_dir, port):
    status = load(training_dir, "02_status/status.py")
    robot = agibot_gdk.Robot()
    assert status.control_state(robot).control_mode == 1
    assert "timestamp" not in status.body_state(robot)
    assert [b.soc for b in status.batteries(robot)] == [100.0, 100.0]


def test_03_frames_and_forces(training_dir, port):
    forces = load(training_dir, "03_frames_and_forces/forces.py")
    robot = agibot_gdk.Robot()
    names = [f.name for f in forces.frames(robot)]
    assert "arm_l_end_link" in names and "arm_r_end_link" in names
    assert len(forces.disturbance(robot)) == 14


def test_04_joint_moves(training_dir, port):
    move = load(training_dir, "04_joint_moves/move.py")
    robot = agibot_gdk.Robot()
    robot.set_control_mode(2)
    with pytest.raises(RuntimeError):
        move.move(robot, {"idx11_head_joint1": 0.1}, 0.2)
    assert move.ensure_position_mode(robot) == 2
    assert move.move(robot, {"idx13_head_joint3": 0.2}, 0.2) == 0
    assert move.move_head(robot, 0.0, 0.0, 0.0, 0.2) == 0
    assert move.measured(robot)["idx13_head_joint3"] == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="outside"):
        move.move(robot, {"idx13_head_joint3": 9.0}, 0.2)


def test_05_gripper(training_dir, port):
    gripper = load(training_dir, "05_gripper/gripper.py")
    robot = agibot_gdk.Robot()
    assert gripper.command(robot, gripper.OPEN, gripper.OPEN) == 0
    port.sleep(0.1)
    assert [x.position for x in gripper.read_gripper(robot)] == pytest.approx([-0.785, -0.785])
    assert gripper.close_one(robot, "left") == 0
    port.sleep(0.1)
    assert [x.position for x in gripper.read_gripper(robot)] == pytest.approx([0.0, -0.785])


def test_06_servo(training_dir, port):
    servo = load(training_dir, "06_servo/servo.py")
    robot = agibot_gdk.Robot()
    start = servo.measure(robot, "idx27_arm_l_joint7")
    for tick in range(5):
        assert servo.send(robot, "idx27_arm_l_joint7", servo.setpoint(start, tick / servo.RATE_HZ)) == 0
        port.sleep(1 / servo.RATE_HZ)
    assert servo.measure(robot, "idx27_arm_l_joint7") == pytest.approx(servo.setpoint(start, 4 / servo.RATE_HZ))


def test_07_ee_pose(training_dir, port):
    ee = load(training_dir, "07_ee_pose/ee_pose.py")
    robot = agibot_gdk.Robot()
    start, idle = ee.end_effector(robot, "left"), ee.end_effector(robot, "right")
    goal = start.copy()
    goal[2, 3] += 0.004
    for waypoint in g_line(start, goal):
        assert ee.send(robot, "left", waypoint, idle) == 0
    assert np.allclose(port.ik_calls[-1][1], goal)


def g_line(start, goal, step=0.002):
    count = max(1, int(np.ceil(np.linalg.norm(goal[:3, 3] - start[:3, 3]) / step)))
    out = []
    for i in range(1, count + 1):
        m = start.copy()
        m[:3, 3] = start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * i / count
        out.append(m)
    return out


def test_08_tf(training_dir, port):
    frames = load(training_dir, "08_tf/frames.py")
    robot, tf = agibot_gdk.Robot(), agibot_gdk.TF()
    rows = frames.compare(robot, tf, ["head_link3", "arm_l_end_link"])
    for row in rows:  # the status agrees with get_tf_from_base_link and with the inverse of the lookup
        assert np.allclose(row["status"], row["get_tf_from_base_link"])
        assert np.allclose(row["status"], row["inverse of it"])
    assert frames.tool_offset(tf)["position"] == pytest.approx((0.0, 0.0, -0.125))


def test_10_projection_hits_a_point_in_front_of_the_camera(training_dir, port):
    pytest.importorskip("cv2")
    project = load(training_dir, "10_camera_tf/project.py")
    tf = agibot_gdk.TF()
    camera_from_base = project.camera_from_base(tf)
    head_from_camera = g.from_transform(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3))
    on_axis = port.frames["head_link3"] @ head_from_camera @ np.array([0, 0, 1.0, 1])
    u, v = project.project(camera_from_base @ on_axis, [300.0, 300.0, 320.0, 200.0])
    assert (u, v) == pytest.approx((320.0, 200.0))
