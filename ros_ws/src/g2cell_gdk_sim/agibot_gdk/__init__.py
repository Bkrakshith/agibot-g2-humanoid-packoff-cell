"""agibot_gdk 3.3.8 API backed by Genie Sim (ROS 2): GDK Python code for the AgiBot G2 runs unchanged.

Same public names, signatures and return shapes as the agibot_gdk 3.3.8 calls used by
agibot-gdk-training (https://github.com/SailnMobula/agibot-gdk-training, Apache-2.0, commit
29e4dc3), whose fake module defined the surface this package follows. Contains no vendor code.

    import agibot_gdk
    agibot_gdk.gdk_init()          # starts one rclpy node in a background thread
    robot = agibot_gdk.Robot()
    ...
    agibot_gdk.gdk_release()

Importing this module does not import rclpy; gdk_init() does. Differences from the robot are
listed in the package README (g2cell_gdk_sim).
"""
from __future__ import annotations

import os
import sys

from agibot_gdk import _geometry as _g
from agibot_gdk import _runtime
from agibot_gdk._types import (BatteryState, CameraIntrinsic, CameraType, ChassisPowerState, ColorFormat, Encoding,
                               EndEffectorControlGroup, EndEffectorPose, GDKRes, Image, JointControlReq, JointServoControlReq,
                               JointState, JointStates, MotionControlStatus, Pose, Quaternion, SensorExtrinsicType,
                               Transform, Twist, Vector3, Wrench)
from agibot_gdk._limits import HEAD, WAIST

__version__ = "3.3.8-sim"

__all__ = [
    "gdk_init", "gdk_release", "get_app_version", "GDKRes", "CameraType", "Encoding", "ColorFormat",
    "SensorExtrinsicType", "EndEffectorControlGroup", "Vector3", "Quaternion", "Pose", "Transform", "Twist", "Wrench",
    "JointControlReq", "JointServoControlReq", "JointState", "JointStates", "EndEffectorPose", "CameraIntrinsic",
    "Image", "MotionControlStatus", "ChassisPowerState", "BatteryState", "Robot", "TF", "Camera",
]


def gdk_init() -> GDKRes:
    """Starts the sim backend: one rclpy node spun by a MultiThreadedExecutor in a daemon thread.

    Nothing is published on /joint_command until the first motion command."""
    if _runtime.active():
        return GDKRes.kSuccess
    try:
        from agibot_gdk._controller import Controller
        from agibot_gdk._sim_backend import SimBackend, load_head_camera

        backend = SimBackend()
        _runtime.install(Controller(backend, head_camera=load_head_camera()), backend)
    except Exception as error:  # noqa: BLE001 - the GDK reports failure through the return code
        print(f"[agibot_gdk sim] gdk_init failed: {type(error).__name__}: {error}", file=sys.stderr)
        return GDKRes.kRuntimeError
    return GDKRes.kSuccess


def gdk_release() -> GDKRes:
    """Stops streaming, destroys the node. The sim holds the last commanded joint positions."""
    backend = _runtime.uninstall()
    if backend is not None:
        backend.shutdown()
    return GDKRes.kSuccess


def get_app_version() -> str:
    return f"{__version__} (Genie Sim backend, ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')})"


def _c():
    return _runtime.controller()


class Robot:
    """Joint states, status, joint/servo/Cartesian/gripper commands on the simulated G2."""

    def get_joint_states(self) -> dict:
        return _c().joint_states()

    def get_whole_body_status(self) -> dict:
        return _c().whole_body_status()

    def get_motion_control_status(self) -> MotionControlStatus:
        return _c().motion_control_status()

    def get_end_state(self) -> dict:
        return _c().end_state()

    def get_chassis_power_state(self) -> ChassisPowerState:
        return _c().chassis_power_state()

    def set_control_mode(self, control_mode: int) -> int:
        return _c().set_control_mode(control_mode)

    def clear_motion_control_error(self) -> int:
        _c()
        return 0

    def joint_control_request(self, request: JointControlReq) -> int:
        return _c().joint_control(request.joint_names, request.joint_positions, request.joint_velocities)

    def move_head_joint(self, positions: list[float], velocities: list[float]) -> int:
        return _c().move_group(HEAD, positions, velocities)

    def move_waist_joint(self, positions: list[float], velocities: list[float]) -> int:
        return _c().move_group(WAIST, positions, velocities)

    def move_arm_joint(self, positions: list[float], velocities: list[float], control_group: int) -> int:
        return _c().move_arm(positions, velocities, control_group)

    def joint_servo_control(self, request: JointServoControlReq, enable_low_latency: bool = False) -> int:
        return _c().servo(request.joint_names, request.joint_positions, enable_low_latency)

    def move_ee_pos(self, command: JointStates) -> int:
        return _c().move_ee_pos(command)

    def end_effector_pose_control(self, pose: EndEffectorPose) -> int:
        return _c().ee_pose(pose.group, _g.from_pose(pose.left_end_effector_pose),
                            _g.from_pose(pose.right_end_effector_pose))


class TF:
    """The sim's TF tree (tf2 on /tf and /tf_static) plus the virtual gripper centre frames."""

    def get_all_frame_names(self) -> list[str]:
        return _c().frame_names()

    def can_transform(self, target_frame: str, source_frame: str) -> bool:
        return _c().lookup(source_frame, target_frame) is not None

    def get_tf_from_base_link(self, child_frame_id: str) -> Transform:
        return _c().tf_from_base(child_frame_id)

    def lookup_transform_latest(self, target_frame: str, source_frame: str,
                                return_timestamp: bool = False) -> tuple[Transform, int | None]:
        return _c().lookup_latest(target_frame, source_frame, return_timestamp)

    def get_tf_from_sensor(self, sensor_extrinsic_type: SensorExtrinsicType) -> Transform:
        return _c().tf_from_sensor(sensor_extrinsic_type)


class Camera:
    """Head colour and depth from the sim's head_front camera. Wrist cameras are not simulated."""

    def __init__(self, camera_types: list[CameraType] | None = None) -> None:
        self.types = set(camera_types or CameraType)
        self._opened = False
        if _runtime.active():  # the real Camera starts its streams here; first frames ~3 s later
            self._open()

    def _open(self):
        controller = _c()
        if not self._opened:
            controller.open_cameras(self.types)
            self._opened = True
        return controller

    def _requested(self, camera_type: CameraType):
        if camera_type not in self.types:
            raise RuntimeError(f"kInvalidInput: {camera_type} was not requested from this Camera")
        return self._open()

    def get_image_shape(self, camera_type: CameraType) -> tuple[int, int]:
        return self._open().image_shape(camera_type)

    def get_image_fps(self, camera_type: CameraType) -> float:
        return self._open().image_fps(camera_type)

    def get_camera_intrinsic(self, camera_type: CameraType) -> CameraIntrinsic:
        return self._open().intrinsic(camera_type)

    def get_latest_image(self, camera_type: CameraType, timeout_ms: float) -> Image | None:
        return self._requested(camera_type).latest_image(camera_type, timeout_ms)

    def close_camera(self) -> GDKRes:
        return GDKRes.kSuccess
