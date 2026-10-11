"""Enums and structs of the agibot_gdk 3.3.8 Python API.

Names, attributes and the values of the members that agibot-gdk-training's fake defines follow
that fake (Apache-2.0, commit 29e4dc3). Further members come from the GDK v2.3.4 reference
(agibot-gdk-skil, MIT); their numeric values are not verified against a 3.3.8 robot, so read
enum members by name, never by number.
"""
from __future__ import annotations

from enum import Enum, IntEnum


class GDKRes(IntEnum):
    kSuccess = 0
    kInvalidInput = 1
    kInvalidOutput = 2
    kRuntimeError = 3
    kUnknown = 4


class EndEffectorControlGroup(IntEnum):
    # kLeftArm/kRightArm/kBothArms values as in the 3.3.8 training fake. The v2.3.4 reference
    # lists 4/8/12 for them and the *Waist* groups below, which the sim does not support.
    kUnknown = 0
    kLeftArm = 1
    kRightArm = 2
    kBothArms = 3
    kLeftArmWaistLift = 20
    kRightArmWaistLift = 24
    kBothArmsWaistLift = 28
    kLeftArmWaistPitch = 36
    kRightArmWaistPitch = 40
    kBothArmsWaistPitch = 44
    kLeftArmWaist = 52
    kRightArmWaist = 56
    kBothArmsWaist = 60


class CameraType(Enum):
    kCameraUnknown = 0
    kHeadBackFisheye = 1
    kHeadLeftFisheye = 2
    kHeadRightFisheye = 3
    kHeadStereoLeft = 4
    kHeadStereoRight = 5
    kHandLeftColor = 6
    kHandRightColor = 7
    kHeadColor = 8
    kHeadDepth = 9
    kHandLeftUpperColor = 10
    kHandRightUpperColor = 11
    kHandLeftLowerColor = 12
    kHandRightLowerColor = 13
    kHandLeftUpperDepth = 14
    kHandRightUpperDepth = 15
    kHandLeftLowerDepth = 16
    kHandRightLowerDepth = 17


class Encoding(Enum):
    UNCOMPRESSED = 0
    JPEG = 1
    PNG = 2


class ColorFormat(Enum):
    RGB = 0
    BGR = 1
    GRAY8 = 2
    GRAY16 = 3
    RS2_FORMAT_Z16 = 4
    RGBA = 5
    BGRA = 6


class SensorExtrinsicType(Enum):
    kHeadDepthToHeadColor = 1
    kHeadRGBDToHeadLink3 = 2
    kLeftHandRGBDToArmLEndLink = 3
    kRightHandRGBDToArmREndLink = 4
    kHeadLeftStereoToHeadRightStereo = 5
    kLeftHandDepthToLeftHandColor = 6
    kRightHandDepthToRightHandColor = 7
    kHeadLeftStereoToHeadLink3 = 8
    kHeadRightStereoToHeadLink3 = 9
    kHeadLeftFisheyeToHeadLink3 = 10
    kHeadRightFisheyeToHeadLink3 = 11
    kHeadBackFisheyeToHeadLink3 = 12
    kChassisFrontLidarToBaseLink = 13
    kChassisBackLidarToBaseLink = 14
    kChassisBackLidarToChassisFrontLidar = 15
    kChassisMid360ImuToChassisMid360Lidar = 16
    kChassisImuToBaseLink = 17


class _Repr:
    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"


class Vector3(_Repr):
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class Quaternion(_Repr):
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.x, self.y, self.z, self.w = x, y, z, w


class Pose(_Repr):
    def __init__(self) -> None:
        self.position = Vector3()
        self.orientation = Quaternion()


class Transform(_Repr):
    def __init__(self) -> None:
        self.translation = Vector3()
        self.rotation = Quaternion()


class Twist(_Repr):
    def __init__(self) -> None:
        self.linear = Vector3()
        self.angular = Vector3()


class Wrench(_Repr):
    def __init__(self) -> None:
        self.force = Vector3()
        self.torque = Vector3()


class JointControlReq(_Repr):
    def __init__(self) -> None:
        self.life_time = 0.0
        self.joint_names: list[str] = []
        self.joint_positions: list[float] = []
        self.joint_velocities: list[float] = []
        self.detail = ""


class JointServoControlReq(_Repr):
    def __init__(self) -> None:
        self.control_period = 0.0
        self.joint_names: list[str] = []
        self.joint_positions: list[float] = []
        self.joint_velocities: list[float] = []


class JointState(_Repr):
    def __init__(self) -> None:
        self.name = ""
        self.mode = 0
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0
        self.motor_position = 0.0
        self.motor_velocity = 0.0
        self.motor_current = 0.0
        self.error_code = 0


class JointStates(_Repr):
    def __init__(self) -> None:
        self.group = ""
        self.target_type = ""
        self.states: list[JointState] = []
        self.nums = 0
        self.timestamp = 0


class EndEffectorPose(_Repr):
    def __init__(self) -> None:
        self.life_time = 0.0
        self.group = 0
        self.left_end_effector_pose = Pose()
        self.right_end_effector_pose = Pose()


class CameraIntrinsic(_Repr):
    def __init__(self, intrinsic: list[float]) -> None:
        self.intrinsic = intrinsic  # [fx, fy, cx, cy] in px
        self.distortion = [0.0] * 5


class Image(_Repr):
    """What Camera.get_latest_image returns. `data` is a flat uint8 numpy array."""

    def __init__(self, timestamp_ns: int = 0, width: int = 0, height: int = 0, bit_depth: int = 8,
                 encoding: Encoding = Encoding.UNCOMPRESSED, color_format: ColorFormat = ColorFormat.BGR,
                 data=b"") -> None:
        self.timestamp_ns, self.width, self.height, self.bit_depth = timestamp_ns, width, height, bit_depth
        self.encoding, self.color_format, self.data = encoding, color_format, data

    def __repr__(self) -> str:
        return (f"Image({self.width}x{self.height}, {self.encoding.name}, {self.color_format.name}, "
                f"t={self.timestamp_ns}, {len(self.data)} bytes)")


class MotionControlStatus(_Repr):
    def __init__(self) -> None:
        self.mode = 1
        self.control_mode = 1
        self.error_code = 0
        self.error_msg = ""
        self.frame_names: list[str] = []
        self.frame_poses: list[Pose] = []
        self.twists: list[Twist] = []
        self.wrenches: list[Wrench] = []
        self.arm_disturbance_force: list[float] = []
        self.collision_pairs_1: list[str] = []
        self.collision_pairs_2: list[str] = []


class BatteryState(_Repr):
    def __init__(self) -> None:
        self.battery_soc = 0.0
        self.battery_output_voltage = 0.0
        self.battery_output_current = 0.0
        self.battery_temperature = 0.0
        self.battery_charging_status = 0


class ChassisPowerState(_Repr):
    def __init__(self) -> None:
        self.timestamp = 0
        self.battery_states: list[BatteryState] = []
        self.charge_plug_insert_state = 0
        self.emergency_stop_pedal_state = 0
