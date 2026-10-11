"""Conversions between Genie Sim's ROS data and the GDK return shapes (no ROS imports).

NaN marks a value the sim does not provide (motor current, gripper temperature, ...). The GDK
fields are kept so code that reads them runs, but no number is made up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agibot_gdk import _geometry as g
from agibot_gdk._limits import ALL, GRIPPERS, clamp_gripper

NAN = float("nan")
# Fingertip centre relative to arm_*_end_link along its +Z (g2cell_motion.geometry.TOOL_OFFSET,
# from FK of the omnipicker finger links + pad). The sim's TF tree has no centre frame.
TOOL_OFFSET = 0.125
VIRTUAL_FRAMES = {
    "gripper_l_center_link": ("arm_l_end_link", g.translation(0.0, 0.0, TOOL_OFFSET)),
    "gripper_r_center_link": ("arm_r_end_link", g.translation(0.0, 0.0, TOOL_OFFSET)),
}


class InvalidInput(ValueError):
    """The robot would reject the request (GDKRes.kInvalidInput)."""


@dataclass
class JointSample:
    """One /joint_states message: names, positions [rad], velocities, efforts, stamp (wall time, ns)."""
    names: list
    position: list
    velocity: list = field(default_factory=list)
    effort: list = field(default_factory=list)
    stamp_ns: int = 0

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.position))

    def _value(self, values, index) -> float:
        return float(values[index]) if index < len(values) else NAN


@dataclass
class RawImage:
    """One sensor_msgs/Image: encoding, size, row step in bytes, raw bytes, stamp (wall time, ns)."""
    encoding: str
    width: int
    height: int
    step: int
    data: bytes
    stamp_ns: int


def joint_states_dict(sample: JointSample) -> dict:
    """Robot.get_joint_states(): the 22 body joints in GDK order. `position` mirrors `motor_position`."""
    index = {n: i for i, n in enumerate(sample.names)}
    states = []
    for name in ALL:
        if name not in index:
            continue
        i = index[name]
        position, velocity = sample._value(sample.position, i), sample._value(sample.velocity, i)
        states.append({"name": name, "mode": 0, "position": position, "velocity": velocity,
                       "effort": sample._value(sample.effort, i), "motor_position": position,
                       "motor_velocity": velocity, "motor_current": NAN, "error_code": 0})
    return {"timestamp": int(sample.stamp_ns), "nums": len(states), "states": states}


def end_state_dict(sample: JointSample) -> dict:
    """Robot.get_end_state(): one OmniPicker per side, its leader joint in rad (0 closed, -0.785 open)."""
    index = {n: i for i, n in enumerate(sample.names)}
    out = {}
    for side, joint in GRIPPERS.items():
        i = index.get(joint)
        value = (lambda values: sample._value(values, i)) if i is not None else (lambda values: NAN)
        out[f"{side}_end_state"] = {
            "controlled": True, "type": "omnipicker", "names": [joint],
            "end_states": [{"id": 1, "enable": True, "position": value(sample.position),
                            "velocity": value(sample.velocity), "effort": value(sample.effort),
                            "current": NAN, "voltage": NAN, "temperature": NAN, "status": 0, "err_code": 0}],
        }
    return out


_TOOL_GROUPS = {"left_tool": ["left"], "right_tool": ["right"], "dual_tool": ["left", "right"]}


def gripper_targets(command) -> dict[str, float]:
    """Robot.move_ee_pos(JointStates) -> {leader joint: target rad}. The GDK value range is the
    sim's (0 closed, -0.785 open), so the mapping is the identity, clamped to that range."""
    sides = _TOOL_GROUPS.get(command.group)
    if sides is None:
        raise InvalidInput(f"group {command.group!r}, expected one of {sorted(_TOOL_GROUPS)}")
    if command.target_type != "omnipicker":
        raise InvalidInput(f"target_type {command.target_type!r}: the sim G2 has OmniPickers ('omnipicker')")
    states = list(command.states)
    if command.nums != len(states) or len(states) != len(sides):
        raise InvalidInput(f"group {command.group} needs {len(sides)} states, got {len(states)} (nums {command.nums})")
    return {GRIPPERS[side]: clamp_gripper(state.position) for side, state in zip(sides, states)}


def _rows(raw: RawImage, channels: int, dtype) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    buf = np.frombuffer(raw.data, np.uint8)[: raw.step * raw.height].reshape(raw.height, raw.step)
    row = buf[:, : raw.width * channels * itemsize]
    return np.ascontiguousarray(row).view(dtype).reshape(raw.height, raw.width, channels)


def ros_image_to_gdk(raw: RawImage, gdk):
    """sensor_msgs/Image -> agibot_gdk Image. Colour: UNCOMPRESSED BGR, 8 bit (solutions 09 and 10
    reshape it to (h, w, 3) and treat it as BGR). Depth: UNCOMPRESSED RS2_FORMAT_Z16, uint16 mm,
    0 where invalid or beyond 65.535 m."""
    enc = raw.encoding.lower()
    if enc in ("rgba8", "bgra8", "rgb8", "bgr8"):
        px = _rows(raw, 4 if enc.endswith("a8") else 3, np.uint8)
        bgr = px[..., [2, 1, 0]] if enc.startswith("rgb") else px[..., :3]
        data = np.ascontiguousarray(bgr).reshape(-1)
        return gdk.Image(raw.stamp_ns, raw.width, raw.height, 8, gdk.Encoding.UNCOMPRESSED, gdk.ColorFormat.BGR, data)
    if enc == "32fc1":
        metres = _rows(raw, 1, np.float32)[..., 0].astype(np.float64)
        mm = np.rint(np.where(np.isfinite(metres), metres, 0.0) * 1000.0)
        mm[(mm <= 0) | (mm > 65535)] = 0
        data = mm.astype(np.uint16).reshape(-1).view(np.uint8)
    elif enc in ("16uc1", "mono16"):
        data = np.ascontiguousarray(_rows(raw, 1, np.uint16)).reshape(-1).view(np.uint8)
    else:
        raise ValueError(f"unsupported image encoding {raw.encoding!r}")
    return gdk.Image(raw.stamp_ns, raw.width, raw.height, 16, gdk.Encoding.UNCOMPRESSED,
                     gdk.ColorFormat.RS2_FORMAT_Z16, data)


@dataclass(frozen=True)
class HeadCamera:
    shape: tuple            # (width, height)
    intrinsic: list         # [fx, fy, cx, cy]
    distortion: list
    fps: float              # nominal, from the scene
    color_topic: str
    depth_topic: str
    head_link3_from_optical: np.ndarray  # maps optical-frame points into head_link3


def _head_camera(xyz, wxyz, width, height, fx, fy, cx, cy, distortion, fps, rgb_topic, depth_topic) -> HeadCamera:
    usd = g.matrix(xyz, g.wxyz_to_xyzw(wxyz))
    return HeadCamera((int(width), int(height)), [float(fx), float(fy), float(cx), float(cy)],
                      [float(d) for d in distortion], float(fps), rgb_topic + "/image_raw",
                      depth_topic + "/image_raw", g.usd_camera_to_optical(usd))


# ros_ws/config/scenes/scene_g2cell.yaml, cameras[0] (official head_front_Camera mount)
DEFAULT_HEAD_CAMERA = _head_camera(
    (0.10237, 0.02375, 0.10256), (0.5945104121873723, 0.3868316461705512, -0.3787291472834019, -0.5945442627086477),
    640, 400, 305.6425170898, 305.6007080078, 318.7979431152, 202.4943847656, [0.0] * 5, 10.0,
    "/genie_sim/head_front_camera_rgb", "/genie_sim/head_front_camera_depth")


def head_camera_from_scene(scene: dict) -> HeadCamera:
    """The head camera block of a Genie Sim scene YAML (USD extrinsic, w-first quaternion)."""
    for cam in scene.get("cameras") or []:
        if str(cam.get("prim_path", "")).startswith("head_link3/"):
            i, s, e, t = cam["intrinsic"], cam["sensor"], cam["extrinsic"], cam["topic"]
            dist = [i.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")]
            return _head_camera(e["xyz"], e["wxyz"], s["width"], s["height"], i["fx"], i["fy"], i["cx"], i["cy"],
                                dist, s.get("freq", 10.0), t["rgb"], t["depth"])
    return DEFAULT_HEAD_CAMERA
