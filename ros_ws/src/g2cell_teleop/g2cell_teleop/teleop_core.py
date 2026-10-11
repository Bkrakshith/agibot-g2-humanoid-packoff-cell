"""Teleop state machine, free of ROS so it can be unit-tested.

Each key nudges the active arm's end-effector target in the robot base frame
(+X forward, +Y left, +Z up) or rotates it about the base axes. The ROS node turns
targets into joint angles through MoveIt /compute_ik.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Quat = tuple[float, float, float, float]  # x, y, z, w
Vec3 = tuple[float, float, float]

LIN_STEP_RANGE = (0.001, 0.05)  # m
GRIPPER_STEP = 0.1  # rad per key press

# key -> (dx, dy, dz) in units of lin_step
TRANSLATE_KEYS = {
    "w": (1, 0, 0), "s": (-1, 0, 0),
    "a": (0, 1, 0), "d": (0, -1, 0),
    "q": (0, 0, 1), "e": (0, 0, -1),
}
# key -> (droll, dpitch, dyaw) in units of ang_step
ROTATE_KEYS = {
    "i": (0, 1, 0), "k": (0, -1, 0),
    "j": (1, 0, 0), "l": (-1, 0, 0),
    "u": (0, 0, 1), "o": (0, 0, -1),
}
ARM_KEYS = {"1": "left", "2": "right"}
QUIT_KEYS = {"x", "\x1b"}
# waist yaw (idx05_body_joint5): ',' turns the torso left (+yaw), '.' right
WAIST_KEYS = {",": 1, ".": -1}
MOTION_KEYS = set(TRANSLATE_KEYS) | set(ROTATE_KEYS) | {"g", "h"} | set(WAIST_KEYS)

ARMS = ("left", "right")
# scene init pose (scene_pnp_g2_op_cam.yaml init_joint_pos), joints 1..7, degrees
DEFAULT_HOME_DEG = [0.0, -45.0, 0.0, -75.0, 0.0, -55.0, 0.0]
_ARM_IDX = {"left": (2, "l"), "right": (6, "r")}

HELP = """\
G2 teleop               1/2 select left/right arm
  w/s  +X/-X (forward)   i/k  pitch +/-      g  close gripper
  a/d  +Y/-Y (left)      j/l  roll  +/-      h  open gripper
  q/e  +Z/-Z (up)        u/o  yaw   +/-      [ ]  halve/double step
  r  record start/stop   n  mark success     f  mark failure
  ,/.  waist yaw left/right (5 deg, held)   t  auto-face: turn the waist toward the target
  0  home both arms      p  pause/resume (holds targets)
  x or Esc  quit
"""


@dataclass(frozen=True)
class Pose:
    position: Vec3
    orientation: Quat


@dataclass
class TeleopState:
    targets: dict[str, Pose]
    gripper: dict[str, float]
    gripper_limits: tuple[float, float]
    active: str = "left"
    lin_step: float = 0.01
    ang_step: float = math.radians(5)
    gripper_closed_is_lower: bool = False  # omnipicker: 0 rad closed, -0.785 rad open
    recording: bool = False
    outcome: str = "unknown"  # "true" | "false" | "unknown", as bag_to_hdf5 --success
    paused: bool = False
    # per arm: ((xmin, ymin, zmin), (xmax, ymax, zmax)) in the base frame, or None
    workspace: dict | None = None
    waist: float = 0.0                       # commanded waist yaw target (rad), held across arm moves
    waist_step: float = math.radians(5)
    waist_limits: tuple = (math.radians(-100), math.radians(100))
    auto_face: bool = False                  # turn the waist toward the active arm's target on each move


def _quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotate_world(q: Quat, rpy: Vec3) -> Quat:
    """Apply a roll/pitch/yaw rotation about the fixed base axes to orientation q."""
    out = _quat_mul(_quat_from_rpy(*rpy), q)
    n = math.sqrt(sum(c * c for c in out))
    return tuple(c / n for c in out)


def clamp_to_box(old: Vec3, new: Vec3, box) -> Vec3:
    """Clamp new into box. An axis already outside the box may only move back toward it,
    so a start pose outside the box never jumps."""
    if box is None:
        return tuple(new)
    lo, hi = box
    out = []
    for o, n, l, h in zip(old, new, lo, hi):
        out.append(min(max(h, o), max(min(l, o), n)))
    return tuple(out)


def set_position(state: TeleopState, arm: str, new: Vec3) -> None:
    target = state.targets[arm]
    box = (state.workspace or {}).get(arm)
    state.targets[arm] = Pose(clamp_to_box(target.position, new, box), target.orientation)


def arm_joint_names(arm: str) -> list[str]:
    idx, side = _ARM_IDX[arm]
    return [f"idx{idx}{i}_arm_{side}_joint{i}" for i in range(1, 8)]


def home_joint_targets(left_deg=None, right_deg=None) -> dict[str, dict[str, float]]:
    """Home joint positions (rad) per arm; default is the scene init pose for both arms."""
    out = {}
    for arm, deg in (("left", left_deg), ("right", right_deg)):
        deg = DEFAULT_HOME_DEG if deg is None else list(deg)
        if len(deg) != 7:
            raise ValueError(f"{arm} home pose needs 7 joint values, got {len(deg)}")
        out[arm] = {n: math.radians(float(d)) for n, d in zip(arm_joint_names(arm), deg)}
    return out


def step_toward(current: dict[str, float], target: dict[str, float], max_step: float):
    """Move each joint at most max_step toward its target. Returns (next, arrived)."""
    nxt, done = dict(current), True
    for name, goal in target.items():
        if name not in nxt:
            continue
        d = goal - nxt[name]
        if abs(d) > max_step:
            nxt[name] += math.copysign(max_step, d)
            done = False
        else:
            nxt[name] = goal
    return nxt, done


def switch_arm(state: TeleopState) -> str:
    state.active = "right" if state.active == "left" else "left"
    return "select"


def _gripper_closed_open(state: TeleopState) -> tuple[float, float]:
    lo, hi = sorted(state.gripper_limits)
    return (lo, hi) if state.gripper_closed_is_lower else (hi, lo)


def toggle_gripper(state: TeleopState) -> str | None:
    """Send the active gripper to whichever end (closed/open) it is further from."""
    if state.paused:
        return None
    closed, opened = _gripper_closed_open(state)
    g = state.gripper[state.active]
    state.gripper[state.active] = opened if abs(g - closed) <= abs(g - opened) else closed
    return "gripper"


def handle_key(state: TeleopState, key: str) -> str | None:
    """Update state for one key press. Returns what changed, an episode event, or None."""
    if key in QUIT_KEYS:
        return "quit"
    if key in ARM_KEYS:
        state.active = ARM_KEYS[key]
        return "select"
    if key == "r":
        state.recording = not state.recording
        if state.recording:
            state.outcome = "unknown"
            return "record_start"
        return "record_stop"
    if key == "n":
        state.outcome = "true"
        return "mark_success"
    if key == "f":
        state.outcome = "false"
        return "mark_failure"
    if key == "0":
        return "home"
    if key == "p":
        state.paused = not state.paused
        return "pause" if state.paused else "resume"
    if key == "t":
        state.auto_face = not state.auto_face
        return "auto_face_on" if state.auto_face else "auto_face_off"
    if state.paused and key in MOTION_KEYS:
        return None
    if key in WAIST_KEYS:
        lo, hi = state.waist_limits
        state.waist = min(hi, max(lo, state.waist + WAIST_KEYS[key] * state.waist_step))
        return "waist"
    target = state.targets[state.active]
    if key in TRANSLATE_KEYS:
        d = TRANSLATE_KEYS[key]
        set_position(state, state.active, tuple(p + c * state.lin_step for p, c in zip(target.position, d)))
        return "arm"
    if key in ROTATE_KEYS:
        rpy = tuple(c * state.ang_step for c in ROTATE_KEYS[key])
        state.targets[state.active] = Pose(target.position, rotate_world(target.orientation, rpy))
        return "arm"
    if key in ("g", "h"):
        lo, hi = sorted(state.gripper_limits)
        toward_closed = -1 if state.gripper_closed_is_lower else 1
        sign = toward_closed if key == "g" else -toward_closed
        value = state.gripper[state.active] + sign * GRIPPER_STEP
        state.gripper[state.active] = min(hi, max(lo, value))
        return "gripper"
    if key in ("[", "]"):
        factor = 0.5 if key == "[" else 2.0
        lo, hi = LIN_STEP_RANGE
        state.lin_step = min(hi, max(lo, state.lin_step * factor))
        return "step"
    return None


def status_line(state: TeleopState) -> str:
    t = state.targets[state.active]
    flags = "".join((" REC" if state.recording else "", " PAUSED" if state.paused else "",
                     " AUTO-FACE" if state.auto_face else "",
                     f" outcome={state.outcome}" if state.outcome != "unknown" else ""))
    return (f"[{state.active}] xyz={tuple(round(c, 3) for c in t.position)} "
            f"grip={state.gripper[state.active]:.2f} step={state.lin_step:.3f} "
            f"waist={math.degrees(state.waist):+.0f}deg{flags}")
