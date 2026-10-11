"""The behaviour behind Robot, TF and Camera, independent of ROS.

The Controller talks to the sim through a Port (agibot_gdk._sim_backend.SimBackend in
production, test/fakes.FakePort in the tests):

    now() / sleep(dt) / wall_ns()           clock
    joint_sample() -> JointSample | None    latest /joint_states
    start_stream(buffer)                    publish buffer.snapshot() on /joint_command at 50 Hz
                                            (raises GuardError when another publisher exists)
    ik(side, T_base_end, seed) -> {joint: rad} | None      MoveIt /compute_ik
    fk(joints, links) -> {link: T_base_link} | None        MoveIt /compute_fk
    lookup(parent, child) -> (T_parent_child, stamp_ns) | None   tf2
    frame_names() -> [str]
    open_image(topic) / latest_image(topic) -> RawImage | None / image_rate(topic) -> Hz | None

Errors follow the robot: a rejected request raises RuntimeError whose text starts with the
GDKRes name (kInvalidInput / kRuntimeError); a request the robot accepts but cannot finish
returns int(GDKRes.kRuntimeError). Success returns 0.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from agibot_gdk import _geometry as g
from agibot_gdk import _types as T
from agibot_gdk._convert import (DEFAULT_HEAD_CAMERA, NAN, VIRTUAL_FRAMES, InvalidInput, end_state_dict,
                                 gripper_targets, joint_states_dict, ros_image_to_gdk)
from agibot_gdk._limits import ARMS, LEFT_ARM, RIGHT_ARM, LimitError, check_and_clamp
from agibot_gdk._motion import CommandBuffer, cap_velocities, interpolate, move_duration, normalize_velocities

OK = 0
RUNTIME_ERROR = int(T.GDKRes.kRuntimeError)
STATUS_FRAMES = ["base_link", "head_link3", "arm_l_end_link", "arm_r_end_link", "gripper_l_center_link",
                 "gripper_r_center_link"]
END_LINKS = {"left": "arm_l_end_link", "right": "arm_r_end_link"}
# Links kept away from the head (as g2cell_motion.robot.CHECK_LINKS): the sim has no arm colliders.
CHECK_LINKS = {s: [f"arm_{s[0]}_link{i}" for i in (3, 4, 5, 6, 7)] + [f"arm_{s[0]}_end_link", f"gripper_{s[0]}_base_link"]
               for s in ARMS}
ARM_GROUPS = {0: LEFT_ARM, 1: RIGHT_ARM, 2: LEFT_ARM + RIGHT_ARM}  # move_arm_joint control_group (as the fake)


class GuardError(RuntimeError):
    """Another node publishes /joint_command."""


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def invalid(message: str) -> RuntimeError:
    return RuntimeError(f"kInvalidInput: {message}")


def runtime(message: str) -> RuntimeError:
    return RuntimeError(f"kRuntimeError: {message}")


class Controller:
    def __init__(self, port, now=None, sleep=None, head_camera=DEFAULT_HEAD_CAMERA, log=None):
        self.port = port
        self.now = now or port.now
        self.sleep = sleep or port.sleep
        self.head_camera = head_camera
        self.log = log or (lambda text: print(f"[agibot_gdk sim] {text}", file=sys.stderr, flush=True))
        self.rate = _env("GDK_SIM_RATE_HZ", 50.0)
        self.settle_tol = _env("GDK_SIM_SETTLE_TOL", 0.02)
        self.settle_timeout = _env("GDK_SIM_SETTLE_TIMEOUT", 2.0)
        self.default_velocity = _env("GDK_SIM_DEFAULT_VEL", 0.3)
        self.head_clearance = _env("GDK_SIM_HEAD_CLEARANCE", 0.20)
        self.max_servo_step = _env("GDK_SIM_MAX_SERVO_STEP", 0.5)
        self.max_ee_step = _env("GDK_SIM_MAX_EE_STEP", 0.05)
        self.joint_wait = _env("GDK_SIM_JOINT_WAIT", 2.0)
        self.buffer = CommandBuffer()
        self._last_ee: dict[str, tuple[np.ndarray, float]] = {}
        self._state_dir = Path(os.environ.get("GDK_SIM_STATE_DIR", Path.home() / ".cache" / "agibot_gdk_sim"))
        self.control_mode = self._load_mode()

    # ---- helpers ---------------------------------------------------------------------------------
    def _load_mode(self) -> int:
        try:
            mode = int((self._state_dir / "control_mode").read_text().strip())
            return mode if mode in (1, 2, 3) else 1
        except (OSError, ValueError):
            return 1

    def _sample(self):
        deadline = self.now() + self.joint_wait
        while True:
            sample = self.port.joint_sample()
            if sample is not None:
                return sample
            if self.now() >= deadline:
                raise runtime("no /joint_states from the sim. Is Genie Sim running (ROS_DOMAIN_ID, use_sim_time false)?")
            self.sleep(0.05)

    def _state(self) -> dict[str, float]:
        """Measured joints, overridden by what is being commanded: the state the next command starts from."""
        state = self._sample().as_dict()
        state.update(self.buffer.snapshot())
        return state

    def _start_stream(self) -> None:
        try:
            self.port.start_stream(self.buffer)
        except GuardError as error:
            raise runtime(f"{error}. Only one node may command the sim: stop the other publisher "
                          "(keyboard_teleop, motion_server, pick_place_demo) first.") from None

    def _check_mode(self) -> None:
        if self.control_mode == 2:
            raise runtime("joint interfaces need control mode 1 or 3, the robot is in 2 (Cartesian impedance)")

    def _validate(self, names, positions, current: dict | None = None) -> list[float]:
        names, positions = list(names), list(positions)
        if not names or len(names) != len(positions):
            raise invalid(f"{len(names)} joint_names and {len(positions)} joint_positions")
        if len(set(names)) != len(names):
            raise invalid("a joint appears twice")
        try:
            targets, notes = check_and_clamp(names, positions, current)
        except LimitError as error:
            raise invalid(str(error)) from None
        for note in notes:
            self.log(note)
        return targets

    # ---- head clearance (the sim's G2 has no arm colliders) -------------------------------------
    def _clearance(self, configs: list[dict], sides: list[str]) -> list[float]:
        links = [link for s in sides for link in CHECK_LINKS[s]] + ["head_link3"]
        cam = self.head_camera.head_link3_from_optical[:, 3]
        out = []
        for joints in configs:
            poses = self.port.fk(joints, links)
            if poses is None:
                raise runtime("the head-clearance check needs MoveIt /compute_fk (move_group); "
                              "set GDK_SIM_HEAD_CLEARANCE=0 to skip it")
            head = poses["head_link3"]
            heads = [head[:3, 3], (head @ cam)[:3]]
            out.append(min(float(np.linalg.norm(poses[l][:3, 3] - h)) for l in links[:-1] for h in heads))
        return out

    def _check_clearance(self, start: dict, goals: list[dict], names) -> None:
        sides = [s for s, arm in ARMS.items() if set(arm) & set(names)]
        if self.head_clearance <= 0 or not sides:
            return
        keys = list(names)
        a = [start[n] for n in keys]
        configs = [start]
        for goal in goals:
            b = [goal[n] for n in keys]
            configs += [{**start, **dict(zip(keys, q))} for q in interpolate(a, b, 8)]
            a = b
        distances = self._clearance(configs, sides)
        worst = min(distances[1:])
        if worst < self.head_clearance and worst < distances[0] - 0.005:
            raise runtime(f"the arm would pass {worst:.3f} m from the head (limit {self.head_clearance} m): "
                          "the sim G2 has no arm colliders, so the sim refuses this motion")

    # ---- joint moves -------------------------------------------------------------------------------
    def joint_control(self, names, positions, velocities) -> int:
        self._check_mode()
        names = list(names)
        state = self._state()
        targets = self._validate(names, positions, state)
        try:
            speeds = cap_velocities(names, normalize_velocities(velocities, len(names), self.default_velocity))
        except ValueError as error:
            raise invalid(str(error)) from None
        start = [state[n] for n in names]
        self._check_clearance(state, [dict(zip(names, targets))], names)
        self._start_stream()
        token = self.buffer.claim(names)
        duration = move_duration(start, targets, speeds)
        steps = max(1, round(duration * self.rate))
        t0 = self.now()
        for i, q in enumerate(interpolate(start, targets, steps), start=1):
            if not self.buffer.owns(names, token):
                self.log("blocking move preempted by a newer command on the same joints")
                return RUNTIME_ERROR
            self.buffer.set(dict(zip(names, q)))
            delay = t0 + i / self.rate - self.now()
            if delay > 0:
                self.sleep(delay)
        deadline = self.now() + self.settle_timeout
        while True:
            measured = self._sample().as_dict()
            error = max(abs(measured.get(n, math.inf) - t) for n, t in zip(names, targets))
            if error <= self.settle_tol:
                return OK
            if not self.buffer.owns(names, token):
                return RUNTIME_ERROR
            if self.now() >= deadline:
                self.log(f"joints did not arrive within {self.settle_timeout} s, largest error {error:.3f} rad")
                return RUNTIME_ERROR
            self.sleep(1.0 / self.rate)

    def move_group(self, group: list[str], positions, velocities) -> int:
        if len(list(positions)) != len(group):
            raise invalid(f"{len(group)} positions expected, got {len(list(positions))}")
        return self.joint_control(group, positions, velocities)

    def move_arm(self, positions, velocities, control_group) -> int:
        group = ARM_GROUPS.get(int(control_group))
        if group is None:
            raise invalid(f"control_group {control_group}: 0 left arm, 1 right arm, 2 both arms")
        return self.move_group(group, positions, velocities)

    def servo(self, names, positions, enable_low_latency: bool = False) -> int:
        self._check_mode()
        names = list(names)
        state = self._state()
        targets = self._validate(names, positions, state)
        for n, t in zip(names, targets):
            if abs(t - state[n]) > self.max_servo_step:
                raise invalid(f"{n} setpoint {t:+.3f} is {abs(t - state[n]):.3f} rad from {state[n]:+.3f}; "
                              f"servo setpoints may move at most {self.max_servo_step} rad")
        if not enable_low_latency:  # as on the robot, low latency switches the collision protection off
            self._check_clearance(state, [dict(zip(names, targets))], names)
        self._start_stream()
        self.buffer.claim(names)
        self.buffer.set(dict(zip(names, targets)))
        return OK

    # ---- end effector --------------------------------------------------------------------------------
    def ee_pose(self, group: int, left: np.ndarray, right: np.ndarray) -> int:
        G = T.EndEffectorControlGroup
        sides = {int(G.kLeftArm): ["left"], int(G.kRightArm): ["right"], int(G.kBothArms): ["left", "right"]}.get(int(group))
        if sides is None:
            raise invalid(f"group {group}: the sim supports kLeftArm, kRightArm and kBothArms")
        poses = {"left": left, "right": right}
        state = self._state()
        solution: dict[str, float] = {}
        for side in sides:
            target = poses[side]
            reference = self._ee_reference(side)
            if reference is None:
                raise runtime(f"no TF for {END_LINKS[side]} in base_link")
            jump = float(np.linalg.norm(target[:3, 3] - reference[:3, 3]))
            if jump > self.max_ee_step:
                raise invalid(f"{side} target is {jump:.3f} m from the current pose; stream poses at most "
                              f"{self.max_ee_step} m apart")
            arm = ARMS[side]
            answer = self.port.ik(side, target, dict(state))
            if not answer or any(n not in answer for n in arm):
                self.log(f"no IK solution for the {side} end effector")
                return RUNTIME_ERROR
            if any(abs(answer[n] - state[n]) > self.max_servo_step for n in arm):
                self.log(f"IK for the {side} end effector jumps to another branch; pose refused")
                return RUNTIME_ERROR
            try:
                values, _ = check_and_clamp(arm, [answer[n] for n in arm], state)
            except LimitError as error:
                self.log(f"IK solution outside the limits: {error}")
                return RUNTIME_ERROR
            solution.update(zip(arm, values))
        self._check_clearance(state, [solution], list(solution))
        self._start_stream()
        self.buffer.claim(list(solution))
        self.buffer.set(solution)
        for side in sides:
            self._last_ee[side] = (poses[side].copy(), self.now())
        return OK

    def _ee_reference(self, side: str):
        last = self._last_ee.get(side)
        if last is not None and self.now() - last[1] < 0.5:
            return last[0]
        found = self.port.lookup("base_link", END_LINKS[side])
        return None if found is None else found[0]

    def move_ee_pos(self, command) -> int:
        try:
            targets = gripper_targets(command)
        except InvalidInput as error:
            raise invalid(str(error)) from None
        self._start_stream()
        self.buffer.claim(list(targets))
        self.buffer.set(targets)
        return OK

    # ---- status -------------------------------------------------------------------------------------
    def joint_states(self) -> dict:
        return joint_states_dict(self._sample())

    def end_state(self) -> dict:
        return end_state_dict(self._sample())

    @staticmethod
    def _synthetic(call: str) -> None:
        if os.environ.get("GDK_SIM_STRICT") == "1":
            raise runtime(f"{call} is not simulated (GDK_SIM_STRICT=1)")

    def whole_body_status(self) -> dict:
        self._synthetic("get_whole_body_status")
        status = {"timestamp": self.port.wall_ns(), "left_end_model": "omnipicker", "right_end_model": "omnipicker"}
        for part in ("right_arm", "left_arm", "right_end", "left_end", "waist", "lift", "neck", "chassis"):
            status[f"{part}_error"] = 0
        for side in ("right", "left"):
            status[f"{side}_arm_control"] = True
            status[f"{side}_arm_estop"] = False
        return status

    def motion_control_status(self):
        status = T.MotionControlStatus()
        status.mode, status.control_mode = 1, self.control_mode
        for name in STATUS_FRAMES:
            found = self.lookup("base_link", name)
            if found is None:
                continue
            status.frame_names.append(name)
            status.frame_poses.append(g.to_pose(found[0], T))
            twist, wrench = T.Twist(), T.Wrench()
            for v in (twist.linear, twist.angular, wrench.force, wrench.torque):
                v.x = v.y = v.z = NAN  # the sim reports no link twists or wrenches
            status.twists.append(twist)
            status.wrenches.append(wrench)
        status.arm_disturbance_force = [NAN] * 14
        return status

    def chassis_power_state(self):
        self._synthetic("get_chassis_power_state")
        power = T.ChassisPowerState()
        power.timestamp = self.port.wall_ns()
        for _ in range(2):
            battery = T.BatteryState()
            battery.battery_soc = 100.0
            battery.battery_output_voltage = battery.battery_output_current = battery.battery_temperature = NAN
            power.battery_states.append(battery)
        return power

    def set_control_mode(self, mode) -> int:
        mode = int(mode)
        if mode not in (1, 2, 3):
            raise invalid("control_mode must be 1, 2 or 3")
        self.control_mode = mode
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            (self._state_dir / "control_mode").write_text(str(mode))
        except OSError as error:
            self.log(f"could not persist the control mode: {error}")
        return OK

    # ---- TF ------------------------------------------------------------------------------------------
    def lookup(self, parent: str, child: str):
        """(T_parent_child, stamp_ns) or None; resolves the virtual gripper centre frames."""
        (rp, op), (rc, oc) = VIRTUAL_FRAMES.get(parent, (parent, np.eye(4))), VIRTUAL_FRAMES.get(child, (child, np.eye(4)))
        if rp == rc:
            found = (np.eye(4), self.port.wall_ns()) if rp in self.port.frame_names() else None
        else:
            found = self.port.lookup(rp, rc)
        if found is None:
            return None
        return g.inverse(op) @ found[0] @ oc, found[1]

    def frame_names(self) -> list[str]:
        names = list(self.port.frame_names())
        return names + [v for v, (base, _) in VIRTUAL_FRAMES.items() if base in names and v not in names]

    def tf_from_base(self, child: str):
        found = self.lookup("base_link", child)
        if found is None:
            raise runtime(f"no transform from base_link to {child}")
        return g.to_transform(found[0], T)

    def lookup_latest(self, target: str, source: str, return_timestamp: bool):
        # GDK direction (training exercise 8): the result maps points in `target` into `source`,
        # i.e. tf2's lookup_transform(target_frame=source, source_frame=target).
        found = self.lookup(source, target)
        if found is None:
            raise runtime(f"no transform between {target} and {source}")
        return g.to_transform(found[0], T), (int(found[1]) if return_timestamp else None)

    def tf_from_sensor(self, kind):
        name = getattr(kind, "name", str(kind))
        if name == "kHeadRGBDToHeadLink3":
            return g.to_transform(self.head_camera.head_link3_from_optical, T)
        if name == "kHeadDepthToHeadColor":  # the sim renders colour and depth from one camera prim
            return g.to_transform(np.eye(4), T)
        raise runtime(f"{name}: this sensor is not simulated")

    # ---- cameras -------------------------------------------------------------------------------------
    def camera_topic(self, kind) -> str | None:
        name = getattr(kind, "name", str(kind))
        topics = {"kHeadColor": self.head_camera.color_topic, "kHeadDepth": self.head_camera.depth_topic}
        for item in os.environ.get("GDK_SIM_CAMERA_TOPICS", "").split(";"):  # e.g. kHandLeftColor=/topic
            if "=" in item:
                key, topic = item.split("=", 1)
                topics[key.strip()] = topic.strip()
        return topics.get(name)

    def _simulated(self, kind) -> str:
        topic = self.camera_topic(kind)
        if topic is None:
            raise runtime(f"{getattr(kind, 'name', kind)} is not simulated (the sim publishes no wrist cameras yet)")
        return topic

    def open_cameras(self, kinds) -> None:
        for kind in kinds:
            topic = self.camera_topic(kind)
            if topic is not None:
                self.port.open_image(topic)

    def image_shape(self, kind) -> tuple[int, int]:
        topic = self._simulated(kind)
        raw = self.port.latest_image(topic)
        if raw is not None:
            return (raw.width, raw.height)
        if getattr(kind, "name", "") in ("kHeadColor", "kHeadDepth"):
            return tuple(self.head_camera.shape)
        raise runtime(f"no image yet on {topic}")

    def image_fps(self, kind) -> float:
        topic = self._simulated(kind)
        measured = self.port.image_rate(topic)
        if measured:
            return float(measured)
        return float(self.head_camera.fps) if getattr(kind, "name", "") in ("kHeadColor", "kHeadDepth") else 0.0

    def intrinsic(self, kind):
        self._simulated(kind)
        if getattr(kind, "name", "") not in ("kHeadColor", "kHeadDepth"):
            raise runtime(f"no intrinsics known for {kind.name}")
        intrinsic = T.CameraIntrinsic(list(self.head_camera.intrinsic))
        intrinsic.distortion = list(self.head_camera.distortion)
        return intrinsic

    def latest_image(self, kind, timeout_ms: float):
        topic = self.camera_topic(kind)
        deadline = self.now() + max(float(timeout_ms), 0.0) / 1000.0
        while True:
            raw = self.port.latest_image(topic) if topic else None
            if raw is not None:
                return ros_image_to_gdk(raw, T)
            if self.now() >= deadline:
                return None
            self.sleep(min(0.01, deadline - self.now()))
