"""G2 teleop node: keyboard (always) plus an optional SpaceMouse, with episode recording.

Runs in a plain terminal (works over SSH / `docker exec -it`). Inputs move the active arm's
end-effector target; MoveIt /compute_ik (move_group launched with use_ros2_control:=false)
turns the target into arm joint angles, which are streamed on /joint_command together with
the gripper leader joints. This node must be the only /joint_command publisher.

Keys r / n / f start-stop a g2cell_recorder episode and mark it success / failure; 0 homes
both arms; p pauses. , / . yaw the waist (idx05_body_joint5, held across arm moves; both arms are
carried with the torso), t toggles auto-face (each arm move first turns the waist toward the
active target, g2cell_motion waist_policy). The waist command is streamed on /joint_command too. All logic that can be tested without ROS lives in teleop_core, motion,
devices, recorder_control and params.
"""
from __future__ import annotations

import math
import os
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from geometry_msgs.msg import PoseStamped, Twist as TwistMsg
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Joy, JointState

from g2cell_teleop.devices import KeyboardDevice, PySpaceMouseDevice, SpaceNavTopicDevice, apply_input
from g2cell_teleop.motion import TwistFilter
from g2cell_teleop.params import DEFAULTS, home_from, limits_from, spacemouse_config_from, workspace_from
from g2cell_teleop.recorder_control import RecorderController
from g2cell_teleop.teleop_core import ARMS, HELP, Pose, TeleopState, handle_key, status_line, step_toward
from g2cell_teleop.waist import auto_face, waist_config_from


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("g2cell_keyboard_teleop")
        self.p = p = {k: self.declare_parameter(k, v).value for k, v in DEFAULTS.items()}
        self.frame = p["base_frame"]
        self.groups = {"left": p["left_group"], "right": p["right_group"]}
        self.ik_links = {"left": p["left_ik_link"], "right": p["right_ik_link"]}
        self.arm_prefix = {"left": p["left_joint_match"], "right": p["right_joint_match"]}
        self.gripper_joint = {"left": p["left_gripper_joint"], "right": p["right_gripper_joint"]}
        self.gripper_limits = (p["gripper_lower"], p["gripper_upper"])
        self.closed_is_lower = p["gripper_closed_is_lower"]  # omnipicker: 0 = closed
        self.ik_timeout = p["ik_timeout"]
        self.rate = p["rate_hz"]
        self.home = home_from(p)
        self.workspace = workspace_from(p)
        self.filter = TwistFilter(limits_from(p))
        self.waist_joint = p["waist_joint"]
        self.waist_cfg = waist_config_from(p)
        self._waist_cmd: float | None = None     # ramped toward state.waist; None = no waist in /joint_states

        self._lock = threading.Lock()
        self._joint_state: JointState | None = None
        self._arm_cmd: dict[str, dict[str, float]] = {}
        self.state: TeleopState | None = None
        self.homing = False

        # the engine publishes /joint_states best-effort; a reliable subscriber would get nothing
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data)
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, "/joint_command", qos)
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.create_timer(1.0 / self.rate, self._publish)

        self.spacemouse = None
        if p["device"] == "spacemouse":
            cfg = spacemouse_config_from(p)
            if p["spacemouse_source"] == "pyspacemouse":
                self.spacemouse = PySpaceMouseDevice(cfg)
            else:
                dev = SpaceNavTopicDevice(cfg)
                # sensor-data QoS (best effort) matches both reliable and best-effort publishers
                self.create_subscription(
                    TwistMsg, p["spacemouse_twist_topic"],
                    lambda m: dev.on_twist((m.linear.x, m.linear.y, m.linear.z),
                                           (m.angular.x, m.angular.y, m.angular.z)),
                    qos_profile_sensor_data)
                self.create_subscription(Joy, p["spacemouse_joy_topic"],
                                         lambda m: dev.on_joy(list(m.buttons)), qos_profile_sensor_data)
                self.spacemouse = dev
        elif p["device"] != "keyboard":
            raise ValueError(f"device must be keyboard or spacemouse, got {p['device']!r}")

        self.recorder = None
        if p["recording_enabled"]:
            self.recorder = RecorderController(
                scene=p["scene"], seed=p["seed"], episodes_dir=p["episodes_dir"],
                log=lambda msg: print(f"\r[rec] {msg}", flush=True))

    # ---- inputs -------------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState):
        with self._lock:
            self._joint_state = msg

    def _robot_state(self, override: dict[str, float] | None = None) -> RobotState:
        rs = RobotState()
        with self._lock:
            js = self._joint_state
        if override:
            merged = dict(zip(js.name, js.position))
            merged.update(override)
            js = JointState(name=list(merged), position=list(merged.values()))
        rs.joint_state = js
        return rs

    def _call(self, client, request, timeout=2.0):
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    # ---- start-up -----------------------------------------------------------------------
    def _fk_targets(self, override=None) -> dict[str, Pose] | None:
        req = GetPositionFK.Request()
        req.header.frame_id = self.frame
        req.fk_link_names = [self.ik_links[a] for a in ARMS]
        req.robot_state = self._robot_state(override)
        res = self._call(self.fk, req)
        if res is None or res.error_code.val != 1:
            self.get_logger().error(f"FK failed: {None if res is None else res.error_code.val}")
            return None
        targets = {}
        for arm, ps in zip(ARMS, res.pose_stamped):
            pos, q = ps.pose.position, ps.pose.orientation
            targets[arm] = Pose((pos.x, pos.y, pos.z), (q.x, q.y, q.z, q.w))
        return targets

    def initialise(self) -> bool:
        """Seed targets from the measured EE poses; arm commands hold the measured joints."""
        for client, name in ((self.ik, "/compute_ik"), (self.fk, "/compute_fk")):
            if not client.wait_for_service(timeout_sec=30.0):
                self.get_logger().error(f"{name} not available: is move_group running?")
                return False
        while rclpy.ok() and self._joint_state is None:
            self.get_logger().info("waiting for /joint_states ...", throttle_duration_sec=2.0)
            threading.Event().wait(0.1)

        targets = self._fk_targets()
        if targets is None:
            return False
        js = self._joint_state
        measured = dict(zip(js.name, js.position))
        for arm in ARMS:
            self._arm_cmd[arm] = {n: v for n, v in measured.items() if self.arm_prefix[arm] in n}
            missing = set(self.home[arm]) - set(self._arm_cmd[arm])
            if missing:
                self.get_logger().warn(f"home pose joints not in /joint_states: {sorted(missing)}")
        gripper = {a: measured.get(self.gripper_joint[a], self.gripper_limits[1]) for a in ARMS}
        self.state = TeleopState(targets=targets, gripper=gripper, gripper_limits=self.gripper_limits,
                                 gripper_closed_is_lower=self.closed_is_lower, workspace=self.workspace)
        s = self.state
        s.waist_step = math.radians(self.p["waist_step_deg"])
        s.waist_limits = (math.radians(self.p["waist_min_deg"]), math.radians(self.p["waist_max_deg"]))
        s.auto_face = bool(self.p["auto_face"])
        if self.waist_joint in measured:
            self._waist_cmd = measured[self.waist_joint]
            s.waist = min(max(self._waist_cmd, s.waist_limits[0]), s.waist_limits[1])
        else:
            self.get_logger().warn(f"{self.waist_joint} not in /joint_states: waist keys disabled")
        return True

    # ---- waist ------------------------------------------------------------------------------
    def _waist_override(self, yaw=None) -> dict[str, float]:
        if self._waist_cmd is None:
            return {}
        return {self.waist_joint: self.state.waist if yaw is None else yaw}

    def on_waist_key(self) -> None:
        """',' / '.': the torso turns and carries both arms (arm joints unchanged), so the EE targets
        are re-seeded by FK at the new waist yaw."""
        if self._waist_cmd is None:
            return
        with self._lock:
            cmd = {n: v for arm in ARMS for n, v in self._arm_cmd[arm].items()}
        cmd.update(self._waist_override())
        targets = self._fk_targets(override=cmd)
        if targets is not None:
            self.state.targets.update(targets)
        self.filter.reset()

    def face_and_solve(self, arm: str) -> bool:
        """Auto-face: turn the waist toward the arm's target (target fixed), solve IK in the turned
        frame; fall back to the current waist if that fails."""
        s = self.state
        if s.auto_face and self._waist_cmd is not None and arm == s.active:
            d = auto_face(s, self.waist_cfg)
            if abs(d.turn) > 1e-6 and self.solve(arm, waist=d.yaw):
                s.waist = d.yaw
                if d.needs_base:
                    print(f"\r[waist] at its limit: base would need {math.degrees(d.base_turn):+.0f} deg", flush=True)
                return True
        return self.solve(arm)

    def waist_ramp(self, dt: float) -> None:
        if self._waist_cmd is None:
            return
        step = self.p["waist_speed"] * dt
        d = self.state.waist - self._waist_cmd
        with self._lock:
            self._waist_cmd += max(-step, min(step, d))

    # ---- IK + output ----------------------------------------------------------------------
    def solve(self, arm: str, waist: float | None = None) -> bool:
        """IK for the arm's target with the waist at `waist` (default: the commanded waist target)."""
        target = self.state.targets[arm]
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.groups[arm]
        r.ik_link_name = self.ik_links[arm]
        r.robot_state = self._robot_state(self._waist_override(waist) or None)
        r.avoid_collisions = False
        r.timeout.nanosec = int(self.ik_timeout * 1e9)
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = target.position
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = target.orientation
        r.pose_stamped = ps
        res = self._call(self.ik, req)
        if res is None or res.error_code.val != 1:
            return False
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        with self._lock:
            for name in self._arm_cmd[arm]:
                if name in sol:
                    self._arm_cmd[arm][name] = sol[name]
        return True

    def _publish(self):
        if self.state is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            for arm in ARMS:
                for name, value in self._arm_cmd[arm].items():
                    msg.name.append(name)
                    msg.position.append(value)
                msg.name.append(self.gripper_joint[arm])
                msg.position.append(self.state.gripper[arm])
            if self._waist_cmd is not None:
                msg.name.append(self.waist_joint)
                msg.position.append(self._waist_cmd)
        self.pub.publish(msg)

    # ---- homing -----------------------------------------------------------------------------
    def home_step(self, dt: float) -> None:
        """Ramp the arm joint commands toward the home pose; re-seed targets on arrival."""
        done = True
        with self._lock:
            for arm in ARMS:
                self._arm_cmd[arm], arrived = step_toward(self._arm_cmd[arm], self.home[arm],
                                                          self.p["home_joint_speed"] * dt)
                done = done and arrived
            cmd = {n: v for arm in ARMS for n, v in self._arm_cmd[arm].items()}
        if done:
            self.homing = False
            cmd.update(self._waist_override())
            targets = self._fk_targets(override=cmd)
            if targets is not None:
                self.state.targets.update(targets)
            self.filter.reset()
            print("\r[home] reached; targets re-seeded", flush=True)

    # ---- episode events ---------------------------------------------------------------------
    def on_action(self, action: str) -> None:
        s = self.state
        if action == "home":
            self.homing = True
            if self.p["home_resets_waist"] and self._waist_cmd is not None:
                s.waist = 0.0
            print("\r[home] moving both arms to the home pose ...", flush=True)
        elif action == "waist":
            self.on_waist_key()
        elif action in ("auto_face_on", "auto_face_off"):
            print(f"\r[waist] auto-face {'on' if s.auto_face else 'off'}", flush=True)
        elif action in ("pause", "resume"):
            self.filter.reset()
            print(f"\r[{action}]", flush=True)
        elif action in ("record_start", "record_stop", "mark_success", "mark_failure"):
            if self.recorder is None:
                print("\r[rec] recording_enabled is false", flush=True)
                s.recording = False
                return
            try:
                if action == "record_start":
                    self.recorder.start()
                elif action == "record_stop":
                    self.recorder.stop(s.outcome)
                else:
                    self.recorder.mark(s.outcome)
                    print(f"\r[rec] outcome: success={s.outcome}", flush=True)
            except RuntimeError as exc:  # previous episode still converting
                s.recording = False
                print(f"\r[rec] {exc}", flush=True)

    def status(self) -> None:
        print(f"\r{status_line(self.state)}", flush=True)

    def shutdown_recorder(self) -> None:
        if self.recorder is not None and self.recorder.busy:
            self.recorder.stop(self.state.outcome if self.state else None)
            print("\r[rec] waiting for the episode to convert ...", flush=True)
            self.recorder.wait(300)


def _stdin_reader(fd):
    def read_key():
        if select.select([fd], [], [], 0)[0]:
            return os.read(fd, 1).decode(errors="ignore") or None
        return None
    return read_key


def run(node: KeyboardTeleop) -> None:
    """Single control loop: keyboard keys, SpaceMouse twist, homing, IK. No executor callbacks
    run here, so the blocking IK/FK service calls cannot deadlock."""
    fd = sys.stdin.fileno()
    tty_ok = os.isatty(fd)
    old = termios.tcgetattr(fd) if tty_ok else None
    keyboard = KeyboardDevice(_stdin_reader(fd)) if tty_ok else None
    if not tty_ok:
        node.get_logger().warn("stdin is not a terminal: keyboard disabled (run with a TTY for r/n/f/0/p)")
    print(HELP, flush=True)
    period = 1.0 / node.rate
    last = time.monotonic()
    try:
        if tty_ok:
            tty.setcbreak(fd)
        while rclpy.ok():
            if tty_ok:
                select.select([fd], [], [], period)
            else:
                time.sleep(period)
            now = time.monotonic()
            dt, last = now - last, now
            s = node.state

            for key in (keyboard.poll().events if keyboard else ()):
                arm = s.active
                before = s.targets[arm]
                action = handle_key(s, key)
                if action == "quit":
                    return
                if action == "waist" and node.homing:
                    continue
                if action == "arm":
                    if node.homing:
                        s.targets[arm] = before
                        continue
                    if not node.face_and_solve(arm):
                        s.targets[arm] = before  # unreachable: keep the last good target
                        print(f"\r[{arm}] IK failed, target not moved", flush=True)
                        continue
                if action is not None:
                    node.on_action(action)
                    node.status()

            if node.spacemouse is not None and not node.homing:
                arm = s.active
                before = s.targets[arm]
                actions = apply_input(s, node.spacemouse.poll(), node.filter, dt)
                if "arm" in actions and not node.face_and_solve(arm):
                    s.targets[arm] = before
                    node.filter.reset()
                if "select" in actions:
                    node.filter.reset()
                if any(a != "arm" for a in actions):
                    node.status()

            if node.homing:
                node.home_step(dt)
            node.waist_ramp(dt)
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = KeyboardTeleop()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        if node.initialise():
            run(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_recorder()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
