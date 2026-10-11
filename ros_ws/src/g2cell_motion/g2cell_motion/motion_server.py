"""MoveArm action server: the single /joint_command publisher for scripted G2 arm motion.

    ros2 run g2cell_motion motion_server --ros-args -p use_sim_time:=true
    ros2 action send_goal /g2cell/move_arm g2cell_interfaces/action/MoveArm "{arm: left, named_pose: home}"

Needs move_group (IK only, use_ros2_control:=false) for /compute_ik and /compute_fk, and
/joint_states from the sim. Refuses to start while another node publishes /joint_command (quit
the keyboard teleop / pick_place_demo first), and stops publishing if one appears later.
Nothing is published until the first goal arrives; from then on the commanded pose is held
and streamed at rate_hz (best-effort, like g2cell_teleop).

Waist-first (parameter waist_mode, default waist_first; goal.waist_mode overrides): Cartesian goals
turn the waist (idx05_body_joint5) so the target sits within +-waist_cone_deg of the torso's forward
direction (g2cell_motion.waist_policy), solve the 7-DoF IK in the turned frame from several seeds,
drop backward-twisting solutions, and stream waist + arm on one quintic profile. The waist turn moves
both arms, so a goal that turns it is refused (BUSY) while the other arm is moving.

Planning speed/determinism (docs/OPTIMISATION.md "Motion planning"): local IK from the staged URDF with
/compute_ik confirming each answer (local_ik, ik_confirm, ik_fallback_attempts), head clearance from local
FK instead of one /compute_fk call per path sample, a bounded budget (plan_max_ik_solves,
plan_max_seconds), and a plan cache (plan_cache file; "" = off) keyed by goal + start state + parameters
and invalidated when the cell layout changes (assets/cell/build.py constants, cell_colliders.json).
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState

from g2cell_interfaces.action import MoveArm
from g2cell_motion.clearance import head_points, load_fixture_boxes, path_clearance_fk
from g2cell_motion.geometry import BASE_Z, down_quat, quat_is_unset, segment_durations
from g2cell_motion.guard import publisher_conflicts
from g2cell_motion.ik_client import IkHelper
from g2cell_motion.named_poses import load_named_poses, resolve
from g2cell_motion.plan_cache import PlanCache, find_build_py, goal_key, layout_key
from g2cell_motion.planner import (WAIST_ABSOLUTE, WAIST_FIRST, WAIST_FIXED, WAIST_MODES, PlanBudget,
                                   plan_chain_waist)
from g2cell_motion.robot import (ARMS, CHECK_LINKS, FINGER_LINKS, GRIPPERS, HEAD_CAM_IN_LINK3, HEAD_LINK, HOME_DEG,
                                 arm_joint_names, gripper_value, home_joints)
from g2cell_motion.waist_policy import TwistLimits, WaistConfig, clamp_waist
from g2cell_motion.streamer import CommandBuffer, run_segment

TOPIC = "/joint_command"
PLAN_VERSION = 3      # 2: absolute-mode via waist sequencing + lookahead points (PickBag transfer fix)
                      # 3: fixture boxes (cell_colliders.json) checked along the path (datagen carton hit)
URDFS = ("/geniesim_assets/scenes/scene_g2cell/robot.urdf", "/geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf")


class GuardError(RuntimeError):
    pass


def _xyz(p):
    return (p.position.x, p.position.y, p.position.z)


class MotionServer(Node, IkHelper):
    def __init__(self):
        super().__init__("g2cell_motion_server")
        p = self.declare_parameter
        self.rate = p("rate_hz", 50.0).value
        self.max_vel = p("max_joint_vel", 1.1).value
        self.min_t = p("min_segment_time", 0.6).value
        self.gripper_time = p("gripper_time", 0.8).value
        self.min_clear = p("head_clearance", 0.20).value
        self.ik_attempts = p("ik_attempts", 6).value
        guard_wait = p("guard_discovery_wait", 2.0).value
        default_poses = os.path.join(get_package_share_directory("g2cell_motion"), "config", "named_poses.yaml")
        poses_file = p("named_poses_file", default_poses).value
        action_name = p("action_name", "/g2cell/move_arm").value
        self.waist_mode = p("waist_mode", WAIST_FIRST).value
        if self.waist_mode not in WAIST_MODES:
            raise ValueError(f"waist_mode must be one of {WAIST_MODES}, got {self.waist_mode!r}")
        self.waist_cfg = WaistConfig(cone_deg=p("waist_cone_deg", 30.0).value,
                                     max_turn_deg=p("waist_max_turn_deg", 100.0).value,
                                     pivot=tuple(p("waist_pivot", [0.258, 0.0]).value),
                                     joint=p("waist_joint", "idx05_body_joint5").value)
        # fallback waist yaws: steps back toward the current yaw, and how many orientations each tries (0 = all)
        self.waist_fallback_step = p("waist_fallback_step_deg", 15.0).value
        self.waist_fallback_orients = p("waist_fallback_orients", 0).value
        self.twist = TwistLimits(j1_max_dev_deg=p("twist_j1_max_deg", 90.0).value,
                                 jump_max=p("twist_jump_max", 1.5).value)
        with open(poses_file) as f:
            self.named = load_named_poses(yaml.safe_load(f))

        # honour remapping (-r /joint_command:=...) in the guard as well as in the publisher
        self.topic = self.resolve_topic_name(TOPIC)
        self.cbg = ReentrantCallbackGroup()
        self.init_ik(callback_group=self.cbg)
        self.init_fk(callback_group=self.cbg)
        urdf = p("urdf", next((u for u in URDFS if os.path.isfile(u)), "")).value
        self.max_solves = p("plan_max_ik_solves", 600).value
        self.max_seconds = p("plan_max_seconds", 60.0).value
        if p("local_ik", True).value and urdf:
            self.init_local_ik(urdf, confirm=p("ik_confirm", True).value,
                               fallback_attempts=p("ik_fallback_attempts", 0).value)
        colliders = p("colliders", "/geniesim_assets/scenes/g2cell/cell_colliders.json").value
        # arm/fingertip points vs the cell's fixture boxes along every path (clearance.py); boxes moved into
        # base_link (the FK frame: BASE_Z above the cell floor, pinned base at the cell origin)
        self.fixture_margin = p("fixture_margin", 0.04).value
        self.fixtures = None
        if p("fixture_check", True).value and colliders and os.path.isfile(colliders):
            self.fixtures = [{**b, "c": [b["c"][0], b["c"][1], b["c"][2] - BASE_Z]}
                             for b in load_fixture_boxes(colliders)]
            self.get_logger().info(f"fixture check: {len(self.fixtures)} boxes from {colliders}, "
                                   f"margin {self.fixture_margin:.2f} m")
        self.plan_cache = PlanCache(p("plan_cache", "/g2cell_output/plan_cache/motion_server.json").value or None,
                                    layout_key(build_py=p("layout_file", "").value or find_build_py(),
                                               colliders=colliders), version=PLAN_VERSION)

        # single-publisher guard: give discovery time to see the other nodes first
        time.sleep(guard_wait)
        others = self._conflicts()
        if others:
            raise GuardError(f"{self.topic} already has publisher(s) {others}; stop them first "
                             "(keyboard_teleop / pick_place_demo). Refusing to start.")
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, TOPIC, qos)
        self.buf = CommandBuffer()
        self.blocked = False
        self._busy: set[str] = set()
        self._busy_lock = threading.Lock()
        self.create_timer(1.0 / self.rate, self._publish)
        self.create_timer(2.0, self._recheck_guard)
        self.server = ActionServer(self, MoveArm, action_name, execute_callback=self._execute,
                                   goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
                                   callback_group=self.cbg)
        self.get_logger().info(f"MoveArm on {action_name}; named poses: {sorted(self.named)}")

    # ---- /joint_command ------------------------------------------------------------------
    def _conflicts(self):
        infos = [(i.node_namespace, i.node_name) for i in self.get_publishers_info_by_topic(self.topic)]
        return publisher_conflicts(infos, own=(self.get_namespace(), self.get_name()))

    def _recheck_guard(self):
        others = self._conflicts()
        if others and not self.blocked:
            self.get_logger().error(f"another {self.topic} publisher appeared: {others}; "
                                    "stopping output and aborting goals")
        self.blocked = bool(others)

    def _publish(self):
        if self.blocked or self.buf.empty():
            return
        items = self.buf.snapshot()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [k for k, _ in items]
        msg.position = [v for _, v in items]
        self.pub.publish(msg)

    def _init_buffer(self) -> bool:
        """Seed the command buffer from /joint_states (both arms + grippers + waist), once."""
        meas = self.measured()
        if not meas:
            return False
        wj = self.waist_cfg.joint
        if not self.buf.has(wj) and wj in meas:
            self.buf.set({wj: meas[wj]})
        for arm in ARMS:
            names = [n for n in arm_joint_names(meas, arm) if not self.buf.has(n)]
            self.buf.set({n: meas[n] for n in names})
            g = GRIPPERS[arm]
            if not self.buf.has(g) and g in meas:
                self.buf.set({g: meas[g]})
        return True

    # ---- action --------------------------------------------------------------------------
    def _on_goal(self, goal):
        if goal.arm not in ARMS:
            self.get_logger().warn(f"rejecting goal: arm {goal.arm!r}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, gh):
        goal = gh.request
        arm = goal.arm
        with self._busy_lock:
            if arm in self._busy or "waist" in self._busy:
                gh.abort()
                why = "the waist is turning" if "waist" in self._busy else f"{arm} arm is already moving"
                return self._result(MoveArm.Result.BUSY, why, arm)
            self._busy.add(arm)
        self._waist_info = {}
        try:
            code, msg, clear, quat = self._run(gh, goal)
        except Exception as e:  # noqa: BLE001 - report, never leave the goal hanging
            code, msg, clear, quat = MoveArm.Result.INVALID_GOAL, f"error: {e}", -1.0, None
        finally:
            with self._busy_lock:
                self._busy.discard(arm)
                self._busy.discard("waist")
        if code == MoveArm.Result.OK:
            gh.succeed()
        elif code == MoveArm.Result.CANCELED and gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        if code != MoveArm.Result.OK:
            self.get_logger().warn(f"[{arm}] {msg}")
        return self._result(code, msg, arm, clear, quat)

    def _result(self, code, msg, arm, clear=-1.0, quat=None):
        r = MoveArm.Result()
        r.error_code, r.message, r.head_clearance = int(code), msg, float(clear)
        (r.orientation_used.x, r.orientation_used.y, r.orientation_used.z,
         r.orientation_used.w) = quat if quat is not None else (0.0, 0.0, 0.0, 0.0)
        names = arm_joint_names(dict(self.buf.snapshot()), arm) if arm in ARMS else []
        r.joint_names, r.positions = names, self.buf.get(names)
        wj = self.waist_cfg.joint
        r.waist_yaw = float(self.buf.get([wj])[0]) if self.buf.has(wj) else 0.0
        info = getattr(self, "_waist_info", {}) or {}
        r.base_turn_needed, r.base_turn = bool(info.get("needs_base", False)), float(info.get("base_turn", 0.0))
        return r

    def _feedback(self, gh, phase, progress):
        fb = MoveArm.Feedback()
        fb.phase, fb.progress = phase, float(progress)
        gh.publish_feedback(fb)

    def _stop_flag(self, gh):
        return lambda: gh.is_cancel_requested or self.blocked or not rclpy.ok()

    def _run(self, gh, goal):
        R = MoveArm.Result
        arm = goal.arm
        if self.blocked:
            return R.INVALID_GOAL, f"another {TOPIC} publisher is active", -1.0, None
        if not self._init_buffer():
            return R.NO_JOINT_STATES, "no /joint_states yet", -1.0, None
        names = arm_joint_names(dict(self.buf.snapshot()), arm)
        current = dict(zip(names, self.buf.get(names)))
        wj = self.waist_cfg.joint
        if self.buf.has(wj):
            current[wj] = self.buf.get([wj])[0]
        clear, quat = -1.0, None
        if not goal.gripper_only:
            self._feedback(gh, "planning", 0.0)
            chain, clear, quat, err = self._plan(goal, arm, names, current)
            if err:
                return err[0], err[1], clear, None
            moves_waist = any(abs(c.get(wj, current.get(wj, 0.0)) - current.get(wj, 0.0)) > 1e-4 for c in chain)
            if moves_waist:
                with self._busy_lock:
                    if self._busy - {arm}:
                        return R.BUSY, "the plan turns the waist but the other arm is moving", clear, None
                    self._busy.add("waist")
            keys = [n for n in names + [wj] if n in current and all(n in c for c in chain)]
            vel = goal.max_joint_vel if goal.max_joint_vel > 0 else self.max_vel
            durs = segment_durations([[c[n] for n in keys] for c in [current] + chain],
                                     goal.duration, vel, self.min_t)
            total, done = sum(durs), 0.0
            for target, d in zip(chain, durs):
                ok = run_segment(self.buf, target, d, self.rate, cancelled=self._stop_flag(gh),
                                 on_step=lambda i, n, d=d, base=done: self._feedback(
                                     gh, "moving", (base + d * i / n) / total) if i % 10 == 0 or i == n else None)
                if not ok:
                    return R.CANCELED, "canceled (or another publisher appeared)", clear, quat
                done += d
        if goal.gripper != MoveArm.Goal.GRIPPER_KEEP:
            self._feedback(gh, "gripper", 1.0)
            val = gripper_value("open" if goal.gripper == MoveArm.Goal.GRIPPER_OPEN else "close")
            if not run_segment(self.buf, {GRIPPERS[arm]: val}, self.gripper_time, self.rate,
                               cancelled=self._stop_flag(gh)):
                return R.CANCELED, "canceled during gripper motion", clear, quat
        return R.OK, "ok", clear, quat

    def _clearance(self, arm):
        """Head clearance along a path; the head is re-evaluated per sample (it turns with the waist).
        Local FK from the URDF when loaded (same model as move_group), else one /compute_fk per sample."""
        links = CHECK_LINKS[arm] + [HEAD_LINK]
        fx = getattr(self, "fixtures", None)
        fxm = getattr(self, "fixture_margin", 0.04)
        if getattr(self, "kin", None) is not None:
            fingers = FINGER_LINKS[arm] if fx else []

            def fk_local(joints):
                tf = self.fk_local(joints, links + fingers)
                h = tf[HEAD_LINK]
                heads = [tuple(h[:3, 3]), tuple(h[:3, :3] @ HEAD_CAM_IN_LINK3 + h[:3, 3])]
                return ([tuple(tf[n][:3, 3]) for n in CHECK_LINKS[arm]], heads,
                        [tuple(tf[n][:3, 3]) for n in fingers])
            return lambda path: path_clearance_fk(path, fk_local, fixtures=fx, fixture_margin=fxm)
        if self.fk_poses({}, [HEAD_LINK]) is None:
            return None

        def fk(joints):
            poses = self.fk_poses(joints, links)
            if poses is None:
                return None
            h = poses[HEAD_LINK]
            heads = head_points((h.position.x, h.position.y, h.position.z),
                                (h.orientation.x, h.orientation.y, h.orientation.z, h.orientation.w),
                                HEAD_CAM_IN_LINK3)
            return [(p.position.x, p.position.y, p.position.z) for n, p in poses.items() if n != HEAD_LINK], heads
        return lambda path: path_clearance_fk(path, fk, fixtures=fx, fixture_margin=fxm)

    def _goal_waist_mode(self, goal):
        mode = goal.waist_mode or self.waist_mode
        if mode not in WAIST_MODES:
            raise ValueError(f"unknown waist_mode {mode!r} (use one of {WAIST_MODES})")
        return mode

    def _goal_dict(self, goal):
        def pose(p):
            return [p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y,
                    p.orientation.z, p.orientation.w]
        return {"named_pose": goal.named_pose, "frame": goal.frame, "pose": pose(goal.pose),
                "via": [pose(v) for v in goal.via], "lookahead": [pose(v) for v in goal.lookahead],
                "auto_orientation": bool(goal.auto_orientation),
                "waist_mode": goal.waist_mode, "waist_yaw": float(goal.waist_yaw)}

    def _plan(self, goal, arm, names, current):
        """-> (chain of joint dicts, head clearance, end-link quat, error or None); cached (see module doc)."""
        params = {"head": self.min_clear, "waist": [self.waist_mode, self.waist_cfg.cone_deg,
                  self.waist_cfg.max_turn_deg, list(self.waist_cfg.pivot)],
                  "twist": [self.twist.j1_max_dev_deg, self.twist.jump_max], "named": repr(sorted(self.named.items())),
                  "fallback": [self.waist_fallback_step, self.waist_fallback_orients],
                  "fixtures": [bool(getattr(self, "fixtures", None)), getattr(self, "fixture_margin", 0.04)]}
        key = goal_key(arm, self._goal_dict(goal), current, params)
        hit = self.plan_cache.get(key)
        if hit is not None:
            self._waist_info = hit["waist_info"]
            self.get_logger().info(f"[{arm}] plan cache hit")
            if hit.get("error"):        # deterministic planner: the same goal fails the same way
                return [], hit["clear"], None, tuple(hit["error"])
            return hit["chain"], hit["clear"], (tuple(hit["quat"]) if hit["quat"] else None), None
        chain, clear, quat, err = self._plan_uncached(goal, arm, names, current)
        info = dict(getattr(self, "_waist_info", {}) or {})
        if err is None:
            self.plan_cache.put(key, {"chain": chain, "clear": clear, "quat": list(quat) if quat else None,
                                      "waist_info": info})
            self.plan_cache.save()
        elif str(err[1]).startswith(("planning failed", "path to")) and "budget" not in str(err[1]):
            # search exhausted (not a missing service / budget stop): remember the failure too
            self.plan_cache.put(key, {"error": [int(err[0]), err[1]], "clear": clear, "waist_info": info})
            self.plan_cache.save()
        return chain, clear, quat, err

    def _plan_uncached(self, goal, arm, names, current):
        R = MoveArm.Result
        clearance = self._clearance(arm)
        if clearance is None:
            return [], -1.0, None, (R.FK_FAILED, "/compute_fk failed (is move_group running?)")
        fixed, prefer, tool_frame = None, None, goal.frame in ("", MoveArm.Goal.FRAME_CELL_TOOL)
        if goal.frame not in ("", MoveArm.Goal.FRAME_CELL_TOOL, MoveArm.Goal.FRAME_BASE_END_LINK):
            return [], -1.0, None, (R.INVALID_GOAL, f"unknown frame {goal.frame!r}")
        if goal.named_pose:
            try:
                t = resolve(self.named, goal.named_pose, arm)
            except KeyError as e:
                return [], -1.0, None, (R.UNKNOWN_NAMED_POSE, str(e))
            if t.kind == "joints":
                target = home_joints(names, t.joints_deg)
                wj = self.waist_cfg.joint
                mode = self._goal_waist_mode(goal)
                if wj in current:
                    w = current[wj]
                    if mode == WAIST_ABSOLUTE:
                        w = clamp_waist(goal.waist_yaw, self.waist_cfg)
                    elif mode == WAIST_FIRST and t.waist_deg is not None:
                        w = clamp_waist(math.radians(t.waist_deg), self.waist_cfg)
                    target[wj] = w
                c = clearance([current, target])
                if c < self.min_clear:
                    return [], c, None, (R.HEAD_CLEARANCE, f"path to {goal.named_pose} passes {c:.2f} m from the head")
                return [target], c, None, None
            points, ahead, tool_frame = [t.xyz], [], True
            if t.yaw_deg is not None:
                fixed = down_quat(math.radians(t.yaw_deg), math.radians(t.tilt_deg))
        else:
            points = [_xyz(p) for p in goal.via] + [_xyz(goal.pose)]
            ahead = [_xyz(p) for p in goal.lookahead]
            o = goal.pose.orientation
            q = (o.x, o.y, o.z, o.w)
            fixed = None if goal.auto_orientation or quat_is_unset(q) else q
        if not self.ik.wait_for_service(timeout_sec=2.0):
            return [], -1.0, None, (R.IK_FAILED, "/compute_ik not available (is move_group running?)")
        mode = self._goal_waist_mode(goal)
        wj = self.waist_cfg.joint
        if wj not in current:
            mode = WAIST_FIXED      # no waist in /joint_states: plan with the arm only
        res = plan_chain_waist(points, lambda pos, q, s: self.solve(arm, pos, q, seed=s, attempts=self.ik_attempts),
                               clearance, seed=current, joint_names=names, home=home_joints(names, HOME_DEG),
                               waist_now=current.get(wj, 0.0), mode=mode, waist_yaw=goal.waist_yaw,
                               cfg=self.waist_cfg, limits=self.twist, prefer=prefer, fixed=fixed, start=current,
                               min_clearance=self.min_clear, tool_frame=tool_frame,
                               log=lambda m: self.get_logger().info(f"[{arm}] {m}"),
                               budget=PlanBudget(self.max_solves or None, self.max_seconds or None),
                               lookahead=ahead, fallback_step_deg=self.waist_fallback_step,
                               fallback_orients=self.waist_fallback_orients)
        self._waist_info = {"needs_base": res.needs_base, "base_turn": res.base_turn}
        if not res.ok:
            code = R.HEAD_CLEARANCE if res.reason == "head_clearance" else R.IK_FAILED
            return [], res.clearance, None, (code, f"planning failed: {res.reason}"
                                             + (f" ({res.twist_rejected} twisted IK solutions rejected)"
                                                if res.twist_rejected else ""))
        chain = res.solutions if wj in current else [{n: c[n] for n in names} for c in res.solutions]
        self.get_logger().info(
            f"[{arm}] plan ok ({mode}): waist {[round(math.degrees(w)) for w in res.waist_yaws]} deg, "
            f"orientation {res.label}, head clearance {res.clearance:.2f} m"
            + (f"; BASE TURN NEEDED {math.degrees(res.base_turn):.0f} deg (not driven)" if res.needs_base else ""))
        return chain, res.clearance, res.quat, None


def main():
    rclpy.init()
    try:
        node = MotionServer()
    except GuardError as e:
        print(f"[g2cell_motion_server] {e}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(2)
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
