"""MoveIt /compute_ik + /compute_fk + /joint_states mixin (moved here from g2cell_demo.ik_client).

Mix into a Node and call init_ik() (and init_fk() for FK/head-clearance helpers).
init_local_ik(urdf) makes solve() deterministic and fast: the numpy DLS solver (g2cell_motion.kinematics,
same URDF as move_group) is the primary solver and MoveIt /compute_ik, seeded with the local answer,
confirms it (KDL converges at once from an exact seed, so the result stays MoveIt's). Only if the local
solver finds nothing does solve() fall back to /compute_ik with fixed-seed jitter, early exit on the first
success and at most `ik_fallback_attempts` calls (each failing call costs KDL's full 0.1 s timeout).
"""
from __future__ import annotations

import threading
import time

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from g2cell_motion.clearance import head_points
from g2cell_motion.planner import ik_seed_sequence
from g2cell_motion.robot import ARM_MATCH, ARMS, GROUPS, HEAD_CAM_IN_LINK3, HEAD_LINK, IK_LINKS


def _wait_future(client, req, timeout):
    done = threading.Event()
    fut = client.call_async(req)
    fut.add_done_callback(lambda _: done.set())
    return fut.result() if done.wait(timeout) else None


class IkHelper:
    """Mix into a Node: subscribes /joint_states, calls /compute_ik (and /compute_fk) with seeds."""

    def init_ik(self: Node, base_frame: str = "base_link", callback_group=None):
        self.base_frame = base_frame
        self._js_lock = threading.Lock()
        self._js: JointState | None = None
        # the engine publishes /joint_states best-effort; a reliable subscriber would get nothing
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data,
                                 callback_group=callback_group)
        self.ik = self.create_client(GetPositionIK, "/compute_ik", callback_group=callback_group)

    def init_fk(self: Node, callback_group=None):
        self.fk = self.create_client(GetPositionFK, "/compute_fk", callback_group=callback_group)
        self._head_pts = None

    def init_local_ik(self: Node, urdf_path: str, confirm: bool = True, fallback_attempts: int = 2):
        """Primary local IK from the staged robot URDF (see module doc). Returns False if unavailable."""
        from g2cell_motion.kinematics import ArmIk, UrdfKinematics
        from g2cell_motion.robot import ARM_COUPLED, ARM_COUPLED_MARGIN
        self.ik_stats = {"local_ok": 0, "local_fail": 0, "local_s": 0.0, "confirm_ok": 0, "confirm_fail": 0,
                         "moveit_calls": 0, "moveit_s": 0.0, "fallback_ok": 0}
        self.ik_confirm, self.ik_fallback_attempts = confirm, fallback_attempts
        try:
            self.kin = UrdfKinematics(urdf_path)
            self.local_ik = {}
            for a in ARMS:
                names = [f"idx{2 if a == 'left' else 6}{k}_arm_{a[0]}_joint{k}" for k in range(1, 8)]
                self.local_ik[a] = ArmIk(self.kin, names, IK_LINKS[a], coupled=[
                    (names[5:7], ARM_COUPLED["A"], ARM_COUPLED["b"])], coupled_margin=ARM_COUPLED_MARGIN)
        except (OSError, KeyError, ValueError) as e:
            self.get_logger().warn(f"local IK unavailable ({e}); using /compute_ik only")
            self.local_ik = None
            return False
        return True

    def _on_js(self, msg):
        with self._js_lock:
            self._js = msg

    def joint_state(self) -> JointState | None:
        with self._js_lock:
            return self._js

    def measured(self) -> dict:
        js = self.joint_state()
        return dict(zip(js.name, js.position)) if js else {}

    def arm_joint_names(self, arm: str) -> list[str]:
        return sorted(n for n in self.measured() if ARM_MATCH[arm] in n)

    def _call(self, req, timeout=2.0):
        return _wait_future(self.ik, req, timeout)

    def _ik_request(self, arm: str, pos, quat, state: dict, timeout: float):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = GROUPS[arm]
        r.ik_link_name = IK_LINKS[arm]
        r.avoid_collisions = False
        r.timeout.nanosec = int(timeout * 1e9)
        r.robot_state.joint_state = JointState(name=list(state.keys()), position=[float(v) for v in state.values()])
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(c) for c in pos)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (float(c) for c in quat)
        r.pose_stamped = ps
        t0 = time.perf_counter()
        res = self._call(req)
        st = getattr(self, "ik_stats", None)
        if st is not None:
            st["moveit_calls"] += 1
            st["moveit_s"] += time.perf_counter() - t0
        if res is None or res.error_code.val != 1:
            return None
        return dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))

    def solve(self, arm: str, pos, quat, seed: dict | None = None, attempts: int = 8,
              timeout: float = 0.1, jitter: float = 0.4):
        """IK for the arm's end link at pos/quat (base_link). Returns {joint: angle} or None.
        Deterministic: local IK + MoveIt confirmation when init_local_ik() was called (see module doc);
        otherwise / as fallback /compute_ik from the seed, then fixed-seed jittered seeds, first success wins."""
        js = self.joint_state()
        if js is None:
            return None
        base = dict(zip(js.name, js.position))
        if seed:
            base.update(seed)
        names = self.arm_joint_names(arm)
        local = getattr(self, "local_ik", None)
        if local:
            st = self.ik_stats
            t0 = time.perf_counter()
            sol = local[arm].solve(pos, quat, base, [{n: base[n] for n in names}])
            st["local_s"] += time.perf_counter() - t0
            if sol is not None:
                st["local_ok"] += 1
                if not self.ik_confirm:
                    return sol
                m = self._ik_request(arm, pos, quat, {**base, **sol}, timeout)
                if m is not None and all(abs(m[n] - sol[n]) < 1e-3 for n in names):
                    st["confirm_ok"] += 1
                    return {n: m[n] for n in names}
                st["confirm_fail"] += 1
            else:
                st["local_fail"] += 1
            attempts = min(attempts, self.ik_fallback_attempts)
        for s in ik_seed_sequence(base, names, attempts, jitter):
            m = self._ik_request(arm, pos, quat, s, timeout)
            if m is not None:
                if local:
                    self.ik_stats["fallback_ok"] += 1
                return {n: m[n] for n in names if n in m}   # early exit: first success
        return None

    def fk_local(self, joints: dict, links):
        """{link: 4x4} in base_link from the local URDF model (no service call), or None if not loaded.
        The measured state fills the joints not given."""
        kin = getattr(self, "kin", None)
        if kin is None:
            return None
        state = self.measured()
        state.update(joints)
        return kin.link_poses(state, links)

    # ---- FK ------------------------------------------------------------------------------
    def fk_poses(self, joints: dict, links, timeout: float = 2.0):
        """{link: geometry_msgs/Pose} in base_frame for the measured state overridden by `joints`."""
        js = self.joint_state()
        if js is None:
            return None
        state = dict(zip(js.name, js.position))
        state.update(joints)
        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = list(links)
        req.robot_state.joint_state = JointState(name=list(state), position=list(state.values()))
        res = _wait_future(self.fk, req, timeout)
        if res is None or res.error_code.val != 1:
            return None
        return {n: ps.pose for n, ps in zip(res.fk_link_names, res.pose_stamped)}

    def fk_points(self, joints: dict, links):
        poses = self.fk_poses(joints, links)
        if poses is None:
            return None
        return [(p.position.x, p.position.y, p.position.z) for p in poses.values()]

    def head_points(self):
        """[head_link3 origin, head camera] in base_link (cached: the head does not move)."""
        if self._head_pts is None:
            poses = self.fk_poses({}, [HEAD_LINK])
            if poses is None:
                return None
            p = poses[HEAD_LINK]
            self._head_pts = head_points((p.position.x, p.position.y, p.position.z),
                                         (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
                                         HEAD_CAM_IN_LINK3)
        return self._head_pts
