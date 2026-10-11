"""ROS 2 port of the Controller: one rclpy node spun by a MultiThreadedExecutor in a daemon thread.

    /joint_states            sensor_msgs/JointState, subscribed best effort (qos_profile_sensor_data)
    /joint_command           sensor_msgs/JointState, best effort, 50 Hz stream of the current targets,
                             only after the first motion command, only while no other node publishes there
    /compute_ik, /compute_fk MoveIt (move_group, IK only), groups simple_arm_l / simple_arm_r
    /tf, /tf_static          tf2 Buffer + TransformListener
    camera topics            sensor_msgs/Image, subscribed when a Camera asks for them

Genie Sim stamps its messages with wall time, so the node runs with use_sim_time false and
stamps /joint_command with wall time too.
"""
from __future__ import annotations

import collections
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

from agibot_gdk import _geometry as g
from agibot_gdk._controller import END_LINKS, GuardError
from agibot_gdk._convert import DEFAULT_HEAD_CAMERA, JointSample, RawImage, head_camera_from_scene

TOPIC = "/joint_command"
IK_GROUPS = {"left": "simple_arm_l", "right": "simple_arm_r"}
DEFAULT_SCENE = "/g2cell_ws/config/scenes/scene_g2cell.yaml"


def load_head_camera():
    """Head camera of the running scene: GDK_SIM_SCENE, else the cell scene, else built-in defaults."""
    path = Path(os.environ.get("GDK_SIM_SCENE", DEFAULT_SCENE))
    try:
        return head_camera_from_scene(yaml.safe_load(path.read_text()) or {})
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return DEFAULT_HEAD_CAMERA


def _stamp_ns(stamp, fallback: int) -> int:
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns > 0 else fallback


def _matrix(t) -> np.ndarray:
    return g.matrix((t.translation.x, t.translation.y, t.translation.z),
                    (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))


def _wait(client, request, timeout: float):
    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        client.remove_pending_request(future)
        return None
    return future.result()


class SimBackend:
    def __init__(self) -> None:
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.rate = float(os.environ.get("GDK_SIM_RATE_HZ", 50.0))
        self.guard_wait = float(os.environ.get("GDK_SIM_GUARD_WAIT", 1.5))
        self.node = rclpy.create_node(f"agibot_gdk_sim_{os.getpid()}",
                                      parameter_overrides=[Parameter("use_sim_time", value=False)])
        self.log = self.node.get_logger()
        self.cbg = ReentrantCallbackGroup()
        self._created = time.monotonic()
        self._lock = threading.Lock()          # sensor caches
        self._stream_lock = threading.Lock()   # publisher start (may sleep for discovery)
        self._joints: JointSample | None = None
        self._images: dict[str, RawImage] = {}
        self._arrivals: dict[str, collections.deque] = {}
        self._image_subs = {}
        self._pub = None
        self._buffer = None
        self._blocked: list[str] = []
        self._warned: set[str] = set()

        self.node.create_subscription(JointState, "/joint_states", self._on_joints, qos_profile_sensor_data,
                                      callback_group=self.cbg)
        self.tf_buffer = Buffer(node=self.node)
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)
        self.ik_client = self.node.create_client(GetPositionIK, "/compute_ik", callback_group=self.cbg)
        self.fk_client = self.node.create_client(GetPositionFK, "/compute_fk", callback_group=self.cbg)

        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name="agibot_gdk_sim", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception:  # noqa: BLE001 - shutdown races end the spin
            pass

    # ---- clock ------------------------------------------------------------------------------------
    @staticmethod
    def now() -> float:
        return time.monotonic()

    @staticmethod
    def sleep(dt: float) -> None:
        if dt > 0:
            time.sleep(dt)

    @staticmethod
    def wall_ns() -> int:
        return time.time_ns()

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.log.warning(text)

    # ---- joints ----------------------------------------------------------------------------------
    def _on_joints(self, msg: JointState) -> None:
        sample = JointSample(list(msg.name), list(msg.position), list(msg.velocity), list(msg.effort),
                             _stamp_ns(msg.header.stamp, time.time_ns()))
        with self._lock:
            self._joints = sample

    def joint_sample(self):
        with self._lock:
            return self._joints

    # ---- /joint_command stream + single-publisher guard ------------------------------------------
    def _others(self) -> list[str]:
        me = (self.node.get_namespace().rstrip("/") + "/" + self.node.get_name())
        found = set()
        for info in self.node.get_publishers_info_by_topic(TOPIC):
            name = info.node_namespace.rstrip("/") + "/" + info.node_name
            if name != me:
                found.add(name)
        return sorted(found)

    def start_stream(self, buffer) -> None:
        with self._stream_lock:
            if self._pub is not None:
                if self._blocked:
                    raise GuardError(f"{TOPIC} has another publisher {self._blocked}; streaming stopped")
                return
            remaining = self.guard_wait - (time.monotonic() - self._created)
            if remaining > 0:
                time.sleep(remaining)  # let discovery see the other nodes first
            others = self._others()
            if others:
                raise GuardError(f"{TOPIC} already has publisher(s) {others}")
            qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self._buffer = buffer
            self._pub = self.node.create_publisher(JointState, TOPIC, qos)
            self.node.create_timer(1.0 / self.rate, self._publish, callback_group=self.cbg)
            self.node.create_timer(2.0, self._recheck, callback_group=self.cbg)
            self.log.info(f"streaming joint targets on {TOPIC} at {self.rate:.0f} Hz")

    def _publish(self) -> None:
        if self._blocked or self._pub is None:
            return
        targets = self._buffer.snapshot()
        if not targets:
            return
        msg = JointState()
        msg.header.stamp = Time(nanoseconds=time.time_ns()).to_msg()  # wall time, like the engine
        msg.name = list(targets)
        msg.position = list(targets.values())
        self._pub.publish(msg)

    def _recheck(self) -> None:
        others = self._others()
        if others and not self._blocked:
            self.log.error(f"another {TOPIC} publisher appeared: {others}; agibot_gdk stops streaming")
        self._blocked = others

    # ---- MoveIt ------------------------------------------------------------------------------------
    def _ready(self, client, name: str) -> bool:
        if client.service_is_ready() or client.wait_for_service(timeout_sec=1.0):
            return True
        self._warn_once(name, f"{name} not available: is move_group running (stage1_up.sh starts it)?")
        return False

    def ik(self, side: str, target: np.ndarray, seed: dict):
        if not self._ready(self.ik_client, "/compute_ik"):
            return None
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = IK_GROUPS[side]
        r.ik_link_name = END_LINKS[side]
        r.avoid_collisions = False
        r.timeout.nanosec = int(float(os.environ.get("GDK_SIM_IK_TIMEOUT", 0.05)) * 1e9)
        r.robot_state.joint_state = JointState(name=list(seed), position=[float(v) for v in seed.values()])
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in target[:3, 3])
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = g.quaternion(target)
        r.pose_stamped = ps
        res = _wait(self.ik_client, req, 0.5)
        if res is None or res.error_code.val != 1:
            return None
        return dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))

    def fk(self, joints: dict, links):
        if not self._ready(self.fk_client, "/compute_fk"):
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = list(links)
        req.robot_state.joint_state = JointState(name=list(joints), position=[float(v) for v in joints.values()])
        res = _wait(self.fk_client, req, 1.0)
        if res is None or res.error_code.val != 1:
            return None
        return {name: _matrix_from_pose(ps.pose) for name, ps in zip(res.fk_link_names, res.pose_stamped)}

    # ---- TF ----------------------------------------------------------------------------------------
    def lookup(self, parent: str, child: str):
        try:
            t = self.tf_buffer.lookup_transform(parent, child, Time())  # T_parent_child, latest
        except Exception:  # noqa: BLE001 - tf2 raises several exception types
            return None
        return _matrix(t.transform), _stamp_ns(t.header.stamp, time.time_ns())

    def frame_names(self) -> list[str]:
        try:
            tree = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        except yaml.YAMLError:
            return []
        names = set(tree)
        names.update(v.get("parent") for v in tree.values() if isinstance(v, dict) and v.get("parent"))
        return sorted(names)

    # ---- images ------------------------------------------------------------------------------------
    def open_image(self, topic: str) -> None:
        with self._lock:
            if topic in self._image_subs:
                return
            self._arrivals[topic] = collections.deque(maxlen=30)
            self._image_subs[topic] = self.node.create_subscription(
                Image, topic, lambda msg, t=topic: self._on_image(t, msg), qos_profile_sensor_data,
                callback_group=self.cbg)

    def _on_image(self, topic: str, msg: Image) -> None:
        raw = RawImage(msg.encoding, msg.width, msg.height, msg.step, bytes(msg.data),
                       _stamp_ns(msg.header.stamp, time.time_ns()))
        with self._lock:
            self._images[topic] = raw
            self._arrivals[topic].append(time.monotonic())

    def latest_image(self, topic: str):
        with self._lock:
            return self._images.get(topic)

    def image_rate(self, topic: str):
        with self._lock:
            times = list(self._arrivals.get(topic, ()))
        if len(times) < 3 or time.monotonic() - times[-1] > 2.0:
            return None
        return (len(times) - 1) / (times[-1] - times[0])

    # ---- shutdown ----------------------------------------------------------------------------------
    def shutdown(self) -> None:
        self._blocked = ["released"]  # stop publishing; the sim holds the last targets
        try:
            self.executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
        finally:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
        self._thread.join(timeout=2.0)


def _matrix_from_pose(pose) -> np.ndarray:
    return g.matrix((pose.position.x, pose.position.y, pose.position.z),
                    (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
