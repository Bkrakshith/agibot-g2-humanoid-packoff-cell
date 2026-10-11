"""ROS 2 backend for EpisodeRunner (runs inside the g2cell container, NOT in the g2cell tmux session,
because reset mode "restart" kills that session).

Per episode:
  reset            write <episode>/tasks.yaml (randomised bags); reset mode "teleport" moves the bags (and
                   restores the pack carton) through the object pose relay's /g2cell/object_poke and checks the
                   poses it reports on /tf_render (scene_io); "restart" writes a scene variant and restarts the sim
                   stack (stop_cmd, start_cmd), then waits for /joint_states.
                   Scripted mode: (re)starts motion_server (once per sim start) and pick_bag_server (every
                   episode, with that tasks.yaml), then homes both arms through MoveArm.
  start_recording  g2cell_recorder record_episode.sh via g2cell_teleop's RecorderController
  run_task         PickBag goal {bag_id, target, lane} on /g2cell/pick_bag; with the relay (teleport mode or
                   reset.score) the bag's resting place re-labels a completed motion (scene_io.score)
  stop_recording   SIGINT -> bag -> HDF5 -> mark_episode (true/false)
Only motion_server publishes /joint_command (scripted mode). Operator mode starts neither server: the
operator's keyboard_teleop is the single publisher.
"""
from __future__ import annotations

import json
import shlex
import threading
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from g2cell_datagen.procs import ManagedProcess, run_shell
from g2cell_datagen.randomize import EpisodeSpec, tasks_doc_for
from g2cell_datagen.recording import EpisodeRecorder
from g2cell_datagen.runner import RecordingResult, TaskResult
from g2cell_datagen.scene_io import (BAGS_ROOT, CARTON, TOTES, check_reset, placement_of, score,
                                     teleport_requests)
from g2cell_datagen.scene_variant import write_variant
from g2cell_interfaces.action import MoveArm, PickBag


class RosBackend:
    def __init__(self, *, batch: str, batch_dir, tasks_doc: dict, reset_cfg: dict, mode: str = "scripted",
                 reset_mode: str = "none", profile: str = "collect", manage_stack: bool = True,
                 home_before: bool = True, record: bool = True, scene_name: str = "scene_g2cell",
                 task_timeout_s: float = 600.0, node_ros_args=(), joint_states_topic: str = "/joint_states",
                 log=print):
        self.batch, self.dir = batch, Path(batch_dir)
        self.tasks_doc, self.reset_cfg = tasks_doc, dict(reset_cfg)
        self.mode, self.reset_mode, self.profile = mode, reset_mode, profile
        self.manage_stack = manage_stack and mode == "scripted"
        self.home_before = home_before and mode == "scripted"
        self.task_timeout_s, self.node_ros_args, self.log = task_timeout_s, list(node_ros_args), log
        self.logs = self.dir / "_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.recorder = EpisodeRecorder(self.dir, scene=scene_name, log=log) if record else None
        self.motion: ManagedProcess | None = None
        self.pick: ManagedProcess | None = None

        rclpy.init()
        self.node = rclpy.create_node("g2cell_datagen")
        cbg = ReentrantCallbackGroup()
        self._js_t = 0.0
        self.node.create_subscription(JointState, joint_states_topic, self._on_js, qos_profile_sensor_data,
                                      callback_group=cbg)
        self._poses, self._poses_lock = {}, threading.Lock()
        self.node.create_subscription(TFMessage, "/tf_render", self._on_tf, qos_profile_sensor_data,
                                      callback_group=cbg)
        self.poke_pub = self.node.create_publisher(String, "/g2cell/object_poke", 50)
        self.score_on = bool(self.reset_cfg.get("score", True))
        self.pick_client = ActionClient(self.node, PickBag, "/g2cell/pick_bag", callback_group=cbg)
        self.move_client = ActionClient(self.node, MoveArm, "/g2cell/move_arm", callback_group=cbg)
        self.ex = MultiThreadedExecutor(num_threads=4)
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    # ---- helpers -----------------------------------------------------------------------------
    def _on_js(self, _msg):
        self._js_t = time.monotonic()

    def _on_tf(self, msg):
        with self._poses_lock:
            for t in msg.transforms:
                if t.child_frame_id.startswith("/World/"):
                    p, q = t.transform.translation, t.transform.rotation
                    self._poses[t.child_frame_id] = ((p.x, p.y, p.z), (q.w, q.x, q.y, q.z))

    def poses(self) -> dict:
        with self._poses_lock:
            return dict(self._poses)

    def _wait_fresh_poses(self, paths, wait_s: float):
        """Drop what we have and wait until every path was seen again (the relay re-sends all bodies on its
        2 s keyframe, movers at 25 Hz)."""
        with self._poses_lock:
            for p in paths:
                self._poses.pop(p, None)
        t_end = time.monotonic() + wait_s
        while time.monotonic() < t_end:
            if all(p in self._poses for p in paths):
                break
            time.sleep(0.1)
        return self.poses()

    def _teleport(self, spec) -> dict:
        reqs = teleport_requests(spec, {k: list(v) for k, v in (self.reset_cfg.get("restore") or {}).items()})
        t_end = time.monotonic() + 10.0
        while self.poke_pub.get_subscription_count() == 0:
            if time.monotonic() > t_end:
                raise RuntimeError("no /g2cell/object_poke subscriber: start the sim with OBJ_RELAY=1")
            time.sleep(0.2)
        for _ in range(2):            # twice: the bag must not keep a velocity picked up during the first
            for r in reqs:
                self.poke_pub.publish(String(data=json.dumps(r)))
            time.sleep(0.3)
        time.sleep(float(self.reset_cfg.get("settle_s", 1.0)))
        poses = self._wait_fresh_poses([r["path"] for r in reqs], 5.0)
        ok, worst = check_reset(poses, reqs)
        info = {"teleport_ok": ok, "teleport_worst": {k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in worst.items()}}
        if not ok:
            raise RuntimeError(f"teleport reset failed: {worst}")
        return info

    def _score(self, spec, res: TaskResult) -> TaskResult:
        bag = BAGS_ROOT + spec.bag_id
        time.sleep(float(self.reset_cfg.get("settle_s", 1.0)))
        poses = self._wait_fresh_poses([bag, CARTON, *TOTES.values()], 5.0)
        if bag not in poses:
            return res
        pl = placement_of(poses[bag], poses, spawn=(spec.x, spec.y))
        outcome, verified = score(res.outcome, spec.target, spec.lane, pl, res.grasp_attempts)
        res.placement = pl
        if verified:
            res.task_outcome, res.outcome, res.verified = res.outcome, outcome, True
        return res

    def _wait_joint_states(self, timeout: float):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if time.monotonic() - self._js_t < 0.5:
                return
            time.sleep(0.2)
        raise TimeoutError(f"no /joint_states for {timeout:.0f} s (is the sim running?)")

    def _ros_run(self, pkg, exe, *params, name=""):
        args = ["ros2", "run", pkg, exe, "--ros-args", "-p", "use_sim_time:=true", *params, *self.node_ros_args]
        return ManagedProcess(args, name=name or exe, log_path=self.logs / f"{name or exe}.log", log=self.log)

    def _send(self, client: ActionClient, goal, timeout: float):
        if not client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(f"action server {client._action_name} not available")
        ev, box = threading.Event(), {}

        def on_goal(fut):
            gh = fut.result()
            box["gh"] = gh
            if not gh.accepted:
                ev.set()
                return
            gh.get_result_async().add_done_callback(lambda f: (box.update(r=f.result().result), ev.set()))

        client.send_goal_async(goal).add_done_callback(on_goal)
        if not ev.wait(timeout):
            gh = box.get("gh")
            if gh is not None:
                gh.cancel_goal_async()
            raise TimeoutError(f"{client._action_name}: no result in {timeout:.0f} s")
        if "r" not in box:
            raise RuntimeError(f"{client._action_name}: goal rejected")
        return box["r"]

    # ---- Backend ------------------------------------------------------------------------------
    def reset(self, name: str, spec: EpisodeSpec) -> dict:
        ep = self.dir / name
        ep.mkdir(parents=True, exist_ok=True)
        tasks_file = ep / "tasks.yaml"
        tasks_file.write_text(yaml.safe_dump(tasks_doc_for(self.tasks_doc, spec), sort_keys=False))
        info = {"mode": self.reset_mode, "tasks_file": str(tasks_file)}
        rc = self.reset_cfg
        if self.reset_mode == "restart":
            usda, yml = write_variant(rc["base_scene_yaml"], rc["cell_usda"], Path(rc["variants_dir"]) / self.batch,
                                      name, spec.poses(), comment=f"{self.batch} {name} seed {spec.seed}")
            info.update(scene_yaml=str(yml), scene_usda=str(usda))
            self._stop_stack()
            run_shell(rc["stop_cmd"], timeout=60, log_path=self.logs / "sim_stop.log")
            time.sleep(2.0)
            cmd = rc["start_cmd"].format(profile=shlex.quote(self.profile), scene_yaml=shlex.quote(str(yml)))
            t = time.monotonic()
            code = run_shell(cmd, timeout=float(rc.get("ready_timeout_s", 900)) + 120,
                             log_path=self.logs / "sim_start.log")
            if code != 0:
                raise RuntimeError(f"sim start failed (rc {code}); see {self.logs / 'sim_start.log'}")
            info["sim_start_s"] = round(time.monotonic() - t, 1)
        elif self.reset_mode not in ("none", "teleport"):
            raise ValueError(f"unknown reset mode {self.reset_mode!r}")
        self._wait_joint_states(60.0 if self.reset_mode == "none" else 300.0)
        if self.manage_stack:
            if self.motion is None or not self.motion.running:
                self.motion = self._ros_run("g2cell_motion", "motion_server").start()
            if self.pick is not None:
                self.pick.stop()
                time.sleep(1.0)       # let the old action server leave the graph
            self.pick = self._ros_run("g2cell_tasks", "pick_bag_server", "-p", f"tasks_file:={tasks_file}",
                                      name=f"pick_bag_server_{name}").start()
            # wait until this episode's server is up (the previous one is stopped above)
            if not self.pick_client.wait_for_server(timeout_sec=60.0):
                raise RuntimeError("pick_bag_server did not come up; see " + str(self.logs))
        if self.home_before:
            for arm in ("left", "right"):
                r = self._send(self.move_client, MoveArm.Goal(arm=arm, named_pose="home"), timeout=120.0)
                if r.error_code != MoveArm.Result.OK:
                    raise RuntimeError(f"home {arm} failed: {r.message} (code {r.error_code})")
            info["homed"] = True
        if self.reset_mode == "teleport":       # after homing: the arms are clear of the table and carton
            info.update(self._teleport(spec))
        return info

    def start_recording(self, name: str, spec: EpisodeSpec) -> None:
        if self.recorder is not None:
            self.recorder.start(name, spec.seed)

    def run_task(self, name: str, spec: EpisodeSpec) -> TaskResult:
        goal = PickBag.Goal(bag_id=spec.bag_id, target=spec.target, lane=spec.lane)
        t = time.monotonic()
        try:
            r = self._send(self.pick_client, goal, self.task_timeout_s).result
        except TimeoutError as e:
            return TaskResult("aborted", message=str(e), duration=time.monotonic() - t)
        res = TaskResult(r.outcome_name or str(r.outcome), message=r.message, arm=r.arm, verified=bool(r.verified),
                         grasp_attempts=int(r.grasp_attempts), failed_stage=r.failed_stage,
                         duration=float(r.duration))
        if self.score_on and self.reset_mode in ("teleport", "restart"):   # the bag started where tasks.yaml says
            res = self._score(spec, res)
        return res

    def stop_recording(self, name: str, success: str) -> RecordingResult:
        if self.recorder is None:
            return RecordingResult(True, error="recording disabled (--no-record)")
        return self.recorder.stop(name, success)

    def _stop_stack(self):
        for p in (self.pick, self.motion):
            if p is not None:
                p.stop()
        self.pick = self.motion = None

    def close(self):
        self._stop_stack()
        try:
            self.ex.shutdown()
            self.node.destroy_node()
        finally:
            rclpy.try_shutdown()
