"""cell_manager: CellStatus publisher + line/spawn/episode services (skeleton for Stages 3 and 5).

    ros2 run g2cell_tasks cell_manager --ros-args -p use_sim_time:=true
    ros2 service call /g2cell/episode_control g2cell_interfaces/srv/EpisodeControl "{command: 0}"

Real now: episode start/stop/mark (optionally recorded with g2cell_recorder via
g2cell_teleop's RecorderController, record_episodes:=true), pick counts from /g2cell/pick_results.
Not implemented (answer STATUS_NOT_IMPLEMENTED, never a fake success):
  /g2cell/set_line_state  -> TODO(Stage 3) conveyor indexing (cell_state.CellState.line_command)
  /g2cell/spawn_bags      -> TODO(Stage 5) bag spawning (cell_state.CellState.spawn_bags)
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from g2cell_interfaces.msg import CellStatus, PickPlaceResult
from g2cell_interfaces.srv import EpisodeControl, SetLineState, SpawnBags
from g2cell_tasks.cell_state import MARK_FAILURE, MARK_SUCCESS, OK, START, STOP, CellState


class CellManager(Node):
    def __init__(self):
        super().__init__("g2cell_cell_manager")
        p = self.declare_parameter
        self.state = CellState(crate_fill_target=p("crate_fill_target", 10).value,
                               bags_on_table=p("bags_on_table", 3).value,
                               crate_at_pack_position=p("crate_at_pack_position", True).value)
        self._lock = threading.Lock()
        self.recorder = None
        if p("record_episodes", False).value:
            from g2cell_teleop.recorder_control import RecorderController  # noqa: PLC0415
            self.recorder = RecorderController(scene=p("scene", "scene_g2cell").value, seed=p("seed", -1).value,
                                               episodes_dir=p("episodes_dir", "/g2cell_output/episodes").value,
                                               name_fn=lambda: self.state.episode_id,
                                               log=lambda m: self.get_logger().info(m))
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(CellStatus, "/g2cell/cell_status", qos)
        self.create_subscription(PickPlaceResult, "/g2cell/pick_results", self._on_result, 50)
        self.create_service(SetLineState, "/g2cell/set_line_state", self._line)
        self.create_service(SpawnBags, "/g2cell/spawn_bags", self._spawn)
        self.create_service(EpisodeControl, "/g2cell/episode_control", self._episode)
        self.create_timer(1.0 / p("status_rate_hz", 2.0).value, self._publish)
        self.get_logger().info("cell_manager up: /g2cell/cell_status, set_line_state (Stage 3 TODO), "
                               "spawn_bags (Stage 5 TODO), episode_control")

    def _publish(self):
        s = self.state
        with self._lock:
            msg = CellStatus(episode_id=s.episode_id, episode_active=s.episode_active, line_running=s.line_running,
                             crate_at_pack_position=s.crate_at_pack_position, crate_fill_count=s.crate_fill_count,
                             crate_fill_target=s.crate_fill_target, bags_on_table=s.bags_on_table,
                             picks_done=s.picks_done, not_implemented=s.not_implemented())
        msg.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def _on_result(self, r: PickPlaceResult):
        with self._lock:
            self.state.on_pick_result(r.outcome_name, r.verified, r.intended_target)
            full = self.state.crate_full
        if full:
            self.get_logger().info("pack crate reached its target count: release it (Stage 3 TODO)")

    def _line(self, req, resp):
        with self._lock:
            resp.status, resp.message = self.state.line_command(req.command)
        resp.success = resp.status == OK
        return resp

    def _spawn(self, req, resp):
        with self._lock:
            resp.status, resp.message = self.state.spawn_bags(req.count, req.seed)
        resp.success = resp.status == OK
        return resp

    def _episode(self, req, resp):
        with self._lock:
            resp.status, resp.message, resp.episode_id = self.state.episode(req.command, req.episode_id)
            mark = self.state.mark
        resp.success = resp.status == OK
        if resp.success and self.recorder is not None:
            try:
                if req.command == START:
                    self.recorder.start()
                elif req.command == STOP:
                    self.recorder.stop({"success": "true", "failure": "false"}.get(mark))
                elif req.command in (MARK_SUCCESS, MARK_FAILURE):
                    self.recorder.mark("true" if req.command == MARK_SUCCESS else "false")
            except Exception as e:  # noqa: BLE001
                resp.message += f"; recorder: {e}"
        self._publish()
        return resp


def main():
    rclpy.init()
    node = CellManager()
    ex = MultiThreadedExecutor()
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
