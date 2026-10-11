"""PickBag action server (/g2cell/pick_bag): one bag from the pick table to the pack crate or a
rack lane, via the MoveArm action of g2cell_motion's motion_server (the only /joint_command
publisher -- this node never publishes joint commands itself).

    ros2 run g2cell_tasks pick_bag_server --ros-args -p use_sim_time:=true
    ros2 action send_goal --feedback /g2cell/pick_bag g2cell_interfaces/action/PickBag "{bag_id: any}"

Task loop = g2cell_tasks.state_machine, stages = g2cell_tasks.runner (both pure, unit-tested). Arm choice = reachability table,
target = sort rule (bag label) unless the goal says otherwise. Each result is also published on
/g2cell/pick_results for cell_manager.
"""
from __future__ import annotations

import os
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from g2cell_interfaces.action import PickBag
from g2cell_interfaces.msg import PickPlaceResult
from g2cell_motion.client import MotionClient
from g2cell_motion.robot import GRIPPERS
from g2cell_tasks.cell_state import LEFT_TABLE
from g2cell_tasks.config import load_task_config
from g2cell_tasks.grasp import grasp_from_gripper
from g2cell_tasks.outcome import OUTCOME_CODES
from g2cell_tasks.reachability import select_arm
from g2cell_tasks.runner import MotionRunner, plan_task
from g2cell_tasks.state_machine import STAGE_INDEX, Hooks, PickPlaceMachine


class PickBagServer(Node):
    def __init__(self):
        super().__init__("g2cell_pick_bag_server")
        p = self.declare_parameter
        default_cfg = os.path.join(get_package_share_directory("g2cell_tasks"), "config", "tasks.yaml")
        with open(p("tasks_file", default_cfg).value) as f:
            self.cfg = load_task_config(yaml.safe_load(f))
        self.grasp_check = p("grasp_check", "none").value      # none | gripper (unvalidated in sim)
        self.settle = p("grasp_settle_time", 0.3).value
        # named joint pose the arm moves to (waist held) before the approach when it starts from home: "" = off
        self.clear_pose = p("approach_clear_pose", "clear").value
        move_arm = p("move_arm_action", "/g2cell/move_arm").value
        cbg = ReentrantCallbackGroup()
        self.mc = MotionClient(self, move_arm, timeout=p("motion_timeout", 120.0).value)
        self._js = {}
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data,
                                 callback_group=cbg)
        self.results = self.create_publisher(PickPlaceResult, "/g2cell/pick_results", 50)
        self._busy = threading.Lock()
        self.server = ActionServer(self, PickBag, "/g2cell/pick_bag", execute_callback=self._execute,
                                   goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
                                   callback_group=cbg)
        self.get_logger().info(f"PickBag on /g2cell/pick_bag (motion via {move_arm}, grasp_check={self.grasp_check}); "
                               f"bags on table: {[b.id for b in self.cfg.bags.on_table()]}")

    def _on_js(self, msg):
        self._js = dict(zip(msg.name, msg.position))

    def _on_goal(self, goal):
        if self._busy.locked():
            self.get_logger().warn("rejecting PickBag goal: a pick is already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ---- hooks ------------------------------------------------------------------------
    def _gripper_says(self, ctx):
        if self.grasp_check != "gripper":
            return None
        time.sleep(self.settle)
        return grasp_from_gripper(self._js.get(GRIPPERS[ctx["arm"]]))

    def _hooks(self) -> Hooks:
        return Hooks(grasp_detected=self._gripper_says,
                     holding=lambda ctx, stage: self._gripper_says(ctx),
                     # TODO(Stage 5): scorer -- read the bag's settled pose (needs a scene-body pose
                     # relay; /tf_render only carries robot links) and return a Placement.
                     observe_placement=lambda ctx: None)

    # ---- action -----------------------------------------------------------------------
    def _execute(self, gh):
        with self._busy:
            return self._run(gh)

    def _finish(self, gh, res: PickPlaceResult, t0: float):
        res.stamp = self.get_clock().now().to_msg()
        res.duration = float(time.monotonic() - t0)
        res.outcome_name = next((k for k, v in OUTCOME_CODES.items() if v == res.outcome), "")
        self.results.publish(res)
        out = PickBag.Result(result=res)
        if res.outcome == PickPlaceResult.OUTCOME_ABORTED and gh.is_cancel_requested:
            gh.canceled()
        elif res.outcome in (PickPlaceResult.OUTCOME_SUCCESS, PickPlaceResult.OUTCOME_REGRASP,
                             PickPlaceResult.OUTCOME_UNVERIFIED):
            gh.succeed()
        else:
            gh.abort()
        self.get_logger().info(f"[{res.bag_id or '?'}] {res.outcome_name} (verified={res.verified}) {res.message}")
        return out

    def _run(self, gh):
        t0 = time.monotonic()
        goal = gh.request
        res = PickPlaceResult(bag_id=goal.bag_id)
        cfg = self.cfg
        bag_id = goal.bag_id or "any"

        def reachable(b):
            target, lane = cfg.sort.target_for(b.label)
            return select_arm(cfg.reach, "table", b.xy, target, goal.arm, lane=lane).arm is not None

        try:
            bag = cfg.bags.choose(bag_id, reachable)
        except KeyError:
            res.outcome, res.message = PickPlaceResult.OUTCOME_INVALID, f"unknown bag {bag_id!r}"
            return self._finish(gh, res, t0)
        if bag is None:
            res.outcome = PickPlaceResult.OUTCOME_UNREACHABLE if bag_id == "any" else PickPlaceResult.OUTCOME_INVALID
            res.message = ("no reachable bag left on the table" if bag_id == "any"
                           else f"{bag_id} already left the table")
            return self._finish(gh, res, t0)
        res.bag_id = bag.id
        try:
            target, lane = cfg.sort.resolve(goal.target, goal.lane, bag.label)
            cfg.targets.point(target, lane)    # KeyError for an unknown lane
        except (ValueError, KeyError) as e:
            res.outcome, res.message = PickPlaceResult.OUTCOME_INVALID, str(e)
            return self._finish(gh, res, t0)
        res.intended_target, res.intended_lane = target, lane
        choice = select_arm(cfg.reach, "table", bag.xy, target, goal.arm, lane=lane)
        if choice.arm is None:
            res.outcome, res.message = PickPlaceResult.OUTCOME_UNREACHABLE, choice.reason
            return self._finish(gh, res, t0)
        res.arm = choice.arm
        if not self.mc.wait(5.0):
            res.outcome, res.message = PickPlaceResult.OUTCOME_ABORTED, "motion_server (MoveArm) not available"
            return self._finish(gh, res, t0)
        if not choice.verified:
            self.get_logger().warn(f"{target} {lane}: reach never measured; IK decides")

        ctx = plan_task(cfg, bag, choice.arm, target, lane)

        def on_stage(stage, attempt):
            fb = PickBag.Feedback(stage=STAGE_INDEX[stage], stage_name=stage.value, arm=choice.arm,
                                  bag_id=bag.id, grasp_attempt=attempt)
            gh.publish_feedback(fb)

        runner = MotionRunner(self.mc, cancelled=lambda: gh.is_cancel_requested, clear_pose=self.clear_pose)
        tr = PickPlaceMachine(runner, self._hooks(), cfg.max_grasp_attempts, on_stage).run(ctx)
        if tr.outcome in LEFT_TABLE:
            cfg.bags.mark_removed(bag.id)
        res.outcome, res.verified = OUTCOME_CODES[tr.outcome], tr.verified
        res.grasp_attempts, res.failed_stage, res.message = tr.grasp_attempts, tr.failed_stage, tr.message
        if tr.placement is not None:
            res.actual_target, res.actual_lane = tr.placement.target, tr.placement.lane
        return self._finish(gh, res, t0)


def main():
    rclpy.init()
    node = PickBagServer()
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
