"""cmd_vel -> /cmd_twist bridge for the Genie Sim G2 chassis.

The G2 base is not driven by /cmd_vel. Genie Sim's chain is
  /cmd_twist (geometry_msgs/Twist, base frame) -> servo_node (genie_sim_controllers, 4WS MPC)
  -> /cmd_4ws (sensor_msgs/JointState: steer positions + drive velocities) -> engine.
This node forwards Nav2 / teleop /cmd_vel to /cmd_twist with clamping, drops linear.y (the servo
ignores lateral motion in practice, measured), and publishes zero twist when the input goes stale so
the base stops (the servo's own twist timeout is 1 s).
"""
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node


def clamp(v, lim):
    return max(-lim, min(lim, v))


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__("g2cell_cmd_vel_bridge")
        p = self.declare_parameter
        self.in_topic = p("input", "/cmd_vel").value
        self.out_topic = p("output", "/cmd_twist").value
        self.max_v = float(p("max_linear", 0.4).value)
        self.max_w = float(p("max_angular", 0.8).value)
        self.timeout = float(p("timeout", 0.5).value)
        self.stamped = bool(p("stamped_input", False).value)
        self.pub = self.create_publisher(Twist, self.out_topic, 10)
        typ = TwistStamped if self.stamped else Twist
        self.create_subscription(typ, self.in_topic, self.on_cmd, 10)
        self.last = None
        self.cmd = Twist()
        self.create_timer(0.05, self.tick)  # 20 Hz to the servo
        self.get_logger().info(f"{self.in_topic} -> {self.out_topic} (|v|<={self.max_v}, |w|<={self.max_w})")

    def on_cmd(self, msg):
        tw = msg.twist if self.stamped else msg
        out = Twist()
        out.linear.x = clamp(tw.linear.x, self.max_v)
        out.angular.z = clamp(tw.angular.z, self.max_w)
        self.cmd, self.last = out, time.monotonic()

    def tick(self):
        if self.last is None:
            return
        if time.monotonic() - self.last > self.timeout:
            self.pub.publish(Twist())
            if time.monotonic() - self.last > self.timeout + 1.0:
                self.last = None  # stop publishing once the base has been told to stop for 1 s
            return
        self.pub.publish(self.cmd)


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
