"""Republish a topic at a lower rate, without deserialising it (record depth at e.g. 5 Hz).

    ros2 run g2cell_recorder throttle_relay --ros-args \
        -p input:=/genie_sim/head_front_camera_depth/image_raw/compressedDepth \
        -p output:=/g2cell/rec/head_front_camera_depth/compressedDepth \
        -p rate_hz:=5.0 -p msg_type:=sensor_msgs/msg/CompressedImage

record_episode.sh starts it for --depth-rate. Messages are forwarded as serialised bytes, so the
header stamps are the original ones; the bag receive time is the relay's publish time (~1 ms
later). Rate gating uses the monotonic clock (works with sim time and wall time alike).
"""
from __future__ import annotations

import sys
import time


class RateGate:
    """Pass at most `rate_hz` messages per second; a gap of >= 90 % of the period passes."""

    def __init__(self, rate_hz: float, tolerance: float = 0.1):
        if not rate_hz or rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.min_gap_ns = int(1e9 / rate_hz * (1.0 - tolerance))
        self.last_ns: int | None = None

    def accept(self, t_ns: int) -> bool:
        if self.last_ns is None or t_ns - self.last_ns >= self.min_gap_ns:
            self.last_ns = t_ns
            return True
        return False


def main(argv=None) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(args=argv if argv is not None else sys.argv)
    node = Node("g2cell_throttle_relay")
    src = node.declare_parameter("input", "").value
    dst = node.declare_parameter("output", "").value
    rate = float(node.declare_parameter("rate_hz", 5.0).value)
    typ = node.declare_parameter("msg_type", "sensor_msgs/msg/CompressedImage").value
    if not src or not dst:
        node.get_logger().error("set -p input:=... -p output:=...")
        return 2
    cls = get_message(typ)
    gate = RateGate(rate)
    pub = node.create_publisher(cls, dst, 10)
    counts = [0, 0]

    def cb(raw: bytes):
        counts[0] += 1
        if gate.accept(time.monotonic_ns()):
            counts[1] += 1
            pub.publish(raw)

    node.create_subscription(cls, src, cb, qos_profile_sensor_data, raw=True)
    node.get_logger().info(f"relaying {src} -> {dst} at <= {rate:g} Hz ({typ})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001 - ExternalShutdown etc.
        if not isinstance(e, KeyboardInterrupt) and "Shutdown" not in type(e).__name__:
            raise
    finally:
        node.get_logger().info(f"relayed {counts[1]} of {counts[0]} messages")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
