"""Merge the two G2 chassis lidar clouds into one full-circle cloud in base_link (motion-compensated).

Why: Genie Sim's render node publishes /lidar/merged with only the lidar(s) whose scan completed in
that render frame, so it alternates front-only / back-only clouds. Each half-blind cloud turned into
a LaserScan has ~180 deg of `inf` beams, which slam_toolbox / costmaps ray-trace as free space and
erase walls the other lidar saw. This node keeps the latest cloud of every lidar, moves each one into
`odom` with the TF at its own stamp (odom->base_link is ground truth in Genie Sim, so this is exact),
and republishes the union in base_link at the newest stamp.

Input clouds come in their sensor frames (livox_front / livox_back). The render node only publishes
the base_link->sensor static TF when it sees a "base_link" /tf_render frame, which the engine never
sends, so the mounts are parameters here (same values as lidars[].mount in scene_g2cell_mobile.yaml)
and this node also publishes them on /tf_static for RViz.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_to_mat(t):
    T = np.eye(4)
    r, p = t.transform.rotation, t.transform.translation
    T[:3, :3] = quat_to_R(r.x, r.y, r.z, r.w)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


class CloudMerger(Node):
    def __init__(self):
        super().__init__("g2cell_cloud_merger")
        p = self.declare_parameter
        self.topics = list(p("topics", ["/lidar/livox_front", "/lidar/livox_back"]).value)
        # mounts in base_link: flat list x, y, z, yaw_deg per topic (ESTIMATED, see scene YAML)
        m = list(p("mounts", [0.40, 0.0, 0.18, 0.0, -0.40, 0.0, 0.18, 180.0]).value)
        self.base = p("base_frame", "base_link").value
        self.fixed = p("fixed_frame", "odom").value
        self.max_age = float(p("max_age", 0.6).value)
        self.pub = self.create_publisher(PointCloud2, p("output", "/lidar/merged_full").value, 5)
        self.mounts = {}
        statics = []
        for i, t in enumerate(self.topics):
            x, y, z, yaw = m[4 * i:4 * i + 4]
            T = np.eye(4)
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = (x, y, z)
            self.mounts[t] = T
            ts = TransformStamped()
            ts.header.frame_id = self.base
            ts.child_frame_id = t.rsplit("/", 1)[-1]
            ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = x, y, z
            ts.transform.rotation.z, ts.transform.rotation.w = math.sin(math.radians(yaw) / 2), math.cos(math.radians(yaw) / 2)
            statics.append(ts)
        self.static_br = StaticTransformBroadcaster(self)
        self.static_br.sendTransform(statics)
        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        self.tfl = TransformListener(self.tf, self)
        self.latest = {}  # topic -> (stamp_ns, Nx3 points in fixed frame)
        for t in self.topics:
            self.create_subscription(PointCloud2, t, lambda msg, t=t: self.on_cloud(t, msg), qos_profile_sensor_data)
        self.n_out = 0
        self.get_logger().info(f"merging {self.topics} -> {self.pub.topic_name} in {self.base} (via {self.fixed})")

    def lookup(self, stamp):
        return tf_to_mat(self.tf.lookup_transform(self.fixed, self.base, stamp, Duration(seconds=0.2)))

    def on_cloud(self, topic, msg):
        stamp = Time.from_msg(msg.header.stamp)
        try:
            T_fb = self.lookup(stamp)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF {self.fixed}->{self.base} at cloud stamp: {e}", throttle_duration_sec=5.0)
            return
        xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z")).astype(np.float64)
        if xyz.size == 0:
            return
        T = T_fb @ self.mounts[topic]
        self.latest[topic] = (stamp.nanoseconds, xyz @ T[:3, :3].T + T[:3, 3])
        # output at the newest stamp, in base_link at that stamp
        T_bf = np.linalg.inv(T_fb)
        parts = [pts for (ns, pts) in self.latest.values() if (stamp.nanoseconds - ns) * 1e-9 <= self.max_age]
        merged = np.concatenate(parts) @ T_bf[:3, :3].T + T_bf[:3, 3]
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg_out = pc2.create_cloud(msg.header.__class__(stamp=msg.header.stamp, frame_id=self.base), fields,
                                   merged.astype(np.float32))
        self.pub.publish(msg_out)
        self.n_out += 1


def main():
    rclpy.init()
    node = CloudMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
