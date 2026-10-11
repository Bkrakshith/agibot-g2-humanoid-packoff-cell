"""Record ROS image topics straight into H.264 MP4s (no raw frames on disk).

A timer at `fps` writes the latest frame of each topic into its own ffmpeg process
(rawvideo stdin -> libx264, yuv420p), so the videos share one clock and can be stacked later.
Optional: start on "start" and stop 1.5 s after "done" on /g2cell_demo/state; "pause"/"resume"
there cut idle time (e.g. planning) out of the video; PNG stills at the given video times (s).

    ros2 run g2cell_demo record_cams --ros-args -p out_dir:=/g2cell_output/renders \
        -p topics:="['/genie_sim/free_camera_rgb/image_raw','/genie_sim/head_front_camera_rgb/image_raw']" \
        -p names:="['g2cell_overview','g2cell_head']" -p wait_for_start:=true -p max_duration:=60.0 \
        -p stills:="[3.0, 10.0, 20.0]"
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

CHANNELS = {"rgba8": 4, "bgra8": 4, "rgb8": 3, "bgr8": 3}


def to_rgb(msg) -> np.ndarray:
    if isinstance(msg, CompressedImage):  # .../image_raw/compressed (JPEG/PNG via image_transport)
        import cv2
        return np.ascontiguousarray(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1])
    c = CHANNELS[msg.encoding]
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * c]
    a = a.reshape(msg.height, msg.width, c)
    if msg.encoding.startswith("bgr"):
        a = a[:, :, 2::-1] if c == 3 else a[:, :, [2, 1, 0]]
    return np.ascontiguousarray(a[:, :, :3])


class RecordCams(Node):
    def __init__(self):
        super().__init__("g2cell_record_cams")
        p = self.declare_parameter
        self.topics = list(p("topics", ["/genie_sim/free_camera_rgb/image_raw"]).value)
        self.names = list(p("names", ["g2cell_overview"]).value)
        self.out_dir = p("out_dir", "/g2cell_output/renders").value
        self.fps = float(p("fps", 12.0).value)
        self.max_duration = float(p("max_duration", 40.0).value)
        self.wait_for_start = bool(p("wait_for_start", False).value)
        self.stills = sorted(float(t) for t in p("stills", [0.0]).value)
        self.crf = int(p("crf", 20).value)
        os.makedirs(self.out_dir, exist_ok=True)
        self.latest: dict[str, Image] = {}
        self.lock = threading.Lock()
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        # Prefer the .../compressed topics: 3.7 MB raw 720p frames over DDS on loopback caused
        # occasional ~1 s /tf_render delivery stalls (visible as jumps) on the 8 GB dev box.
        for t in self.topics:
            typ = CompressedImage if t.endswith("/compressed") else Image
            self.create_subscription(typ, t, lambda m, t=t: self._on_img(t, m), qos)
        self.create_subscription(String, "/g2cell_demo/state", self._on_state, 10)
        self.procs: dict[str, subprocess.Popen] = {}
        self.t_start = None if self.wait_for_start else time.monotonic()
        self.t_stop = None
        self.paused_at = None
        self.frames = 0
        self.done = threading.Event()
        self.create_timer(1.0 / self.fps, self._tick)

    def _on_img(self, topic, msg):
        with self.lock:
            self.latest[topic] = msg

    def _on_state(self, msg):
        if msg.data == "start" and self.t_start is None:
            self.t_start = time.monotonic()
            self.get_logger().info("start")
        elif msg.data == "pause" and self.t_start is not None and self.paused_at is None:
            self.paused_at = time.monotonic()
        elif msg.data == "resume" and self.paused_at is not None:
            self.t_start += time.monotonic() - self.paused_at      # video time skips the pause
            self.paused_at = None
        elif msg.data == "done" and self.t_start is not None and self.t_stop is None:
            self.t_stop = time.monotonic() + 1.5

    def _ffmpeg(self, name, w, h):
        out = os.path.join(self.out_dir, f"{name}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
               "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _tick(self):
        if self.t_start is None or self.done.is_set() or self.paused_at is not None:
            return
        with self.lock:
            if any(t not in self.latest for t in self.topics):
                return
            msgs = [self.latest[t] for t in self.topics]
        now = time.monotonic()
        elapsed = now - self.t_start
        for name, msg in zip(self.names, msgs):
            rgb = to_rgb(msg)
            if name not in self.procs:
                self.procs[name] = self._ffmpeg(name, rgb.shape[1], rgb.shape[0])
            self.procs[name].stdin.write(rgb.tobytes())
            if self.stills and elapsed >= self.stills[0]:
                import cv2
                cv2.imwrite(os.path.join(self.out_dir, f"{name}_t{self.stills[0]:04.1f}s.png"), rgb[:, :, ::-1])
        if self.stills and elapsed >= self.stills[0]:
            self.stills.pop(0)
        self.frames += 1
        if elapsed >= self.max_duration or (self.t_stop is not None and now >= self.t_stop):
            self.finish()

    def finish(self):
        if self.done.is_set():
            return
        self.done.set()
        for name, p in self.procs.items():
            p.stdin.close()
            p.wait()
            self.get_logger().info(f"wrote {self.out_dir}/{name}.mp4 ({self.frames} frames @ {self.fps} fps)")


def main():
    rclpy.init()
    node = RecordCams()
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.finish()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
