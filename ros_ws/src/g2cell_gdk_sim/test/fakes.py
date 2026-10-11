"""A fake Port (the sim side of the Controller) with a virtual clock. No ROS, no threads."""
from __future__ import annotations

import numpy as np

from agibot_gdk import _geometry as g
from agibot_gdk import _limits as lim
from agibot_gdk._convert import JointSample
from agibot_gdk._controller import GuardError

# Toy link poses in base_link, enough for TF and the head-clearance check.
HEAD3 = g.matrix((0.05, 0.0, 1.45), (0, 0, 0, 1))


class FakePort:
    def __init__(self):
        self.t = 0.0
        self.position = {n: 0.0 for n in lim.ALL + list(lim.GRIPPERS.values())}
        self.position["idx13_head_joint3"] = 0.1
        self.velocity = {n: 0.0 for n in self.position}
        self.buffer = None
        self.stuck: set[str] = set()     # joints that do not follow their target
        self.lag = 0.0                   # 0: joints reach the target at the next sleep; else first-order fraction
        self.conflicts: list[str] = []
        self.started = 0
        self.ik_calls: list[tuple] = []
        self.ik_answer = lambda side, target, seed: {n: seed[n] + 0.01 for n in (lim.LEFT_ARM if side == "left" else lim.RIGHT_ARM)}
        self.fk_calls = 0
        self.arm_near_head = False       # FK puts arm links on the head when True
        self.fk_available = True
        self.frames = {"base_link": np.eye(4), "head_link3": HEAD3,
                       "arm_l_end_link": g.matrix((0.6, 0.3, 1.0), (1, 0, 0, 0)),
                       "arm_r_end_link": g.matrix((0.6, -0.3, 1.0), (1, 0, 0, 0))}
        self.images = {}
        self.opened: list[str] = []
        self.rates = {}

    # clock
    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(dt, 0.0)
        if self.buffer is None:
            return
        for name, target in self.buffer.snapshot().items():
            if name in self.stuck:
                continue
            before = self.position[name]
            self.position[name] = target if not self.lag else before + (1 - self.lag) * (target - before)
            self.velocity[name] = (self.position[name] - before) / max(dt, 1e-9)

    def wall_ns(self) -> int:
        return 1_700_000_000_000_000_000 + int(self.t * 1e9)

    # joints
    def joint_sample(self):
        names = list(self.position)
        return JointSample(names=names, position=[self.position[n] for n in names],
                           velocity=[self.velocity[n] for n in names], effort=[0.5] * len(names),
                           stamp_ns=self.wall_ns())

    def start_stream(self, buffer) -> None:
        if self.conflicts:
            raise GuardError(f"/joint_command already has publisher(s) {self.conflicts}")
        if self.buffer is None:
            self.started += 1
        self.buffer = buffer

    # kinematics
    def ik(self, side, target, seed):
        self.ik_calls.append((side, target.copy(), dict(seed)))
        return self.ik_answer(side, target, seed)

    def fk(self, joints, links):
        self.fk_calls += 1
        if not self.fk_available:
            return None
        out = {}
        for link in links:
            if link == "head_link3":
                out[link] = HEAD3
            else:
                # arm_near_head: left joint 1 beyond 0.25 rad sweeps the arm onto the head
                near = self.arm_near_head and abs(joints.get("idx21_arm_l_joint1", 0.0)) > 0.25
                out[link] = HEAD3 if near else g.matrix((0.6, 0.3, 0.9), (0, 0, 0, 1))
        return out

    # tf
    def lookup(self, parent, child):
        if parent not in self.frames or child not in self.frames:
            return None
        return g.inverse(self.frames[parent]) @ self.frames[child], self.wall_ns()

    def frame_names(self):
        return list(self.frames)

    # images
    def open_image(self, topic):
        self.opened.append(topic)

    def latest_image(self, topic):
        return self.images.get(topic)

    def image_rate(self, topic):
        return self.rates.get(topic)
