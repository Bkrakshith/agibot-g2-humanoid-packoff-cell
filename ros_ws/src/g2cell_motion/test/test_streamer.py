import math

from g2cell_motion.streamer import CommandBuffer, run_segment


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def test_buffer_snapshot_and_get():
    b = CommandBuffer()
    assert b.empty()
    b.set({"a": 1.0, "b": 2.0})
    b.set({"a": 3.0})
    assert b.get(["a", "b"]) == [3.0, 2.0]
    assert dict(b.snapshot()) == {"a": 3.0, "b": 2.0}


def test_run_segment_streams_at_rate_and_ends_on_target():
    b = CommandBuffer()
    b.set({"a": 0.0})
    c = FakeClock()
    seen = []
    ok = run_segment(b, {"a": 1.0}, duration=1.0, rate=50.0, now=c.now, sleep=c.sleep,
                     on_step=lambda i, n: seen.append((i, n)))
    assert ok and b.get(["a"]) == [1.0]
    assert seen[-1] == (50, 50) and math.isclose(c.t, 1.0, abs_tol=1e-6)


def test_run_segment_cancel_stops_midway():
    b = CommandBuffer()
    b.set({"a": 0.0})
    c = FakeClock()
    ok = run_segment(b, {"a": 1.0}, duration=1.0, rate=50.0, now=c.now, sleep=c.sleep,
                     cancelled=lambda: c.t > 0.5)
    assert not ok and 0.0 < b.get(["a"])[0] < 1.0


def test_run_segment_unknown_joint_starts_from_target():
    b = CommandBuffer()
    c = FakeClock()
    assert run_segment(b, {"g": -0.785}, duration=0.1, rate=50.0, now=c.now, sleep=c.sleep)
    assert b.get(["g"]) == [-0.785]
