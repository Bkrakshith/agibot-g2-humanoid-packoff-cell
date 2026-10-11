"""tools/episode_sheet.py event detection (numpy only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episode_sheet import gripper_events  # noqa: E402


def test_grasp_is_the_first_close_after_the_gripper_was_open():
    # closed at home, opened for the approach, closed on the bag at t=5, opened over the carton at t=9
    t = np.arange(0, 12, 0.5)
    q = np.where(t < 1, 0.0, -0.785)
    q[(t >= 5) & (t < 9)] = 0.0
    ev = gripper_events(t, q)
    assert ev["grasp"] == 5.0 and ev["place"] == 9.0 and ev["lift"] == 6.5 and ev["transfer"] == 7.0


def test_never_closed_gives_start_and_end_only():
    t = np.arange(0, 3, 0.5)
    ev = gripper_events(t, np.full_like(t, -0.785))
    assert set(ev) == {"start", "end"}
