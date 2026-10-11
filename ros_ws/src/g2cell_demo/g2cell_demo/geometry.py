"""Moved to g2cell_motion.geometry (shared by the demo, motion_server and the tasks); re-exported
here so existing imports keep working."""
from g2cell_motion.geometry import (BASE_Z, TOOL_OFFSET, axis_angle, down_quat, duration_for,  # noqa: F401
                                    end_link_for_tool, interpolate, min_jerk, qmul, qrot)
