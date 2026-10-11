"""Teleop node parameters: declared defaults and builders (ROS-free, unit-tested).

config/teleop_g2.yaml must list exactly these keys with the same types (a test checks it).
"""
from __future__ import annotations

from g2cell_teleop.devices import SpaceMouseConfig
from g2cell_teleop.motion import MotionLimits
from g2cell_teleop.teleop_core import DEFAULT_HOME_DEG, home_joint_targets

DEFAULTS: dict = {
    # robot / MoveIt
    "base_frame": "base_link",
    "left_group": "simple_arm_l",
    "right_group": "simple_arm_r",
    "left_ik_link": "arm_l_end_link",
    "right_ik_link": "arm_r_end_link",
    "left_joint_match": "arm_l_joint",
    "right_joint_match": "arm_r_joint",
    "ik_timeout": 0.05,
    "rate_hz": 50.0,
    # grippers (omnipicker leader joints: 0 = closed, -0.785 = open)
    "left_gripper_joint": "idx31_gripper_l_inner_joint1",
    "right_gripper_joint": "idx71_gripper_r_inner_joint1",
    "gripper_lower": -0.785,
    "gripper_upper": 0.0,
    "gripper_closed_is_lower": False,
    # home pose, joints 1..7 in degrees (default: scene init pose)
    "home_left_deg": list(DEFAULT_HOME_DEG),
    "home_right_deg": list(DEFAULT_HOME_DEG),
    "home_joint_speed": 0.5,  # rad/s while homing
    # waist yaw (keys , . and auto-face t), idx05_body_joint5
    "waist_joint": "idx05_body_joint5",
    "waist_step_deg": 5.0,
    "waist_min_deg": -100.0,
    "waist_max_deg": 100.0,
    "waist_speed": 0.5,  # rad/s ramp of the commanded waist
    "home_resets_waist": True,
    "auto_face": False,  # start with auto-face on
    "auto_face_cone_deg": 30.0,
    "auto_face_pivot": [0.258, 0.0],  # waist axis at arm-base height, base_frame xy
    # end-effector workspace boxes in base_frame, metres
    "use_workspace": True,
    "workspace_left_min": [0.15, -0.25, 0.5],
    "workspace_left_max": [0.95, 0.75, 1.7],
    "workspace_right_min": [0.15, -0.75, 0.5],
    "workspace_right_max": [0.95, 0.25, 1.7],
    # continuous-device motion limits
    "max_lin_speed": 0.1,
    "max_ang_speed": 0.5,
    "max_lin_acc": 0.5,
    "max_ang_acc": 2.5,
    "lin_deadband": 0.0,
    "ang_deadband": 0.0,
    # input device
    "device": "keyboard",  # keyboard | spacemouse
    "spacemouse_source": "topic",  # topic (ros-jazzy-spacenav) | pyspacemouse
    "spacemouse_twist_topic": "/spacenav/twist",
    "spacemouse_joy_topic": "/spacenav/joy",
    "spacemouse_lin_scale": 0.1,
    "spacemouse_ang_scale": 0.5,
    "spacemouse_deadband": 0.1,
    "spacemouse_axis_map": [0, 1, 2, 3, 4, 5],
    "spacemouse_axis_sign": [1, 1, 1, 1, 1, 1],
    "spacemouse_gripper_button": 0,
    "spacemouse_switch_button": 1,
    "spacemouse_stale_after": 0.5,
    # recording
    "recording_enabled": True,
    "scene": "scene_g2cell",
    "seed": 0,
    "episodes_dir": "/g2cell_output/episodes",
}


def limits_from(p: dict) -> MotionLimits:
    return MotionLimits(max_lin=p["max_lin_speed"], max_ang=p["max_ang_speed"],
                        max_lin_acc=p["max_lin_acc"], max_ang_acc=p["max_ang_acc"],
                        lin_deadband=p["lin_deadband"], ang_deadband=p["ang_deadband"])


def spacemouse_config_from(p: dict) -> SpaceMouseConfig:
    return SpaceMouseConfig(
        lin_scale=p["spacemouse_lin_scale"], ang_scale=p["spacemouse_ang_scale"],
        deadband=p["spacemouse_deadband"],
        axis_map=tuple(int(i) for i in p["spacemouse_axis_map"]),
        axis_sign=tuple(int(s) for s in p["spacemouse_axis_sign"]),
        gripper_button=p["spacemouse_gripper_button"], switch_button=p["spacemouse_switch_button"],
        stale_after=p["spacemouse_stale_after"])


def workspace_from(p: dict):
    if not p["use_workspace"]:
        return None
    return {arm: (tuple(p[f"workspace_{arm}_min"]), tuple(p[f"workspace_{arm}_max"]))
            for arm in ("left", "right")}


def home_from(p: dict) -> dict[str, dict[str, float]]:
    return home_joint_targets(p["home_left_deg"], p["home_right_deg"])
