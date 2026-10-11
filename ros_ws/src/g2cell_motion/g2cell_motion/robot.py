"""G2 (crs arm, omnipicker gripper) constants as used by MoveIt and Genie Sim (no ROS imports)."""
from __future__ import annotations

import math

ARMS = ("left", "right")
GROUPS = {"left": "simple_arm_l", "right": "simple_arm_r"}
IK_LINKS = {"left": "arm_l_end_link", "right": "arm_r_end_link"}
ARM_MATCH = {"left": "arm_l_joint", "right": "arm_r_joint"}
# omnipicker leader joints: 0 = closed, -0.785 = open (confirmed on camera, Stage 1)
GRIPPERS = {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}
GRIP_OPEN, GRIP_CLOSED = -0.785, 0.0
# Home = arm part of scene_g2cell.yaml init_joint_pos, joints 1..7 in degrees.
HOME_DEG = [0.0, -45.0, 0.0, -75.0, 0.0, -55.0, 0.0]

# Arm colliders are disabled in Genie Sim's G2, so the planner keeps every arm/gripper link this
# far from the head (head_link3 origin and the head camera) along the interpolated joint path.
HEAD_CLEARANCE = 0.20
HEAD_LINK = "head_link3"
HEAD_CAM_IN_LINK3 = (0.10237, 0.02375, 0.10256)   # official head_front_Camera mount (scene YAML)
CHECK_LINKS = {a: [f"arm_{a[0]}_link{i}" for i in (3, 4, 5, 6, 7)] + [f"arm_{a[0]}_end_link",
                                                                          f"gripper_{a[0]}_base_link"]
               for a in ARMS}
# fingertip links (~0.11 m beyond the gripper base), checked against the cell's fixtures only
FINGER_LINKS = {a: [f"gripper_{a[0]}_inner_link4", f"gripper_{a[0]}_outer_link4"] for a in ARMS}


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def other_arm(arm: str) -> str:
    _check_arm(arm)
    return "right" if arm == "left" else "left"


def arm_joint_names(names, arm: str) -> list[str]:
    """The arm's 7 joints out of a /joint_states name list, sorted (idx21..27 / idx61..67)."""
    _check_arm(arm)
    return sorted(n for n in names if ARM_MATCH[arm] in n)


def home_joints(arm_names: list[str], degrees=HOME_DEG) -> dict[str, float]:
    """{joint: rad} for sorted arm joint names (joint k = degrees[k-1])."""
    if len(arm_names) != len(degrees):
        raise ValueError(f"{len(arm_names)} joints but {len(degrees)} angles")
    return {n: math.radians(d) for n, d in zip(arm_names, degrees)}


def gripper_value(cmd) -> float:
    """'open' | 'close' | a joint value (clamped to [GRIP_OPEN, GRIP_CLOSED])."""
    if isinstance(cmd, str):
        if cmd == "open":
            return GRIP_OPEN
        if cmd == "close":
            return GRIP_CLOSED
        raise ValueError(f"gripper command must be 'open', 'close' or a number, got {cmd!r}")
    return min(max(float(cmd), GRIP_OPEN), GRIP_CLOSED)

# genie_sim_moveit_plugins/config/coupled_constraints.yaml (arm rows): MoveIt's KDL plugin rejects IK
# solutions with A . (joint6, joint7) > b - coupled_constraint_margin (0.1) and re-randomises, so the local
# IK (g2cell_motion.kinematics) enforces the same rows. Same matrix for both arms.
ARM_COUPLED = {"A": [[0.7409, 0.6716], [-0.7403, -0.6723], [0.5990, -0.8008], [-0.7401, 0.6725]],
               "b": [1.2430, 1.2427, 1.3267, 1.2452]}
ARM_COUPLED_MARGIN = 0.1
