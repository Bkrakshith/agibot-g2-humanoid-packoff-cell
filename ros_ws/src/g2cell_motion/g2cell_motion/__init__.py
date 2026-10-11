"""g2cell_motion: reusable G2 arm motion (IK waypoints, head clearance, quintic joint streaming).

Pure modules (no ROS imports, unit-tested): geometry, robot, named_poses, clearance, planner,
guard, streamer. ROS modules: ik_client (MoveIt /compute_ik + /compute_fk mixin),
motion_server (MoveArm action server, the single /joint_command publisher) and client
(MoveArm action client used by g2cell_tasks).
"""
