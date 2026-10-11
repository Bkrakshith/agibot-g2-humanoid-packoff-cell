# Episode visualisation tools

Two independent tools in this directory:

1. **Offline MP4 kinematic clips**: `render_clip.py` + `predict_episode.py`. Their docstrings
   document them (G2 meshes posed by pinocchio FK, policy ghost, tool-point trails, camera inset,
   synced joint plots → `output/viz/*_kinematics.mp4`). Usage:
   `.venv-viz/bin/python tools/viz/render_clip.py EPISODE.hdf5 [--ckpt CKPT.pt] [--out OUT.mp4]`.
   *(This section replaces an earlier README written alongside those scripts, which the replay
   tool's README overwrote by mistake. Restore any detail from the scripts' docstrings.)*
2. **Live RViz replay**: everything below.

# Live RViz episode replay (recorded vs policy)

Replays a recorded episode on the G2 URDF in RViz, at real time and on a loop:

- **solid robot** = recorded `/joint_states` (all 46 joints, ~100 Hz);
- **ghost robot** (alpha 0.3) = the policy's open-loop step-0 prediction for the 16 commanded
  joints (other joints copied from the recording; gripper mimic joints follow the URDF
  `<mimic>` rules). Without `--checkpoint`, the ghost shows the recorded *command* instead;
- **markers** in `base_link`: fingertip-midpoint path of each arm (blue/cyan = recorded L/R,
  orange/red = ghost L/R; faint = whole episode, solid = trail so far, sphere = now), plus a text
  panel with time/speed/state, gripper opening for both robots (0 rad = closed, -0.785 rad = 100 %
  open), checkpoint step, action MSE and the mean recorded-vs-ghost EE distance.

```bash
cd g2_packoff_cell
tools/viz/replay_rviz.sh output/episodes/stage1_teleop/stage1_teleop.hdf5 \
    --checkpoint output/train/smoke_act --ghost-offset -0.8      # [--speed 0.5] [--no-loop]
tools/viz/stop_replay.sh                                         # stop everything
```

Options: `--checkpoint RUN_DIR|file.pt` (a run dir uses `checkpoints/best.pt`, else `last.pt`),
`--speed`, `--ghost-offset Y` (metres; shifts the ghost sideways, use it when the ghost overlaps the
solid robot, e.g. with a policy that was trained on the episode), `--no-loop`, `--no-rviz`
(headless), `--npz-only`. Any episode HDF5 works (e.g. `output/episodes/pilot_v1/*/*.hdf5`).

Live control (another clean shell, `ROS_DOMAIN_ID=7`):
`ros2 param set /g2_replay paused true|false`, `... speed 0.25`, `... loop false`, `... seek 5.0`.

## Pieces

| file | runs in | does |
|---|---|---|
| `export_replay.py` | `.venv-learn` (CPU, `CUDA_VISIBLE_DEVICES=""`) | HDF5 → `output/viz/<episode>[__<run>].npz`: raw joint_states, aligned 30 Hz positions/commands, predictions (same code path as `eval_offline.py`) |
| `replay_node.py` | host `/usr/bin/python3` + ROS 2 Humble | publishes `/recorded/joint_states`, `/policy/joint_states` (60 Hz), `/replay/markers` (15 Hz), static TF `base_link → {recorded,policy}/base_link`; numpy URDF FK for the paths |
| `replay.launch.py` | ROS 2 Humble | replay_node + 2 × robot_state_publisher (`namespace` + `frame_prefix` recorded/ and policy/) + RViz |
| `replay.rviz` | RViz | Grid, 2 RobotModels (`/recorded/robot_description` alpha 1, `/policy/robot_description` alpha 0.3), MarkerArray; fixed frame `base_link` |
| `replay_rviz.sh` / `stop_replay.sh` | login shell | export, then launch in a clean env (`env -i`, conda libs break Humble), `ROS_DOMAIN_ID=7`, `DISPLAY=:1` |

Never uses domain 0, the `g2cell` container or the sim. Episode files are only read.

## Verified (2026-10-11)

- Headless: both joint_states topics at 60.0 Hz, markers at 15.0 Hz, `/tf` at ~120 Hz;
  `tf2_echo base_link recorded/…` and `policy/…` both resolve.
- Path FK checked against TF while paused at t = 8 s: fingertip links agree to 1 mm for both robots.
- Desktop: RViz on :1 shows both robots, paths and the text panel, and loops.
- Both test episodes export: stage1_teleop (with smoke_act), teleop_20261010_180355 (no checkpoint).

## Notes

- The smoke policy was trained on stage1_teleop (train = val), so its ghost tracks the recording
  within ~2 mm. It only says the model memorised the episode. Use held-out pilot episodes for a
  meaningful view.
- In stage1_teleop the left arm mostly turns the wrist and works the gripper, so the
  fingertip-midpoint path is only ~3 cm long. Watch the robot itself for the motion.
- rviz2's text font here draws every space as a wide gap, so the panel uses `_` instead of spaces.
- RobotModel cannot tint meshes, so the ghost differs from the solid robot by alpha (and the
  optional offset) only. Colour is used on the path markers.
- The KDL warning about root-link inertia and the RViz "unrealistic inertia" errors come from the
  vendor URDF. They are harmless.
