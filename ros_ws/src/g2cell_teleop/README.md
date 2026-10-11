# g2cell_teleop

End-effector teleop for the AgiBot G2 in Genie Sim, built for collecting data. You drive the
active arm with the keyboard (discrete steps) or a SpaceMouse (continuous 6D). MoveIt
`/compute_ik` turns the targets into joint angles on `/joint_command`, and the same terminal
starts and stops `g2cell_recorder` episodes and marks them as successes or failures.

Run this node as the only `/joint_command` publisher. It needs `move_group` running with
`use_ros2_control:=false` (Stage 1 in the top-level README).

## Run

```bash
# keyboard; the episode keys need a real terminal (TTY)
ros2 run g2cell_teleop keyboard_teleop --ros-args \
  --params-file $(ros2 pkg prefix g2cell_teleop)/share/g2cell_teleop/config/teleop_g2.yaml \
  -p scene:=scene_pnp_g2_op -p seed:=3

# launch file: device:=keyboard|spacemouse, scene, seed, params_file
ros2 launch g2cell_teleop teleop.launch.py device:=spacemouse scene:=scene_pnp_g2_op seed:=3
```

`ros2 launch` does not pass your terminal's stdin to the node. With the launch file, the
keyboard (including `r n f 0 p`) works only if you add `terminal_prefix:='xterm -e'`, which
needs a display. Over SSH, use `ros2 run` as shown above. With `device:=spacemouse` the
keyboard still works whenever the node has a TTY.

Set every parameter in [`config/teleop_g2.yaml`](config/teleop_g2.yaml): MoveIt groups and
IK links, gripper joints and limits, the home pose, workspace boxes, speed limits, SpaceMouse
mapping, and the recording directory.

## Keys

| key | action |
|---|---|
| `1` / `2` | select the left / right arm |
| `w` `s` / `a` `d` / `q` `e` | ±X (forward) / ±Y (left) / ±Z (up) in the base frame, one step |
| `i` `k` / `j` `l` / `u` `o` | pitch / roll / yaw ±, one angular step (5°) |
| `g` / `h` | close / open the gripper by 0.1 rad |
| `[` / `]` | halve / double the step (1–50 mm) |
| `r` | start / stop recording an episode |
| `n` / `f` | mark the current (or the just-stopped) episode as success / failure |
| `,` / `.` | yaw the waist (`idx05_body_joint5`) left / right by 5°, clamped to ±100°, held across arm moves; both arms turn with the torso (targets re-seeded by FK) |
| `t` | auto-face on/off: each arm move first turns the waist so the active target is within ±30° of the torso (g2cell_motion `waist_policy`), IK in the turned frame; falls back to the current waist |
| `0` | ramp both arms to the home pose (default: the scene init pose), then re-seed targets; also turns the waist back to 0 (`home_resets_waist`) |
| `p` | pause / resume: holds the targets and grippers; `r n f 0` still work |
| `x` or `Esc` | quit; a running recording is stopped and converted first |

When IK has no solution, the target stays where it was. Targets are clamped to each arm's
workspace box. A target that starts outside its box is never pulled into it, but it cannot
move further out.

SpaceMouse: push or twist to move the active arm. Speed is capped at 0.1 m/s and 0.5 rad/s,
with a deadband and an acceleration limit. Button 0 toggles the gripper between open and
closed. Button 1 switches the arm.

## Recording

`r` runs `ros2 run g2cell_recorder record_episode.sh teleop_<scene>_<YYYYmmdd_HHMMSS> --scene
<scene> --seed <seed>` in its own process group. The episode goes to
`<episodes_dir>/<name>/`. The second `r` sends that group SIGINT, the same as Ctrl+C, so the
bag closes cleanly and converts to HDF5 in the background. You can mark the outcome with `n`
or `f` during the recording or after it. Once the conversion has finished, the node runs
`ros2 run g2cell_recorder mark_episode <h5> --success true|false`, which sets the HDF5 root
`success` attr. An unmarked episode keeps `success=unknown`. If you change your mind, press
`n` or `f` again to overwrite the mark. A new episode cannot start until the previous one has
finished converting.

## SpaceMouse

You can feed the node in two ways, set with `spacemouse_source`:

- `topic` (default): `/spacenav/twist` and `/spacenav/joy` from `spacenav_node`
  (`ros-jazzy-spacenav`). The node uses best-effort QoS, so reliable publishers also connect.
  The cell image does not include this package yet. Inside the container, run
  `apt-get install ros-jazzy-spacenav`, or add it to `docker/Dockerfile.cell`.
- `pyspacemouse`: reads the USB device directly with `pip install pyspacemouse` (needs
  hidapi and hidraw access). This only works where the device is plugged in.

The default `spacemouse_axis_map` and `spacemouse_axis_sign` assume the source already
reports in the ROS base convention: x forward, y left, z up. These defaults have not been
checked on hardware. Push each axis once, watch the target in Foxglove, and fix the map and
signs in the yaml. The same applies to `spacemouse_lin_scale` and `spacemouse_ang_scale`,
which assume full deflection is about ±1. spacenav_node's default output is normalized, and
pyspacemouse reports values in −1..1.

### SpaceMouse on a remote server (Paperspace)

The SpaceMouse is plugged into your laptop and the sim runs on Paperspace, so the laptop's
spacenavd is not reachable from the server. The simplest option is to forward spacenavd's
Unix socket over the SSH session you already use. `spacenav_node` then runs on the server and
talks to the laptop's daemon as if it were local. The laptop needs no ROS install, VPN or
DDS/zenoh setup.

1. On the laptop (Linux): `sudo apt install spacenavd`, start it, and check that
   `/var/run/spnav.sock` exists.
2. Forward the socket into the cell's output directory. The container already mounts that
   directory at `/g2cell_output`:

   ```bash
   ssh -o StreamLocalBindUnlink=yes \
       -R /home/paperspace/ntu_dev/Agibot_dev/g2_packoff_cell/output/spnav.sock:/var/run/spnav.sock \
       -L 8765:localhost:8765 paperspace@<machine-ip>
   ```

   Adjust the remote path to where `g2_packoff_cell` lives on the server.
   `StreamLocalBindUnlink` replaces a stale socket left by an earlier session.
3. In the container, libspnav looks only at `/var/run/spnav.sock`, so link it once as root.
   Then run the node and teleop:

   ```bash
   docker exec -u root g2cell ln -sf /g2cell_output/spnav.sock /var/run/spnav.sock
   ros2 run spacenav spacenav_node &
   ros2 run g2cell_teleop keyboard_teleop --ros-args --params-file .../teleop_g2.yaml -p device:=spacemouse
   ```

This setup is untested: it has not been run against real hardware or the Paperspace machine.
If the socket forward doesn't work, run ROS 2 on the laptop instead. Run `spacenav_node`
there and bridge only `/spacenav/*` to the server with `zenoh-bridge-ros2dds`, one bridge on
each side. The laptop bridge connects through `ssh -L 7447:localhost:7447`, and the server's
nodes don't change RMW. Avoid USB/IP forwarding: it needs kernel modules and root on the
Paperspace VM.

## Code layout

All logic is ROS-free and unit-tested (`python3 -m pytest test`). Only the node imports rclpy.

| file | contents |
|---|---|
| `teleop_core.py` | state, key map, episode keys, home pose, workspace clamp, help text |
| `motion.py` | twist filter (deadband, speed cap, acceleration limit) and twist → pose integration |
| `devices.py` | `DeviceInput` interface; keyboard, spacenav-topic and pyspacemouse devices |
| `recorder_control.py` | starts and stops `record_episode.sh`, then marks the outcome after conversion |
| `params.py` | declared parameter defaults and builders (`config/teleop_g2.yaml` must match them) |
| `keyboard_teleop_node.py` | the ROS node: MoveIt IK/FK, `/joint_command`, one control loop |
