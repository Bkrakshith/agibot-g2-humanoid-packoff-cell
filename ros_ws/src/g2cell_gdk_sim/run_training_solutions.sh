#!/usr/bin/env bash
# Live acceptance: run every agibot-gdk-training solution against Genie Sim through the
# sim-backed agibot_gdk, with a timeout per exercise, and print a summary table.
#
# Run INSIDE the g2cell container with the sim + move_group up (stage1_up.sh with
# scene_g2cell.yaml) and NO other /joint_command publisher (quit keyboard_teleop / motion_server):
#
#   bash /g2cell_ws/src/g2cell_gdk_sim/run_training_solutions.sh [TRAINING_DIR] [-- 01 04 ...]
#
# The robot MOVES in 04 (head), 05 (no keys pressed: reads only), 06 (left wrist sine, 8 s) and
# 07 (left end effector +5 cm in z, then back). Exercises that need hardware the sim lacks are
# reported as XSKIP (expected skip); RUN_SKIPPED=1 runs them anyway.
# Logs: $OUT (default /g2cell_output/gdk_acceptance/<time>).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING="${GDK_TRAINING_DIR:-/g2cell_ws/third_party/agibot-gdk-training}"
if [ $# -gt 0 ] && [ "$1" != "--" ]; then TRAINING="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
ONLY=("$@")

# sim.sh is next to this script in the source tree and in install/share/g2cell_gdk_sim
# shellcheck source=/dev/null
set +u   # ROS setup scripts read unset variables
source "$HERE/sim.sh" "$TRAINING" >/dev/null || exit 1
set -u
[ -d "$TRAINING/solutions" ] || { echo "no training repo at $TRAINING (copy agibot-gdk-training there)" >&2; exit 1; }

OUT="${OUT:-/g2cell_output/gdk_acceptance/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
SOL="$TRAINING/solutions"
export SHOW_ONCE=1   # live tables print one frame and return (as the training tests run them)

pubs=$(ros2 topic info /joint_command 2>/dev/null | sed -n 's/^Publisher count: *//p')
if [ "${pubs:-0}" != "0" ] && [ -z "${FORCE:-}" ]; then
    echo "/joint_command already has $pubs publisher(s). Quit keyboard_teleop / motion_server first (FORCE=1 to try anyway)." >&2
    exit 1
fi

# id | timeout s | expected stdout | command (eval'd) | expected-skip reason
EXERCISES=(
    "01_joints|60|idx27_arm_l_joint7|python3 $SOL/01_joints/joints.py|"
    "02_status|60|control_mode|python3 $SOL/02_status/status.py|"
    "03_frames_and_forces|60|arm_l_end_link|python3 $SOL/03_frames_and_forces/forces.py|"
    "04_joint_moves|180|control mode was|MOVE04|"
    "05_gripper|60|omnipicker|python3 $SOL/05_gripper/gripper.py|"
    "06_servo|60|largest error|python3 $SOL/06_servo/servo.py idx27_arm_l_joint7|"
    "07_ee_pose_up|60|miss in mm|python3 $SOL/07_ee_pose/ee_pose.py left 0 0 0.05|"
    "07_ee_pose_down|60|miss in mm|python3 $SOL/07_ee_pose/ee_pose.py left 0 0 -0.05|"
    "08_tf|60|get_tf_from_base_link|python3 $SOL/08_tf/frames.py|"
    "09_camera|60|head_depth|python3 $SOL/09_camera/camera.py|opens both wrist cameras (kHandLeftColor/kHandRightColor), which the sim does not publish"
    "09_camera_head_only|60|head_depth: (400, 640) uint16|python3 $HERE/scripts/head_camera_check.py $TRAINING|"
    "10_camera_tf|60|in camera|python3 $SOL/10_camera_tf/project.py|"
    "doctor|30|All checks passed|python3 -m gdk_training.doctor|checks the real network (AORTA 2379, port 8849) and a compiled vendor binding"
)

# Exercise 4 reads a prompt. Work in a copy so poses.json stays out of the repo; save the start
# pose, move the head roll, go back, and return to the start pose.
MOVE04_INPUT=$'save start\nidx13_head_joint3 0.2\nsave a\nidx13_head_joint3 0.35\ngo a\ngo start\nq\n'
run_move04() {
    local dir; dir="$(mktemp -d)"
    cp "$SOL/04_joint_moves/move.py" "$dir/move.py"
    printf '%s' "$MOVE04_INPUT" | python3 "$dir/move.py"
    local rc=$?
    rm -rf "$dir"
    return $rc
}

declare -a ROWS
pass=0; fail=0; skip=0
for spec in "${EXERCISES[@]}"; do
    IFS='|' read -r id limit expect cmd reason <<< "$spec"
    if [ ${#ONLY[@]} -gt 0 ] && ! printf '%s\n' "${ONLY[@]}" | grep -qx "${id:0:2}"; then continue; fi
    if [ -n "$reason" ] && [ -z "${RUN_SKIPPED:-}" ]; then
        ROWS+=("$(printf '%-22s %-7s %6s  %s' "$id" XSKIP - "$reason")"); skip=$((skip + 1)); continue
    fi
    log="$OUT/$id.log"
    echo "== $id (timeout ${limit}s) -> $log"
    start=$(date +%s%N)
    if [ "$cmd" = MOVE04 ]; then
        export -f run_move04; export SOL MOVE04_INPUT
        (cd "$TRAINING" && timeout --signal=INT --kill-after=10 "$limit" bash -c run_move04) > "$log" 2>&1
    else
        (cd "$TRAINING" && timeout --signal=INT --kill-after=10 "$limit" bash -c "$cmd" < /dev/null) > "$log" 2>&1
    fi
    rc=$?
    secs=$(awk -v a="$start" -v b="$(date +%s%N)" 'BEGIN { printf "%.1f", (b - a) / 1e9 }')
    note=""
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then status=TIMEOUT
    elif [ $rc -ne 0 ]; then status=FAIL; note="exit $rc: $(grep -v '^\s*$' "$log" | tail -n 1 | cut -c1-90)"
    elif ! grep -qF "$expect" "$log"; then status=FAIL; note="missing '$expect' in output"
    else status=PASS
    fi
    if [ "$id" = 04_joint_moves ] && [ $status = PASS ]; then
        n=$(grep -c 'result 0' "$log"); [ "$n" -eq 4 ] || { status=FAIL; note="$n of 4 moves returned 0"; }
    fi
    [ -z "$note" ] && note="$(grep -m1 -E 'largest error|miss in mm|result ' "$log" | cut -c1-90)"
    [ $status = PASS ] && pass=$((pass + 1)) || fail=$((fail + 1))
    ROWS+=("$(printf '%-22s %-7s %6s  %s' "$id" "$status" "$secs" "$note")")
done

echo
printf '%-22s %-7s %6s  %s\n' exercise status seconds note
printf '%s\n' "------------------------------------------------------------------------------"
printf '%s\n' "${ROWS[@]}"
echo
echo "pass $pass   fail $fail   expected skip $skip   logs: $OUT"
[ $fail -eq 0 ]
