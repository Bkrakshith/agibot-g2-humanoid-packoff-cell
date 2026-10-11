"""Layout v2 demo, one continuous sequence on the MOBILE G2 (scene_g2cell_mobile.yaml, WITH_MOVEIT=1):

  1. drive from the start pose outside the U into the working pose (drive_in, /cmd_twist on /odom)
  2. base stopped (zero twist held); head turns to the table, arms go tuck -> home
  3. left arm: pick a bag on the pick table (grasp, lift), carry it over the open carton at the pack
     position on the conveyor, open the gripper over the carton, retract, home
  4. right arm: reach to the drop point of rack tote L1_1 (front of its opening, 0.10 m above it),
     open the gripper ("sort"), retract, home
  5. "done" on /g2cell_demo/state

Arm targets are fingertip points in the CELL frame. They are mapped into the robot's actual base frame
from /odom after the drive (so a few cm of final drive-in error do not shift the targets), then solved
with MoveIt /compute_ik in base_link exactly like pick_place_demo (head-clearance check, minimum-jerk
joint streaming at 50 Hz on /joint_command; this node is the ONLY /joint_command publisher).
Base drift during arm work is monitored on /odom and reported.

Planning is deterministic and fast (docs/OPTIMISATION.md "Motion planning"): local numpy IK from the staged
URDF with /compute_ik confirming each answer (local_ik, ik_confirm), each IK chain solved once (memoised),
the path fixture-checked in three independent parts (g2cell_demo.search: same first plan as the old
brute-force loop) with a broad-phase EnvChecker, and a plan cache (plan_cache, keyed by the layout:
assets/cell/build.py constants + cell_colliders.json) so a repeated run reuses the plan after one check.

    ros2 run g2cell_demo packing_demo --ros-args -p out_dir:=/g2cell_output/v2_work
"""
from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from g2cell_demo.demo_node import PickPlaceDemo
from g2cell_demo.drive_control import DriveParams
from g2cell_demo.drive_in import DriveIn
from g2cell_demo.env_check import EnvChecker, joint_origin, path_check
from g2cell_demo.search import LazySeq, memo_check, pick_place_search
from g2cell_motion.geometry import down_quat, end_link_for_tool
from g2cell_motion.plan_cache import PlanCache, find_build_py, layout_key
from g2cell_motion.robot import HEAD_CLEARANCE
from g2cell_motion.robot import GRIP_CLOSED, GRIP_OPEN, GRIPPERS

TUCK_DEG = {2: -90.0, 4: -110.0, 6: -30.0}     # scene_g2cell_mobile.yaml init (arms within |Y| 0.30 m)
HOME_DEG = {2: -45.0, 4: -75.0, 6: -55.0}      # scene_g2cell.yaml init
HEAD = {"idx11_head_joint1": 0.0, "idx13_head_joint3": 30.0}    # along the torso, down at the work (deg)
WAIST = "idx05_body_joint5"
HEAD_DRIVE = {"idx11_head_joint1": 0.0, "idx13_head_joint3": 8.0}
BAG_TOP = 0.87 + 0.07
CARTON_RIM = 0.82 + 0.25                        # carton_open top (flaps hang down outside)
# (name, (x, y)) on the pick table, cell frame (assets/cell/build.py LAYOUT["bags"])
BAGS = [("Bag_01", (0.05, 0.62)), ("Bag_02", (-0.25, 0.62))]
PACK = (0.60, 0.0)
# rack tote drop points (assets/cell/README.md Layout v2): 0.10 m above the tote opening along its normal,
# over the opening's front strip (28 cm down-slope of its centre, 2 cm inside the front rim) -- the lower
# tote is only open in front of the upper shelf
TOTE_DROP = {"L1_1": (0.0, -0.415, 1.121), "L2_1": (0.0, -0.415, 1.671)}
PLAN_VERSION = 2             # bump when the planner's semantics change (invalidates cached plans)
EXACT_BASE = (0.001, math.radians(0.1))    # reuse a cached path verbatim within 1 mm / 0.1 deg


class PackingDemo(PickPlaceDemo):
    def __init__(self):
        super().__init__()
        p = self.declare_parameter
        self.out_dir = p("out_dir", "/g2cell_output/v2_work").value
        self.goal = list(p("goal", [0.0, 0.0, 0.0]).value)
        self.do_drive = p("drive", True).value
        self.tote = p("tote", "L1_1").value
        self.arms = p("arms", "both").value          # both | left | right (debugging)
        self.drive = DriveIn(self, DriveParams(v_max=min(0.3, p("v_max", 0.25).value)))
        self.base = (0.0, 0.0, 0.0)   # base pose used to map cell -> base frame
        self._drift = {"max_m": 0.0, "max_yaw_deg": 0.0, "n": 0}
        self._hold = False
        self.hold_mode = p("hold_mode", "active").value
        self.env_margin = p("env_margin", 0.015).value   # m; exact mesh vertices vs collider boxes
        urdf = p("urdf", "/geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf").value
        colliders = p("colliders", "/geniesim_assets/scenes/g2cell/cell_colliders.json").value
        self.env = EnvChecker(urdf, colliders)
        self.env_log = []
        self.why = {}
        self.flip = p("flip_rad", 1.5).value
        self.plan_only = p("plan_only", False).value
        self.plan_retries = p("plan_retries", 1).value          # planning is deterministic: 1 is enough
        self.fallback_attempts = p("ik_fallback_attempts", 3).value   # /compute_ik pass if local IK finds no plan
        if p("local_ik", True).value:
            self.init_local_ik(urdf, confirm=p("ik_confirm", True).value, fallback_attempts=0)
        build_py = p("layout_file", "").value or find_build_py()
        self.layout = layout_key(build_py=build_py, colliders=colliders)
        self.plan_cache = PlanCache(p("plan_cache", "/g2cell_output/plan_cache/packing_demo.json").value or None,
                                    self.layout, version=PLAN_VERSION)
        if self.plan_cache.invalidated:
            self.get_logger().info("plan cache: layout or planner changed, cached plans dropped")
        self.plan_cache_reset()

    def plan_cache_reset(self):
        """Forget the in-run memo tables (IK chains, path checks); the persistent cache stays."""
        self._chains = {}
        self._check_memo = {}
        self.pstats = {"checks": 0, "check_s": 0.0}

    def move_joints(self, target: dict, min_t: float = 0.6):
        """Logged segment (wall time) so base motion on /odom can be matched to arm segments."""
        self._seg = getattr(self, "_seg", 0) + 1
        pose = self.drive.pose()
        self.get_logger().info(f"seg {self._seg} start t={time.time():.3f} base={tuple(round(c, 4) for c in pose) if pose else None} "
                               f"joints={sorted(target)[:1]}..{len(target)}")
        super().move_joints(target, min_t)

    # ---- cell -> base frame -----------------------------------------------------------------
    def to_robot(self, p):
        """Cell-frame fingertip point -> the same point in a 'cell frame' attached to the robot's
        actual base pose (what plan_chain/end_link_for_tool expect for a base at the origin)."""
        bx, by, byaw = self.base
        dx, dy = p[0] - bx, p[1] - by
        c, s = math.cos(-byaw), math.sin(-byaw)
        return (c * dx - s * dy, s * dx + c * dy, p[2])

    # ---- base hold + drift monitor --------------------------------------------------------------
    def _hold_loop(self):
        """Station-keeping while the arms work. Genie Sim's G2 wheels are velocity-driven with a weak
        gain (kd 10) and no brake, so arm reaction torques back-drive the free base (measured: 0.23 m /
        40 deg during one right-arm reach with a zero twist). hold_mode "active" (default) closes a
        holonomic P loop on /odom (GeneralServo accepts linear.y); "zero" only sends zero twists."""
        ref = self.drive.pose()
        k_xy, k_yaw, v_lim, w_lim = 2.0, 2.5, 0.08, 0.25
        while self._hold:
            pose = self.drive.pose()
            ex, ey = ref[0] - pose[0], ref[1] - pose[1]
            eyaw = math.atan2(math.sin(ref[2] - pose[2]), math.cos(ref[2] - pose[2]))
            if self.hold_mode == "active" and (math.hypot(ex, ey) > 0.002 or abs(eyaw) > math.radians(0.15)):
                c, s_ = math.cos(-pose[2]), math.sin(-pose[2])
                bx, by = c * ex - s_ * ey, s_ * ex + c * ey
                t = Twist()
                t.linear.x = max(-v_lim, min(v_lim, k_xy * bx))
                t.linear.y = max(-v_lim, min(v_lim, k_xy * by))
                t.angular.z = max(-w_lim, min(w_lim, k_yaw * eyaw))
                self.drive.pub.publish(t)
            else:
                self.drive.send(0.0, 0.0)
            d = math.hypot(ex, ey)
            self._drift["max_m"] = max(self._drift["max_m"], d)
            self._drift["max_yaw_deg"] = max(self._drift["max_yaw_deg"], abs(math.degrees(eyaw)))
            self._drift["n"] += 1
            self._drift["ref"], self._drift["last"] = ref, pose
            time.sleep(0.02)

    # ---- waist-first planning (user rule 2026-10-11) -----------------------------------------------
    # Turn with the waist (idx05_body_joint5, yaw about z) so the target is within ~+-30 deg of the torso's
    # forward direction, then solve the 7-DoF arm IK seeded from the arm's home pose (the arm groups exclude
    # the waist, so /compute_ik keeps it at the seeded value). Reject backward twists (joint1 > 90 deg from
    # home) and wrist/elbow flips (joint 3/5/7 jumps > 1.2 rad between consecutive waypoints). Every joint
    # path (waist + arm interpolated together) is checked against the fixtures (env_check, >= env_margin)
    # and the head (>= 0.20 m) with the waist and head at their commanded values.
    ORIENTS = [(0, 0), (0, 20), (0, 40), (90, 0), (-90, 0), (45, 0), (-45, 0), (90, 20), (-90, 20), (45, 40), (-45, 40),
               (90, 40), (-90, 40), (30, 40), (-30, 40), (30, 20), (-30, 20), (180, 0)]

    def waist_candidates(self, tool_cell):
        x, y, _ = self.to_robot(tool_cell)
        ax = self.waist_axis
        th = math.degrees(math.atan2(y - ax[1], x - ax[0]))
        out = []
        # target within +-30 deg of the torso's forward first; +-45 only as a last resort (the left arm reaches
        # a table bag best with the bag ~45 deg to the torso's right of forward, i.e. in front of the left arm)
        for off in (0, -15, 15, -30, 30, -45, 45):
            w = max(-150.0, min(150.0, th + off))
            if all(abs(w - o) > 1 for o in out):
                out.append(w)
        return out

    def chains(self, arm, tools, waist_deg, home, orients=None):
        """Memoised ik_chain: each (arm, points, waist, base, orientations) chain is solved once per run."""
        key = (arm, tuple(map(tuple, tools)), round(waist_deg, 9), tuple(self.base), tuple(orients or ()))
        if key not in self._chains:
            self._chains[key] = LazySeq(self.ik_chain(arm, tools, waist_deg, home, orients))
        return self._chains[key]

    def ik_chain(self, arm, tools, waist_deg, home, orients=None):
        """[(orient, [arm sols])] in priority order for fingertip points (cell frame) at a fixed waist."""
        w = math.radians(waist_deg)
        names = self.arm_joint_names(arm)
        for rel, tilt in (orients or self.ORIENTS):
            q = down_quat(math.radians(waist_deg + rel), math.radians(tilt))
            seed, sols = dict(home), []
            for t in tools:
                sol = self.solve(arm, end_link_for_tool(self.to_robot(t), q), q, seed={WAIST: w, **seed}, attempts=12)
                if sol is None:
                    self.why[arm + ":ik"] = self.why.get(arm + ":ik", 0) + 1
                    break
                sol = {n: sol[n] for n in names}
                if abs(sol[names[0]] - home[names[0]]) > math.pi / 2:
                    self.why[arm + ":twist"] = self.why.get(arm + ":twist", 0) + 1
                    break                                   # backward twist
                if any(abs(sol[names[k]] - seed[names[k]]) > self.flip for k in (2, 4, 6)):
                    self.why[arm + ":flip"] = self.why.get(arm + ":flip", 0) + 1
                    break                                   # elbow/wrist flip
                sols.append(sol)
                seed = sol
            if len(sols) == len(tools):
                yield (rel, tilt), sols

    def worst_segment(self, arm, path):
        json.dump({"arm": arm, "path": path, "fixed": self.fixed, "base": self.base},
                  open(f"{self.out_dir}/rejected_path.json", "w"))
        segs = [path_check(self.env, [a, b], arm, base=self.base, samples=8, fixed=self.fixed)
                for a, b in zip(path, path[1:])]
        i = min(range(len(segs)), key=lambda k: segs[k][0])
        return i, round(segs[i][0], 4), segs[i][1]

    def check_path(self, arm, path):
        """Active arm vs fixtures + head, and the idle arm (held in tuck) vs fixtures while the waist turns.
        Stops at the first sample below a limit (the path is rejected either way; the reported clearances of
        a rejected path are then the minima up to that sample). Memoised per path within a run."""
        k = (arm, json.dumps(path, sort_keys=True))
        if k in self._check_memo:
            return self._check_memo[k]
        t0 = time.perf_counter()
        stop = (self.env_margin, HEAD_CLEARANCE)
        env, name, head = path_check(self.env, path, arm, base=self.base, samples=8, fixed=self.fixed,
                                     stop_below=stop)
        env2, name2 = math.inf, ""
        if env >= self.env_margin and head >= HEAD_CLEARANCE:
            other = "right" if arm == "left" else "left"
            env2, name2, _ = path_check(self.env, path, other, base=self.base, samples=6, fixed=self.fixed,
                                        stop_below=(self.env_margin, -math.inf))
        ok = env >= self.env_margin and env2 >= self.env_margin and head >= HEAD_CLEARANCE
        r = ok, {"fixture_clearance_m": round(env, 4), "closest": name, "head_clearance_m": round(head, 4),
                 "idle_arm_clearance_m": round(env2, 4) if env2 < math.inf else None, "idle_closest": name2}
        self.pstats["checks"] += 1
        self.pstats["check_s"] += time.perf_counter() - t0
        self._check_memo[k] = r
        return r

    def plan_left(self, home, hint=None):
        """Pick a table bag, place it over the carton. Same candidate order as the original nested loop
        (bag, pick waist, pick orientation, place waist, place orientation), pruned per path part
        (g2cell_demo.search). hint (a cached result) is tried first, alone."""
        arm = "left"
        R = self.rest[arm]
        A = lambda wp, pick: ([{WAIST: 0.0, **R}, {WAIST: math.radians(wp), **R}] +  # noqa: E731
                              [{WAIST: math.radians(wp), **s} for s in pick] + [{WAIST: math.radians(wp), **R}])
        B = lambda wp, wd: [{WAIST: math.radians(wp), **R}, {WAIST: math.radians(wd), **R}]  # noqa: E731
        C = lambda wd, plc: ([{WAIST: math.radians(wd), **R}] + [{WAIST: math.radians(wd), **s} for s in plc] +  # noqa: E731
                             [{WAIST: math.radians(wd), **R}, {WAIST: 0.0, **R}])

        def log(m):
            self.get_logger().info(f"[left] {m}")
        for name, (x, y) in BAGS:
            if hint and hint.get("bag") != name:
                continue
            # lift high enough to carry the bag over the upstream carton's walls (top 1.07)
            # grasp height: fingertips 20 mm ABOVE the bag top. The rigid placeholder bag (0.21 m wide) is
            # wider than the gripper opening, so closing at the bag pressed the fingers into it and pushed
            # the free base 3.2 cm / 2.4 deg (take 1). Bags never move on camera anyway (render limit).
            picks = [(x, y, BAG_TOP + 0.20), (x, y, BAG_TOP + 0.02), (x, y, BAG_TOP + 0.25)]
            px, py = PACK
            # release 0.09 m over the open carton's rim (gripper clear of its walls), retract up
            place = [(px, py, CARTON_RIM + 0.20), (px, py, CARTON_RIM + 0.09), (px, py, CARTON_RIM + 0.22)]
            if hint:
                o1, o2 = [tuple(o) for o in hint["orient"]]
                found = pick_place_search([hint["waist_pick_deg"]],
                                          lambda wp: self.chains(arm, picks, wp, home, [o1]),
                                          [hint["waist_place_deg"]],
                                          lambda wd: self.chains(arm, place, wd, home, [o2]),
                                          A, B, C, lambda pth: self.check_path(arm, pth), log)
            else:
                found = pick_place_search(self.waist_candidates(picks[1]),
                                          lambda wp: self.chains(arm, picks, wp, home),
                                          self.waist_candidates(place[1]),
                                          lambda wd: self.chains(arm, place, wd, home),
                                          A, B, C, lambda pth: self.check_path(arm, pth), log)
            if found:
                wp, wd = found["wp"], found["wd"]
                path = A(wp, found["pick"]) + B(wp, wd)[1:] + C(wd, found["plc"])[1:]
                log(f"{name} waist {wp:+.0f}/{wd:+.0f} orient {found['o1']}/{found['o2']}: OK {found['info']}")
                return {"bag": name, "waist_pick_deg": wp, "waist_place_deg": wd,
                        "orient": [found["o1"], found["o2"]], **found["info"]}, path
        return None, None

    def plan_right(self, home, hint=None):
        arm = "right"
        x, y, z = TOTE_DROP[self.tote]
        waists = [hint["waist_drop_deg"]] if hint else self.waist_candidates((x, y, z))
        orients = [tuple(hint["orient"])] if hint else None
        for wd in waists:
            w = math.radians(wd)
            # approach from straight in front of the turned torso, 0.15 m back from the drop point
            ax = self.waist_axis
            dx, dy = (x - ax[0], y - ax[1])
            n = math.hypot(dx, dy)
            pre = (x - 0.15 * dx / n, y - 0.15 * dy / n, z + 0.03)
            pts = [pre, (x, y, z), pre]
            for o, sols in self.chains(arm, pts, wd, home, orients):
                R = self.rest[arm]
                path = [{WAIST: 0.0, **R}, {WAIST: w, **R}] + [{WAIST: w, **s} for s in sols] + \
                       [{WAIST: w, **R}, {WAIST: 0.0, **R}]
                ok, info = self.check_path(arm, path)
                self.get_logger().info(f"[right] tote {self.tote} waist {wd:+.0f} orient {o}: "
                                       f"{'OK' if ok else 'rejected'} {info}")
                if ok:
                    return {"tote": self.tote, "waist_drop_deg": wd, "orient": o, **info}, path
        return None, None

    # ---- plan cache + deterministic planning budget ------------------------------------------------
    def plan_key(self, arm):
        """Everything a plan depends on except the base pose (stored with the plan, see plan_arm)."""
        fixed = {k: round(v, 4) for k, v in sorted(self.fixed.items()) if "arm_" in k or "head" in k or "body" in k}
        return {"arm": arm, "bags": BAGS, "pack": PACK, "bag_top": BAG_TOP, "rim": CARTON_RIM,
                "tote": self.tote if arm == "right" else None, "drop": TOTE_DROP[self.tote] if arm == "right" else None,
                "orients": self.ORIENTS, "fixed": fixed, "margin": self.env_margin, "flip": self.flip,
                "head": HEAD_CLEARANCE, "waist_axis": [round(c, 4) for c in self.waist_axis]}

    def plan_arm(self, arm, home):
        """(result, path, how) with how in hit | hint | search | fallback | failed.
        hit: cached path at (almost) the same base, re-checked once against the fixtures.
        hint: base moved a little: re-solve only the cached choice (waist, orientation) at this base.
        search: full deterministic search, local IK + MoveIt confirmation.
        fallback: the search again with up to ik_fallback_attempts /compute_ik calls per waypoint."""
        plan = self.plan_left if arm == "left" else self.plan_right
        key = self.plan_key(arm)
        hit = self.plan_cache.get(key)
        if hit:
            db = math.hypot(self.base[0] - hit["base"][0], self.base[1] - hit["base"][1])
            dyaw = abs(math.atan2(math.sin(self.base[2] - hit["base"][2]), math.cos(self.base[2] - hit["base"][2])))
            if db <= EXACT_BASE[0] and dyaw <= EXACT_BASE[1]:
                ok, info = self.check_path(arm, hit["path"])
                if ok:
                    return {**hit["result"], **info}, hit["path"], "hit"
            res, path = plan(home, hint=hit["result"])
            if path:
                self._remember(key, res, path)
                return res, path, "hint"
        res = path = None
        for _ in range(max(1, self.plan_retries)):
            res, path = plan(home)
            if path:
                break
        how = "search"
        if not path and getattr(self, "local_ik", None) and self.fallback_attempts > 0:
            self.ik_fallback_attempts = self.fallback_attempts      # bounded, deterministic MoveIt pass
            self._chains = {}
            try:
                res, path = plan(home)
            finally:
                self.ik_fallback_attempts = 0
            how = "fallback"
        if path:
            self._remember(key, res, path)
        return res, path, how if path else "failed"

    def _remember(self, key, res, path):
        self.plan_cache.put(key, {"result": res, "path": path, "base": list(self.base)})
        self.plan_cache.save()

    def joints_deg(self, arm, table):
        names = self.arm_joint_names(arm)
        return {n: math.radians(table.get(int(n[-1]), 0.0)) for n in names}

    def exec_left(self, path):
        rest0, turn1, pre, grasp, lift, tuck1, turn2, above, drop, retract, tuck2, rest1 = path
        arm = "left"
        self.move_joints(turn1, min_t=2.5)          # waist turns to the table, arm tucked
        threading.Thread(target=self.gripper, args=(arm, GRIP_OPEN, 1.0), daemon=True).start()
        self.move_joints(pre, min_t=2.0)
        self.move_joints(grasp, min_t=1.5)
        self.gripper(arm, GRIP_CLOSED)
        self.pause(0.3)
        self.move_joints(lift, min_t=1.2)
        self.move_joints(tuck1, min_t=1.5)          # bag held close
        self.move_joints(turn2, min_t=2.5)          # waist back toward the conveyor
        self.move_joints(above, min_t=1.5)
        self.move_joints(drop, min_t=1.2)
        self.gripper(arm, GRIP_OPEN)
        self.pause(0.5)
        self.move_joints(retract, min_t=1.0)
        threading.Thread(target=self.gripper, args=(arm, GRIP_CLOSED, 1.0), daemon=True).start()
        self.move_joints(tuck2, min_t=1.5)
        self.move_joints(rest1, min_t=1.5)

    def exec_right(self, path):
        rest0, turn, pre, drop, retract, tuck, rest1 = path
        arm = "right"
        self.move_joints(turn, min_t=3.0)           # waist turns to the rack, arm tucked
        self.move_joints(pre, min_t=2.0)
        self.move_joints(drop, min_t=1.5)
        self.gripper(arm, GRIP_OPEN)
        self.pause(0.6)
        self.move_joints(retract, min_t=1.2)
        threading.Thread(target=self.gripper, args=(arm, GRIP_CLOSED, 1.0), daemon=True).start()
        self.move_joints(tuck, min_t=1.5)
        self.move_joints(rest1, min_t=2.5)

    # ---- main --------------------------------------------------------------------------------------
    def run(self) -> bool:
        if not (self.ik.wait_for_service(timeout_sec=90.0) and self.fk.wait_for_service(timeout_sec=10.0)):
            self.get_logger().error("/compute_ik not available: start the mobile scene with WITH_MOVEIT=1")
            return False
        while rclpy.ok() and (self.joint_state() is None or self.drive.pose() is None):
            time.sleep(0.1)
        meas = self.measured()
        for arm in ("left", "right"):
            names = self.arm_joint_names(arm)
            self._set({n: meas[n] for n in names})
            self._set({GRIPPERS[arm]: meas.get(GRIPPERS[arm], GRIP_CLOSED)})
        self._set({k: meas.get(k, 0.0) for k in HEAD})
        self._set({WAIST: meas.get(WAIST, 0.0)})
        homes = {a: self.joints_deg(a, HOME_DEG) for a in ("left", "right")}
        tucks = {a: self.joints_deg(a, TUCK_DEG) for a in ("left", "right")}
        result = {"start_pose": self.drive.pose()}
        self.state_pub.publish(String(data="start"))
        self.pause(self.declare_parameter("start_delay", 2.0).value)
        # hold tuck + head forward while driving
        self.move_joints({**tucks["left"], **tucks["right"],
                          **{k: math.radians(v) for k, v in HEAD_DRIVE.items()}}, min_t=1.0)
        if self.do_drive:
            res = self.drive.run(self.goal)
            self.drive.write_log(f"{self.out_dir}/drive_in.csv")
            result["drive"] = res
            if not res["ok"]:
                self.get_logger().error("drive-in failed; arm work skipped")
                self.state_pub.publish(String(data="done"))
                json.dump(result, open(f"{self.out_dir}/packing_demo.json", "w"), indent=1, default=str)
                return False
        self.base = self.drive.pose()
        result["base_for_ik"] = self.base
        self._hold = True
        threading.Thread(target=self._hold_loop, daemon=True).start()
        # Arms rest in the drive tuck (hands forward, within the chassis width, above the carton tops) and
        # start/end every task there; the home pose only seeds IK. The head looks ahead/down along the
        # torso: the waist does the turning, so the camera follows the work.
        self.rest = tucks
        self.move_joints({WAIST: 0.0, **{k: math.radians(v) for k, v in HEAD.items()}}, min_t=2.0)
        self.pause(0.5)
        self.get_logger().info("planning (waist-first IK, base frame from /odom) ...")
        self.state_pub.publish(String(data="pause"))     # cut planning time out of the recording
        self.fixed = dict(self.measured())      # tuck arms, head, torso as commanded now
        self.fixed[WAIST] = 0.0
        self.waist_axis = tuple(joint_origin(self.env, WAIST, self.fixed)[:2])
        lres = lpath = rres = rpath = None
        timing = {}
        self.plan_cache_reset()
        if self.arms in ("both", "left"):
            t0 = time.perf_counter()
            lres, lpath, how = self.plan_arm("left", homes["left"])
            timing["left"] = {"s": round(time.perf_counter() - t0, 2), "how": how}
        if self.arms in ("both", "right"):
            t0 = time.perf_counter()
            rres, rpath, how = self.plan_arm("right", homes["right"])
            timing["right"] = {"s": round(time.perf_counter() - t0, 2), "how": how}
        timing["ik"] = dict(getattr(self, "ik_stats", {}) or {})
        timing["checks"] = dict(self.pstats)
        result["left"], result["right"], result["planning"] = lres, rres, timing
        self.get_logger().info(f"plans: left={lres} right={rres}; IK rejections {self.why}; planning {timing}")
        if self.plan_only:
            lpath = rpath = None
        lsols, rsols = lpath, rpath
        self.state_pub.publish(String(data="resume"))
        if lpath:
            self.exec_left(lpath)
        if rpath:
            self.exec_right(rpath)
        self.pause(1.0)
        self._hold = False
        result["base_drift_during_arm_work"] = dict(self._drift)
        result["final_pose"] = self.drive.pose()
        self.state_pub.publish(String(data="done"))
        self.get_logger().info(f"done: base drift during arm work {self._drift['max_m'] * 100:.2f} cm, "
                               f"{self._drift['max_yaw_deg']:.2f} deg")
        json.dump(result, open(f"{self.out_dir}/packing_demo.json", "w"), indent=1, default=str)
        return bool(lsols and rsols)


def main():
    rclpy.init()
    node = PackingDemo()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node._hold = False
        node.drive.stop(5)
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
