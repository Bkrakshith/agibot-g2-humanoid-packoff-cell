"""``g2cell_engine_isaacsim.py``: Genie Sim's Isaac Sim engine, unchanged, plus the object pose relay hook.

Loads the installed upstream ``genie_sim_engine_isaacsim.py`` as a module (this bootstraps the
SimulationApp exactly as upstream does), wraps ``EngineSession.startup`` so that, once the stage is built,
one ``post_step_hook`` (object_relay.ObjectPoseRelay) is appended, then calls upstream ``main()``.
All args (--ros-args, params files) pass through untouched. See object_relay.py for the why/how.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _upstream_script() -> str:
    from ament_index_python.packages import get_package_prefix  # noqa: PLC0415
    return os.path.join(get_package_prefix("genie_sim_engine"), "lib", "genie_sim_engine",
                        "genie_sim_engine_isaacsim.py")


def _install_hook() -> None:
    from common import session as S  # noqa: PLC0415  (upstream, on sys.path after the module load)
    from g2cell_bringup import object_relay as R  # noqa: PLC0415

    orig_startup = S.EngineSession.startup

    def startup(self, headless):
        orig_startup(self, headless)
        log = self.logger
        try:
            hz = float(os.environ.get("G2CELL_OBJ_RELAY_HZ", "30"))
            key = float(os.environ.get("G2CELL_OBJ_RELAY_KEYFRAME_S", "2"))
            kin = os.environ.get("G2CELL_OBJ_RELAY_KINEMATIC", "0") == "1"
            stage = self.sim.stage
            paths = R.discover_bodies(stage, self.sim.robot_prefix, include_kinematic=kin)
            if not paths:
                log.info("[obj_relay] no object rigid bodies on the stage; relay off")
                return
            pokes = R.PokeQueue()
            pub = R.RclpyTfPublisher(pokes=pokes, allowed=paths, log=log)
            # G2CELL_OBJ_RELAY_PUBLISH=0: pokes still work but nothing is sent to the renderer (A/B check only)
            sink = pub if os.environ.get("G2CELL_OBJ_RELAY_PUBLISH", "1") != "0" else _NullSink()
            relay = R.ObjectPoseRelay(R.UsdPoseSource(stage, paths), sink, physics_hz=self.physics_hz,
                                      rate_hz=hz, keyframe_s=key, pokes=pokes)
            if os.environ.get("G2CELL_GRASP_ASSIST", "1") == "1":
                _install_grasp_assist(self, R, stage, paths, relay.div, log,    # before the relay: same-tick pose
                                      node=getattr(pub, "_node", None))
            self.post_step_hooks.append(relay)
            self.post_step_hooks.append(_stats_logger(relay, log, self.physics_hz))
            log.info(f"[obj_relay] relaying {len(paths)} bodies to /tf_render every {relay.div} ticks "
                     f"({self.physics_hz / relay.div:.0f} Hz), keyframe {key:g} s: {', '.join(paths)}")
        except Exception as exc:  # never take the sim down for a render nicety
            log.warn(f"[obj_relay] disabled: {exc}")

    S.EngineSession.startup = startup


GRIPPER_JOINTS = {"left": "idx31_gripper_l_inner_joint1", "right": "idx71_gripper_r_inner_joint1"}


def _install_grasp_assist(session, R, stage, paths, div, log, node=None) -> None:
    """Sticky grasp for the rigid bags (object_relay.GraspAssist): G2CELL_GRASP_ASSIST=0 disables it."""
    try:
        from pxr import UsdGeom  # noqa: PLC0415
        bags = [p for p in paths if p.startswith("/World/Bags/")]
        links = R.find_gripper_links(stage, session.sim.robot_prefix)
        names = list(session.sim.joint_names)
        idx = {a: names.index(j) for a, j in GRIPPER_JOINTS.items() if j in names and a in links}
        if not bags or not idx:
            log.info(f"[grasp_assist] off: bags={bags} gripper links={links} joints={list(idx)}")
            return

        # the COMMANDED gripper value decides grasp/release: fingers pressed on the rigid bag top cannot close
        # until the lift starts (measured leader joint stays at -0.795), so the measurement alone attaches late
        tracker = R.CommandTracker({a: GRIPPER_JOINTS[a] for a in idx})
        if node is not None:
            from rclpy.qos import qos_profile_sensor_data  # noqa: PLC0415
            from sensor_msgs.msg import JointState  # noqa: PLC0415
            node.create_subscription(JointState, "/joint_command",
                                     lambda m: tracker.update(m.name, m.position), qos_profile_sensor_data)

        def read_grip():
            pos = session.sim.get_joint_states()[0]
            return tracker.merge({a: float(pos[i]) for a, i in idx.items()})
        assist = R.GraspAssist(bags, list(idx), tool_offset=(0.0, 0.0, 0.12))
        hook = R.GraspAssistHook(assist, R.UsdGraspIO(stage, {a: links[a] for a in idx}), read_grip, div=div, log=log)
        session.post_step_hooks.append(hook)
        ops = [o.GetOpName() for o in UsdGeom.Xformable(stage.GetPrimAtPath(bags[0])).GetOrderedXformOps()]
        log.info(f"[grasp_assist] on: {len(bags)} bags, grippers {sorted(idx)}, detection every {div} ticks; "
                 f"{bags[0]} xform ops {ops}")
    except Exception as exc:  # noqa: BLE001
        log.warn(f"[grasp_assist] disabled: {exc}")


class _NullSink:
    def send(self, stamp, items):
        pass


def _stats_logger(relay, log, physics_hz: float, every_s: float = 10.0):
    n = max(1, int(round(every_s * physics_hz)))
    st = {"tick": 0, "prev": relay.stats()}

    def hook(_sim_time):
        st["tick"] += 1
        if st["tick"] % n:
            return
        s, p = relay.stats(), st["prev"]
        ds = s["samples"] - p["samples"]
        mean = (s["cost_ms_sum"] - p["cost_ms_sum"]) / max(1, ds)
        log.info(f"[obj_relay] {every_s:g} s: samples={ds} msgs={s['published_msgs'] - p['published_msgs']} "
                 f"tfs={s['published_tfs'] - p['published_tfs']} errors={s['errors'] - p['errors']} "
                 f"cost/sample mean={mean:.3f} ms max(all-time)={s['cost_ms_max']:.3f} ms "
                 f"=> {mean * ds / every_s / 10.0:.3f} % of one core")
        st["prev"] = s
    return hook


def main() -> int:
    path = _upstream_script()
    spec = importlib.util.spec_from_file_location("genie_sim_engine_isaacsim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genie_sim_engine_isaacsim"] = mod
    spec.loader.exec_module(mod)   # upstream module body: sys.path + SimulationApp bootstrap
    _install_hook()
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
