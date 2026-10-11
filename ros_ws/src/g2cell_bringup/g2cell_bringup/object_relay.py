"""Object pose relay: make non-robot rigid bodies (bags, cartons, totes) render where PhysX puts them.

Why: Genie Sim's engine publishes /tf_render only for the robot's rigid bodies (kit/stage.py
``_collect_body_paths`` walks /<robot_prefix> only), so the OVRTX render node never moves a bag, carton or
tote: a grasped bag stays drawn on the table. The render stage is the same cell.usda the physics stage opens
(manifest ``scene_usda``), so every object prim exists there under the same path; the render node applies a
/tf_render ``child_frame_id`` that starts with "/" verbatim as that prim's LOCAL xform (render_node.cpp
``on_tf_render`` -> ``ovrtx_set_xform_mat``). Object poses only exist inside the engine process.

How: ``engine_objrelay`` (console script ``g2cell_engine_isaacsim.py``) runs the upstream engine unchanged and appends one
``post_step_hook`` (the engine's supported extension point, common/session.py) that:
  * samples only the top-level dynamic rigid bodies outside /<robot_prefix> (14 in the cell scene),
  * every ``tick_divider(physics_hz, rate_hz)`` physics ticks (default 30 Hz -> every 4th tick at 100 Hz,
    i.e. 25 Hz <= the 30 Hz render rate), reading the USD local xform PhysX writes back after fetch_results,
  * sends only bodies that moved > 0.5 mm / 0.2 deg since their last published pose (plus a full keyframe
    every 2 s, so a render node that starts or restarts late still converges),
  * publishes them as ONE TFMessage on /tf_render from a background rclpy thread (latest-wins), stamped with
    wall time and frame "map" like the engine's own /tf_render.
It also listens on /g2cell/object_poke (std_msgs/String JSON, relayed bodies only) and applies the request on
the physics thread between steps: {"path": "/World/Bags/Bag_01", "xyz"|"dxyz": [..] (parent frame, m),
"vel": [..] (m/s)} -> teleport and/or set linear velocity. Used for the relay checks and episode resets.
The hook's cost is reported by the engine itself ("Extras" line of the 1 Hz physics stats) and by our own
10 s log line "[obj_relay] ...".

Env (read by engine_objrelay): G2CELL_OBJ_RELAY_HZ (30), G2CELL_OBJ_RELAY_KEYFRAME_S (2),
G2CELL_OBJ_RELAY_KINEMATIC (0: kinematic bodies such as the conveyor StopGate are not relayed).

CLI: python3 -m g2cell_bringup.object_relay make-launcher IN.yaml OUT.yaml
     (copy of a Genie Sim launcher YAML whose isaac engine runs g2cell_bringup/g2cell_engine_isaacsim.py)
"""
from __future__ import annotations

import copy
import math
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]  # (xyz, wxyz)

ENGINE_PACKAGE = "g2cell_bringup"
ENGINE_EXECUTABLE = "g2cell_engine_isaacsim.py"


# ---------------------------------------------------------------------------------------------- logic

def pose_changed(prev: Pose, cur: Pose, lin_tol: float, ang_tol: float) -> bool:
    (p0, q0), (p1, q1) = prev, cur
    if math.dist(p0, p1) > lin_tol:
        return True
    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    ang = 2.0 * math.acos(min(1.0, dot))
    return ang > ang_tol


class ChangeFilter:
    """Return the subset of poses to publish: moved since last publish, or all of them on a keyframe."""

    def __init__(self, keyframe_s: float = 2.0, lin_tol: float = 5e-4, ang_tol: float = math.radians(0.2)):
        self.keyframe_s = keyframe_s
        self.lin_tol = lin_tol
        self.ang_tol = ang_tol
        self._last: Dict[str, Pose] = {}
        self._last_key: Optional[float] = None

    def update(self, t: float, poses: Dict[str, Pose]) -> Dict[str, Pose]:
        key = self._last_key is None or (self.keyframe_s > 0 and t - self._last_key >= self.keyframe_s)
        if key:
            out = dict(poses)
            self._last_key = t
        else:
            out = {k: v for k, v in poses.items()
                   if k not in self._last or pose_changed(self._last[k], v, self.lin_tol, self.ang_tol)}
        self._last.update(out)
        return out


def select_object_bodies(entries: Iterable[Tuple[str, bool, bool, bool]], robot_prefix: str,
                         include_kinematic: bool = False) -> List[str]:
    """entries: (path, has RigidBodyAPI, kinematicEnabled, rigidBodyEnabled) in traversal order."""
    robot_root = "/" + robot_prefix.strip("/")
    out: List[str] = []
    for path, rigid, kinematic, enabled in entries:
        if not rigid or not enabled or (kinematic and not include_kinematic):
            continue
        if path == robot_root or path.startswith(robot_root + "/"):
            continue
        if any(path.startswith(p + "/") for p in out):
            continue  # nested body: moved with its top-level body
        out.append(path)
    return out


def tick_divider(physics_hz: float, rate_hz: float) -> int:
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    return max(1, math.ceil(physics_hz / rate_hz - 1e-9))


def patch_launcher(cfg: dict, package: str = ENGINE_PACKAGE, executable: str = ENGINE_EXECUTABLE) -> dict:
    out = copy.deepcopy(cfg)
    phys = out["launcher"]["physics"]
    name = phys["engine"]
    eng = phys["engines"][name]
    if eng.get("executable") != "genie_sim_engine_isaacsim.py":
        raise ValueError(f"object relay needs the Isaac Sim engine, launcher uses {eng.get('executable')!r}")
    eng["package"], eng["executable"] = package, executable
    return out


class Poke(tuple):
    """(path, xyz | None, dxyz | None, vel | None) plus .yaw_deg (absolute yaw about +Z, parent frame) and .stop
    (zero linear + angular velocity). Compares and unpacks as the 4-tuple."""

    def __new__(cls, path, xyz=None, dxyz=None, vel=None, yaw_deg=None, stop=False):
        t = super().__new__(cls, (path, xyz, dxyz, vel))
        t.yaw_deg, t.stop = yaw_deg, bool(stop)
        return t

    path = property(lambda self: self[0])
    xyz = property(lambda self: self[1])
    dxyz = property(lambda self: self[2])
    vel = property(lambda self: self[3])


def _vec3(v, name):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{name} must be [x, y, z]")
    return tuple(float(c) for c in v)


def parse_poke(text: str, allowed) -> Poke:
    """JSON {"path", "xyz"|"dxyz" (parent-frame position / offset, m), "vel" (world linear velocity, m/s),
    "yaw_deg" (absolute yaw, parent frame; roll/pitch reset to 0), "stop" (zero linear + angular velocity)}."""
    import json  # noqa: PLC0415
    try:
        d = json.loads(text)
    except Exception as exc:
        raise ValueError(f"not JSON: {exc}") from None
    if not isinstance(d, dict) or d.get("path") not in allowed:
        raise ValueError(f"path must be one of the relayed bodies, got {d.get('path') if isinstance(d, dict) else d!r}")
    if "xyz" in d and "dxyz" in d:
        raise ValueError("give xyz or dxyz, not both")
    xyz = _vec3(d["xyz"], "xyz") if "xyz" in d else None
    dxyz = _vec3(d["dxyz"], "dxyz") if "dxyz" in d else None
    vel = _vec3(d["vel"], "vel") if "vel" in d else None
    yaw = None
    if "yaw_deg" in d:
        try:
            yaw = float(d["yaw_deg"])
        except (TypeError, ValueError):
            raise ValueError("yaw_deg must be a number") from None
    stop = bool(d.get("stop", False))
    if xyz is None and dxyz is None and vel is None and yaw is None and not stop:
        raise ValueError("nothing to do: give xyz, dxyz, vel, yaw_deg and/or stop")
    return Poke(d["path"], xyz, dxyz, vel, yaw, stop)


# --------------------------------------------------------------------------------------- grasp assist

def pose_matrix(xyz, yaw_deg: float = 0.0):
    """4x4 (numpy) from a position and a yaw about +Z."""
    import numpy as np  # noqa: PLC0415
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = xyz
    return T


class GraspAssist:
    """Sticky grasp for the rigid placeholder bags (Stage 4 replaces them with deformable bags).

    Why: the omnipicker cannot physically hold a 0.21 x 0.30 x 0.07 m rigid box (PickBag grips 1 cm below its
    top), so a scripted or teleop grasp never lifted the bag and the recorded images showed the bag staying on
    the table. Like most scripted data generators, the engine makes the grasp stick instead:
      * attach when a gripper starts closing (leader joint rises above `close_above`, was below it) while its
        fingertip point lies inside a bag's footprint (+`slack`) between the bag bottom and top + `slack`;
        the nearest such bag wins; the bag-in-link transform at that moment is kept;
      * while held, the bag follows the gripper link with that transform (targets(), applied every tick);
      * detach when the gripper opens past `open_below` (hysteresis), and the bag drops under physics.
    Inputs are injected (gripper joint values, link and body world 4x4 matrices) so this is pure numpy.
    bags: body prim paths; arms: arm names; tool_offset: fingertip in the gripper link frame (m)."""

    def __init__(self, bags, arms, tool_offset=(0.0, 0.0, 0.12), half_xy=(0.105, 0.15), height=0.07,
                 slack=0.04, close_above=-0.30, open_below=-0.55):
        import numpy as np  # noqa: PLC0415
        self.np = np
        self.bags, self.arms = list(bags), list(arms)
        self.tip = np.array([*tool_offset, 1.0])
        self.half_xy, self.height, self.slack = half_xy, height, slack
        self.close_above, self.open_below = close_above, open_below
        self._closed = {a: None for a in self.arms}     # last closed state per arm (None = unknown)
        self._rel = {}                                  # arm -> (bag, bag-in-link 4x4)
        self.misses = []                                # (arm, tip xyz, nearest bag, tip in that bag's frame)

    @property
    def held(self):
        return {a: b for a, (b, _) in self._rel.items()}

    def _inside(self, tip_world, body_T):
        local = self.np.linalg.inv(body_T) @ tip_world
        hx, hy = self.half_xy
        s = self.slack
        if abs(local[0]) <= hx + s and abs(local[1]) <= hy + s and -s <= local[2] <= self.height + s:
            return float(local[0] ** 2 + local[1] ** 2)
        return None

    def step(self, grip_q, link_world, body_world):
        """grip_q: {arm: leader joint (rad) | None}; link_world(arm) / body_world(path) -> 4x4.
        Returns events [("attach", arm, bag) | ("detach", arm, bag)]."""
        ev = []
        for a in self.arms:
            q = grip_q.get(a)
            if q is None:
                continue
            was = self._closed[a]
            if a in self._rel:
                if q < self.open_below:
                    ev.append(("detach", a, self._rel.pop(a)[0]))
                    self._closed[a] = False
                continue
            closed = q > self.close_above
            self._closed[a] = closed
            if not closed or was is not False:
                continue                     # only a fresh close (open -> closing) attaches
            L = link_world(a)
            tip = L @ self.tip
            taken = {b for b, _ in self._rel.values()}
            best = None
            for b in self.bags:
                if b in taken:
                    continue
                B = body_world(b)
                d = self._inside(tip, B)
                if d is not None and (best is None or d < best[0]):
                    best = (d, b, B)
            if best is not None:
                self._rel[a] = (best[1], self.np.linalg.inv(L) @ best[2])
                ev.append(("attach", a, best[1]))
            else:
                near = min(((float(self.np.linalg.norm(body_world(b)[:3, 3] - tip[:3])), b) for b in self.bags),
                           default=(None, ""))
                loc = (self.np.linalg.inv(body_world(near[1])) @ tip)[:3] if near[1] else None
                self.misses.append((a, tuple(round(float(v), 3) for v in tip[:3]), near[1],
                                    None if loc is None else tuple(round(float(v), 3) for v in loc)))
                ev.append(("miss", a, near[1]))
        return ev

    def targets(self, link_world):
        """{bag: world 4x4} for every held bag."""
        return {b: link_world(a) @ rel for a, (b, rel) in self._rel.items()}


class CommandTracker:
    """Latest commanded gripper leader values from /joint_command (any publisher: motion_server, teleop).
    merge(measured) -> the commanded value where one was seen, else the measured one."""

    def __init__(self, joints: Dict[str, str]):
        self.joints = dict(joints)
        self._by_joint = {j: a for a, j in self.joints.items()}
        self._cmd: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, names, positions) -> None:
        with self._lock:
            for n, v in zip(names, positions):
                a = self._by_joint.get(n)
                if a is not None:
                    self._cmd[a] = float(v)

    def merge(self, measured: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            return {**measured, **{a: v for a, v in self._cmd.items() if a in measured}}


class GraspAssistHook:
    """post_step_hook around GraspAssist: held bags follow their gripper EVERY tick (smooth in the render);
    grasp/release detection runs every `div` ticks. io: link_world(arm), body_world(path),
    set_body_world(path, T), set_held(path, bool) (kinematic + no collisions while held). read_grip() ->
    {arm: leader joint}. Never raises (a failure is counted and logged once)."""

    def __init__(self, assist: GraspAssist, io, read_grip, div: int = 4, log=None):
        self.assist, self.io, self.read_grip, self.div, self.log = assist, io, read_grip, max(1, div), log
        self._tick = 0
        self._s = {"attach": 0, "detach": 0, "errors": 0, "follow_ticks": 0}
        self._warned = False

    def __call__(self, sim_time: float) -> None:
        tick = self._tick
        self._tick += 1
        try:
            if self.assist.held:
                for b, T in self.assist.targets(self.io.link_world).items():
                    self.io.set_body_world(b, T)
                self._s["follow_ticks"] += 1
            if tick % self.div:
                return
            for kind, arm, bag in self.assist.step(self.read_grip(), self.io.link_world, self.io.body_world):
                if kind == "miss":
                    self._s["miss"] = self._s.get("miss", 0) + 1
                    if self.log is not None:
                        a, tip, near, loc = self.assist.misses[-1]
                        self.log.info(f"[grasp_assist] {arm} gripper closed with no bag in reach: fingertip {tip}, "
                                      f"nearest {near} (fingertip in its frame {loc})")
                    continue
                self.io.set_held(bag, kind == "attach")
                self._s[kind] += 1
                if self.log is not None:
                    self.log.info(f"[grasp_assist] {kind} {bag} ({arm} gripper) at t={sim_time:.2f}")
        except Exception as exc:  # noqa: BLE001 - never take the physics loop down
            self._s["errors"] += 1
            if self.log is not None and not self._warned:
                self.log.warn(f"[grasp_assist] error (further errors only counted): {exc}")
                self._warned = True

    def stats(self) -> dict:
        return dict(self._s)


class PokeQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._items: List[Poke] = []

    def put(self, req: Poke) -> None:
        with self._lock:
            self._items.append(req)

    def drain(self) -> List[Poke]:
        with self._lock:
            out, self._items = self._items, []
        return out


class ObjectPoseRelay:
    """post_step_hook: sample every N ticks, filter, hand the movers to the publisher."""

    def __init__(self, source, publisher, physics_hz: float, rate_hz: float = 30.0, keyframe_s: float = 2.0,
                 clock: Callable[[], float] = time.time, pokes: Optional[PokeQueue] = None):
        self.source = source
        self.pokes = pokes
        self.publisher = publisher
        self.div = tick_divider(physics_hz, rate_hz)
        self.filter = ChangeFilter(keyframe_s=keyframe_s)
        self.clock = clock
        self._tick = 0
        self._s = {"samples": 0, "published_msgs": 0, "published_tfs": 0, "errors": 0,
                   "cost_ms_sum": 0.0, "cost_ms_max": 0.0, "pokes": 0}

    def __call__(self, sim_time: float) -> None:
        tick = self._tick
        self._tick += 1
        if self.pokes is not None:
            for req in self.pokes.drain():   # on the physics thread, between steps
                try:
                    self.source.apply(req)
                    self._s["pokes"] += 1
                except Exception:
                    self._s["errors"] += 1
        if tick % self.div:
            return
        t0 = time.perf_counter()
        try:
            poses = self.source.read()
            out = self.filter.update(sim_time, poses)
            if out:
                self.publisher.send(self.clock(), out)
                self._s["published_msgs"] += 1
                self._s["published_tfs"] += len(out)
            self._s["samples"] += 1
        except Exception:
            self._s["errors"] += 1
        ms = (time.perf_counter() - t0) * 1000.0
        self._s["cost_ms_sum"] += ms
        self._s["cost_ms_max"] = max(self._s["cost_ms_max"], ms)

    def stats(self) -> dict:
        return dict(self._s)


# ------------------------------------------------------------------------------- engine-side adapters

class UsdPoseSource:
    """Local (parent-relative) poses of the selected bodies from the physics USD stage."""

    def __init__(self, stage, paths: Sequence[str]):
        from pxr import UsdGeom  # noqa: PLC0415  (only inside the engine process)
        self.paths = list(paths)
        self._xf = [UsdGeom.Xformable(stage.GetPrimAtPath(p)) for p in self.paths]

    def read(self) -> Dict[str, Pose]:
        out: Dict[str, Pose] = {}
        for path, xf in zip(self.paths, self._xf):
            m = xf.GetLocalTransformation()
            t = m.ExtractTranslation()
            q = m.GetOrthonormalized().ExtractRotationQuat()
            im = q.GetImaginary()
            out[path] = ((t[0], t[1], t[2]), (q.GetReal(), im[0], im[1], im[2]))
        return out

    def apply(self, req) -> None:
        """Teleport (xyz / dxyz, parent frame; yaw_deg), set the linear velocity, and/or stop (zero linear and
        angular velocity) one body. PhysX picks up USD edits of a rigid body's xformOp / physics:velocity
        between steps (omni.physx USD change listener)."""
        from pxr import Gf, UsdGeom  # noqa: PLC0415
        path, xyz, dxyz, vel = req
        xf = self._xf[self.paths.index(path)]
        prim = xf.GetPrim()
        if xyz is not None or dxyz is not None:
            op = next(o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            cur = op.Get()
            new = xyz if xyz is not None else (cur[0] + dxyz[0], cur[1] + dxyz[1], cur[2] + dxyz[2])
            op.Set(type(cur)(*new))
        yaw = getattr(req, "yaw_deg", None)
        if yaw is not None:
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            set_xform_ops(xf, None, ((c, -s_, 0.0), (s_, c, 0.0), (0.0, 0.0, 1.0)))
        if vel is not None:
            prim.GetAttribute("physics:velocity").Set(Gf.Vec3f(*vel))
        if getattr(req, "stop", False):
            for a in ("physics:velocity", "physics:angularVelocity"):
                attr = prim.GetAttribute(a)
                if attr:
                    attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _quat_from_R(R):
    """(w, x, y, z) from a 3x3 rotation (nested sequences)."""
    m = [[float(R[i][j]) for j in range(3)] for i in range(3)]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j][i] + m[i][j]) / s
    q[k] = (m[k][i] + m[i][k]) / s
    return ((m[k][j] - m[j][k]) / s, q[0], q[1], q[2])


def set_xform_ops(xf, pos, R) -> None:
    """Write a parent-frame position and/or 3x3 rotation into the prim's existing xform ops: translate, and
    orient (full rotation) | rotateZ (yaw only) | rotateXYZ. Scale ops are left alone."""
    from pxr import Gf, UsdGeom  # noqa: PLC0415
    T = UsdGeom.XformOp
    for op in xf.GetOrderedXformOps():
        t = op.GetOpType()
        if t == T.TypeTranslate and pos is not None:
            op.Set(type(op.Get())(*[float(c) for c in pos]))
        elif R is not None and t == T.TypeOrient:
            w, x, y, z = _quat_from_R(R)
            qt = Gf.Quatf if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
            op.Set(qt(w, x, y, z))
        elif R is not None and t == T.TypeRotateZ:
            op.Set(math.degrees(math.atan2(float(R[1][0]), float(R[0][0]))))
        elif R is not None and t == T.TypeRotateXYZ:
            r = [[float(R[i][j]) for j in range(3)] for i in range(3)]
            # R = Rz(c) Ry(b) Rx(a) for USD rotateXYZ (x applied first)
            b = math.asin(max(-1.0, min(1.0, -r[2][0])))
            a = math.atan2(r[2][1], r[2][2])
            c = math.atan2(r[1][0], r[0][0])
            op.Set(Gf.Vec3f(math.degrees(a), math.degrees(b), math.degrees(c)))


class UsdGraspIO:
    """GraspAssistHook IO on the physics USD stage: gripper link world poses (PhysX writes the robot links'
    xforms back every tick), bag world poses, and the held state (kinematic + colliders off while carried, so
    the carried box neither fights the gripper nor shoves the carton; dynamic again with zero velocity on
    release)."""

    def __init__(self, stage, link_paths: Dict[str, str]):
        from pxr import UsdGeom  # noqa: PLC0415
        self.stage = stage
        self._links = {a: UsdGeom.Xformable(stage.GetPrimAtPath(p)) for a, p in link_paths.items()}
        self._bodies: Dict[str, object] = {}

    @staticmethod
    def _np(m):
        import numpy as np  # noqa: PLC0415
        return np.array([[m[r][c] for c in range(4)] for r in range(4)]).T   # Gf row-vector -> column 4x4

    def _body(self, path):
        from pxr import UsdGeom  # noqa: PLC0415
        if path not in self._bodies:
            self._bodies[path] = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        return self._bodies[path]

    def link_world(self, arm):
        from pxr import Usd  # noqa: PLC0415
        return self._np(self._links[arm].ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    def body_world(self, path):
        from pxr import Usd  # noqa: PLC0415
        return self._np(self._body(path).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    def set_body_world(self, path, T):
        import numpy as np  # noqa: PLC0415
        from pxr import Usd  # noqa: PLC0415
        xf = self._body(path)
        P = self._np(xf.ComputeParentToWorldTransform(Usd.TimeCode.Default()))
        L = np.linalg.inv(P) @ T
        set_xform_ops(xf, L[:3, 3], L[:3, :3])

    def set_held(self, path, held: bool):
        from pxr import Gf, Usd, UsdPhysics  # noqa: PLC0415
        prim = self._body(path).GetPrim()
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(bool(held))
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(not held)
        if not held:
            for a in ("physics:velocity", "physics:angularVelocity"):
                attr = prim.GetAttribute(a)
                if attr:
                    attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def find_gripper_links(stage, robot_prefix: str, names=(("left", "gripper_l_base_link"),
                                                        ("right", "gripper_r_base_link"))) -> Dict[str, str]:
    root = stage.GetPrimAtPath("/" + robot_prefix.strip("/"))
    from pxr import Usd  # noqa: PLC0415
    want = dict((n, a) for a, n in names)
    out: Dict[str, str] = {}
    for p in Usd.PrimRange(root):
        a = want.get(p.GetName())
        if a and a not in out:
            out[a] = p.GetPath().pathString
    return out


def discover_bodies(stage, robot_prefix: str, include_kinematic: bool = False) -> List[str]:
    from pxr import UsdPhysics  # noqa: PLC0415
    entries = []
    for prim in stage.Traverse():
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if not rigid:
            continue
        kin = prim.GetAttribute("physics:kinematicEnabled")
        en = prim.GetAttribute("physics:rigidBodyEnabled")
        entries.append((prim.GetPath().pathString, True, bool(kin.Get()) if kin else False,
                        bool(en.Get()) if en and en.Get() is not None else True))
    return select_object_bodies(entries, robot_prefix, include_kinematic)


class RclpyTfPublisher:
    """/tf_render publisher on its own rclpy context + thread; send() never blocks the physics tick."""

    def __init__(self, topic: str = "/tf_render", frame_id: str = "map", node_name: str = "g2cell_object_pose_relay",
                 pokes: Optional[PokeQueue] = None, allowed=(), poke_topic: str = "/g2cell/object_poke", log=None):
        import rclpy  # noqa: PLC0415
        from rclpy.context import Context  # noqa: PLC0415
        from rclpy.qos import qos_profile_sensor_data  # noqa: PLC0415
        from tf2_msgs.msg import TFMessage  # noqa: PLC0415
        from geometry_msgs.msg import TransformStamped  # noqa: PLC0415
        self._TF, self._TS = TFMessage, TransformStamped
        self._ctx = Context()
        rclpy.init(args=[], context=self._ctx)   # args=[]: the engine's --ros-args (node name remap) stay its own
        self._node = rclpy.create_node(node_name, context=self._ctx)
        self._pub = self._node.create_publisher(TFMessage, topic, qos_profile_sensor_data)
        self.frame_id = frame_id
        self._lock = threading.Lock()
        self._pending = None
        self._ev = threading.Event()
        self._stop = False
        self._th = threading.Thread(target=self._run, name="obj_relay_pub", daemon=True)
        self._th.start()
        if pokes is not None:
            from std_msgs.msg import String  # noqa: PLC0415
            from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
            allowed = set(allowed)

            def on_poke(msg):
                try:
                    pokes.put(parse_poke(msg.data, allowed))
                except ValueError as exc:
                    if log is not None:
                        log.warn(f"[obj_relay] poke rejected: {exc}")
            self._node.create_subscription(String, poke_topic, on_poke, 10)
            self._exe = SingleThreadedExecutor(context=self._ctx)
            self._exe.add_node(self._node)
            threading.Thread(target=self._spin, name="obj_relay_spin", daemon=True).start()

    def _spin(self) -> None:
        while not self._stop:
            try:
                self._exe.spin_once(timeout_sec=0.2)
            except Exception:
                time.sleep(0.2)

    def send(self, stamp: float, items: Dict[str, Pose]) -> None:
        with self._lock:
            if self._pending is not None:      # unsent older sample: merge, newer wins
                items = {**self._pending[1], **items}
            self._pending = (stamp, items)
        self._ev.set()

    def _run(self) -> None:
        while not self._stop:
            self._ev.wait(0.5)
            self._ev.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            stamp, items = job
            sec = int(stamp)
            msg = self._TF()
            for path, ((x, y, z), (w, qx, qy, qz)) in items.items():
                t = self._TS()
                t.header.stamp.sec, t.header.stamp.nanosec = sec, int((stamp - sec) * 1e9)
                t.header.frame_id = self.frame_id
                t.child_frame_id = path
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
                t.transform.rotation.w, t.transform.rotation.x = w, qx
                t.transform.rotation.y, t.transform.rotation.z = qy, qz
                msg.transforms.append(t)
            try:
                self._pub.publish(msg)
            except Exception:
                pass

    def close(self) -> None:
        self._stop = True
        self._ev.set()
        try:
            self._node.destroy_node()
            self._ctx.try_shutdown()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: Optional[List[str]] = None) -> int:
    import argparse  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    ap = argparse.ArgumentParser(prog="object_relay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ml = sub.add_parser("make-launcher", help="copy a launcher YAML with the engine swapped for g2cell_engine_isaacsim.py")
    ml.add_argument("src")
    ml.add_argument("dst")
    a = ap.parse_args(argv)
    with open(a.src) as f:
        cfg = yaml.safe_load(f)
    with open(a.dst, "w") as f:
        f.write(f"# generated by g2cell_bringup.object_relay from {a.src}: engine -> "
                f"{ENGINE_PACKAGE}/{ENGINE_EXECUTABLE} (object pose relay)\n")
        yaml.safe_dump(patch_launcher(cfg), f, sort_keys=False)
    print(a.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
