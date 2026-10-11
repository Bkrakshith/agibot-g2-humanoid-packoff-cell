"""Named arm poses from YAML (no ROS imports).

    named_poses:
      home:        {joints_deg: [0, -45, 0, -75, 0, -55, 0]}           # both arms
      above_crate: {tool: {xyz: [0.60, 0.0, 1.194], yaw_deg: 0, tilt_deg: 0}}
      above_table: {left: {tool: {xyz: [0.0, 0.60, 1.10]}}}            # per arm
      home:        {joints_deg: [...], waist_deg: 0}                      # also turns the waist back

`waist_deg` (optional, joints poses only) moves idx05_body_joint5 with the arm unless the goal's
waist_mode is `fixed`.

`tool.xyz` is the fingertip point in the cell frame; without yaw_deg/tilt_deg the planner searches
gripper-down orientations. A per-arm entry (`left:` / `right:`) overrides the shared one.
"""
from __future__ import annotations

from dataclasses import dataclass

from g2cell_motion.robot import ARMS


@dataclass(frozen=True)
class NamedTarget:
    kind: str                                   # "joints" | "tool"
    joints_deg: tuple | None = None             # 7 values, joints 1..7
    xyz: tuple | None = None                    # fingertip, cell frame
    yaw_deg: float | None = None
    tilt_deg: float | None = None
    waist_deg: float | None = None              # joints poses: waist yaw to move to (None = keep)


def _parse_one(name: str, d) -> NamedTarget:
    if not isinstance(d, dict):
        raise ValueError(f"named pose {name!r}: expected a mapping")
    has_j, has_t = "joints_deg" in d, "tool" in d
    if has_j == has_t:
        raise ValueError(f"named pose {name!r}: give exactly one of joints_deg / tool")
    if has_j:
        j = d["joints_deg"]
        if not isinstance(j, (list, tuple)) or len(j) != 7:
            raise ValueError(f"named pose {name!r}: joints_deg needs 7 values")
        w = d.get("waist_deg")
        return NamedTarget("joints", joints_deg=tuple(float(v) for v in j),
                           waist_deg=None if w is None else float(w))
    t = d["tool"]
    xyz = t.get("xyz") if isinstance(t, dict) else None
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(f"named pose {name!r}: tool.xyz needs 3 values")
    yaw, tilt = t.get("yaw_deg"), t.get("tilt_deg")
    if yaw is None and tilt is not None:
        raise ValueError(f"named pose {name!r}: tilt_deg without yaw_deg")
    return NamedTarget("tool", xyz=tuple(float(v) for v in xyz),
                       yaw_deg=None if yaw is None else float(yaw),
                       tilt_deg=None if yaw is None else float(tilt or 0.0))


def load_named_poses(doc) -> dict[str, dict[str, NamedTarget]]:
    """{name: {arm: NamedTarget}} from a parsed YAML document. Raises ValueError."""
    if not isinstance(doc, dict) or not isinstance(doc.get("named_poses"), dict):
        raise ValueError("expected a top-level 'named_poses' mapping")
    out: dict[str, dict[str, NamedTarget]] = {}
    for name, d in doc["named_poses"].items():
        if not isinstance(d, dict) or not d:
            raise ValueError(f"named pose {name!r}: empty")
        per_arm = {a: d[a] for a in ARMS if a in d}
        shared = {k: v for k, v in d.items() if k not in ARMS}
        entry: dict[str, NamedTarget] = {}
        if shared:
            t = _parse_one(name, shared)
            entry = {a: t for a in ARMS}
        for a, v in per_arm.items():
            entry[a] = _parse_one(f"{name}.{a}", v)
        out[name] = entry
    return out


def resolve(poses, name: str, arm: str) -> NamedTarget:
    if name not in poses:
        raise KeyError(f"unknown named pose {name!r} (known: {sorted(poses)})")
    if arm not in poses[name]:
        raise KeyError(f"named pose {name!r} is not defined for the {arm} arm")
    return poses[name][arm]
