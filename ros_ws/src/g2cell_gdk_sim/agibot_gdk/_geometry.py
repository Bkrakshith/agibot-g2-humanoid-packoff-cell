"""Poses as 4x4 numpy matrices; quaternions in the GDK order x, y, z, w. T_a_b maps b points into a."""
from __future__ import annotations

import math

import numpy as np


def matrix(position, quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = position
    return m


def quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation of `m` as x, y, z, w (Shepperd: largest component first, stable near half turns)."""
    r = m[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    candidates = [trace, r[0, 0], r[1, 1], r[2, 2]]
    k = int(np.argmax(candidates))
    if k == 0:
        s = 2 * math.sqrt(1 + trace)
        q = ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4)
    elif k == 1:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = (s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s)
    elif k == 2:
        s = 2 * math.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2])
        q = ((r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s)
    else:
        s = 2 * math.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2])
        q = ((r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s)
    return tuple(float(v) for v in q)


def inverse(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (float(x), float(y), float(z), float(w))


def usd_camera_to_optical(m: np.ndarray) -> np.ndarray:
    """USD cameras look along -Z with +Y up; the optical frame looks along +Z with +Y down."""
    return m @ np.diag([1.0, -1.0, -1.0, 1.0])


def from_pose(pose) -> np.ndarray:
    p, q = pose.position, pose.orientation
    return matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def from_transform(transform) -> np.ndarray:
    t, q = transform.translation, transform.rotation
    return matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def fill_pose(pose, m: np.ndarray) -> None:
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in m[:3, 3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion(m)


def to_pose(m: np.ndarray, gdk):
    pose = gdk.Pose()
    fill_pose(pose, m)
    return pose


def to_transform(m: np.ndarray, gdk):
    t = gdk.Transform()
    t.translation.x, t.translation.y, t.translation.z = (float(v) for v in m[:3, 3])
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = quaternion(m)
    return t


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m
