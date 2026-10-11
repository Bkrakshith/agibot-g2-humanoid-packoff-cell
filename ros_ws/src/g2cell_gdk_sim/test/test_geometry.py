"""Pose helpers: x, y, z, w quaternions and 4x4 matrices, checked against gdk_training.geometry."""
import importlib.util
import math

import numpy as np
import pytest

import agibot_gdk
from agibot_gdk import _geometry as g

HALF_TURNS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.7071068, -0.7071068, 0, 0), (0.6, 0, -0.8, 0)]


def _training_geometry(training_dir):
    spec = importlib.util.spec_from_file_location("training_geometry", training_dir / "gdk_training" / "geometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("q", HALF_TURNS)
def test_quaternion_round_trip_near_half_turns(q):
    m = g.matrix((0.1, 0.2, 0.3), q)
    assert np.allclose(g.matrix((0.1, 0.2, 0.3), g.quaternion(m)), m, atol=1e-9)


def test_random_round_trip_and_inverse():
    for q in np.random.default_rng(1).normal(size=(500, 4)):
        m = g.matrix((0.3, -0.1, 1.0), tuple(q))
        assert np.allclose(g.matrix((0.3, -0.1, 1.0), g.quaternion(m)), m, atol=1e-9)
        assert np.allclose(g.inverse(m) @ m, np.eye(4), atol=1e-12)


def test_matches_gdk_training_geometry(training_dir):
    tg = _training_geometry(training_dir)
    for q in np.random.default_rng(2).normal(size=(200, 4)):
        p = tuple(np.random.default_rng(3).normal(size=3))
        assert np.allclose(g.matrix(p, tuple(q)), tg.matrix(p, tuple(q)))
        m = tg.matrix(p, tuple(q))
        assert np.allclose(tg.matrix(p, g.quaternion(m)), m, atol=1e-9)


def test_pose_and_transform_round_trip_through_gdk_structs(training_dir):
    tg = _training_geometry(training_dir)
    m = g.matrix((0.5, -0.2, 1.1), (0.1, 0.2, 0.3, 0.9))
    pose, transform = g.to_pose(m, agibot_gdk), g.to_transform(m, agibot_gdk)
    assert isinstance(pose, agibot_gdk.Pose) and isinstance(transform, agibot_gdk.Transform)
    assert np.allclose(tg.from_pose(pose), m) and np.allclose(tg.from_transform(transform), m)
    assert np.allclose(g.from_pose(pose), m) and np.allclose(g.from_transform(transform), m)


def test_wxyz_conversion():
    assert g.wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)
    assert g.wxyz_to_xyzw((0.5, 0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3, 0.5)


def test_usd_camera_to_optical_flips_y_and_z():
    usd = g.matrix((0.1, 0.0, 0.2), (0, 0, 0, 1))
    optical = g.usd_camera_to_optical(usd)
    assert np.allclose(optical[:3, :3], np.diag([1, -1, -1]))
    assert np.allclose(optical[:3, 3], (0.1, 0.0, 0.2))


def test_rotation_about_z():
    m = g.matrix((0, 0, 0), (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    assert np.allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0])
