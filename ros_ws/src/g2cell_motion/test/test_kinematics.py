import math
import pathlib

import numpy as np
import pytest

from g2cell_motion.kinematics import ArmIk, UrdfKinematics, quat_to_mat
from g2cell_motion.robot import HOME_DEG

_here = pathlib.Path(__file__).resolve()
URDF = next((p for p in [pathlib.Path("/geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf")] +
             [q / "geniesim_assets/scenes/scene_g2cell_mobile/robot.urdf" for q in _here.parents] if p.is_file()),
            None)
needs = pytest.mark.skipif(URDF is None, reason="staged robot.urdf missing")

# 2-link planar toy (z axes) + fixed tool: enough to test the parser/FK without the robot
TOY = """<robot name="t">
<link name="base_link"/><link name="a"/><link name="b"/><link name="tip"/>
<joint name="j1" type="revolute"><parent link="base_link"/><child link="a"/>
  <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
<joint name="j2" type="revolute"><parent link="a"/><child link="b"/>
  <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
<joint name="t" type="fixed"><parent link="b"/><child link="tip"/><origin xyz="0.5 0 0"/></joint>
</robot>"""

LEFT = [f"idx2{k}_arm_l_joint{k}" for k in range(1, 8)]
STATE = {"idx03_body_joint3": math.radians(20.0), "idx05_body_joint5": math.radians(40.0)}


def test_toy_fk():
    k = UrdfKinematics(TOY)
    T = k.link_poses({"j1": math.pi / 2, "j2": 0.0}, ["tip"])["tip"]
    assert np.allclose(T[:3, 3], [0.0, 1.5, 1.0])


@needs
def test_arm_fk_matches_full_tree_fk():
    kin = UrdfKinematics(str(URDF))
    ik = ArmIk(kin, LEFT, "arm_l_end_link")
    rng = np.random.default_rng(1)
    Q = rng.uniform(-1, 1, size=(5, 7))
    p, R, _, _ = ik.fk(Q, ik._prefix(STATE))
    for i in range(5):
        T = kin.link_poses({**STATE, **dict(zip(LEFT, Q[i]))}, ["arm_l_end_link"])["arm_l_end_link"]
        assert np.allclose(T[:3, 3], p[i]) and np.allclose(T[:3, :3], R[i])


def _mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (x, y, z, w)


@needs
def test_ik_round_trip_and_determinism():
    kin = UrdfKinematics(str(URDF))
    ik = ArmIk(kin, LEFT, "arm_l_end_link")
    home = dict(zip(LEFT, (math.radians(d) for d in HOME_DEG)))
    goal = {n: home[n] + d for n, d in zip(LEFT, (0.3, 0.2, -0.2, 0.3, 0.1, 0.25, 0.2))}
    T = kin.link_poses({**STATE, **goal}, ["arm_l_end_link"])["arm_l_end_link"]
    q = _mat_to_quat(T[:3, :3])
    assert np.allclose(quat_to_mat(q), T[:3, :3], atol=1e-9)
    a = ik.solve(T[:3, 3], q, STATE, [home])
    b = ik.solve(T[:3, 3], q, STATE, [home])
    assert a is not None and a == b                       # bit-identical
    T2 = kin.link_poses({**STATE, **a}, ["arm_l_end_link"])["arm_l_end_link"]
    assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 1e-5
    assert np.allclose(T2[:3, :3], T[:3, :3], atol=1e-4)


@needs
def test_ik_unreachable_returns_none_deterministically():
    kin = UrdfKinematics(str(URDF))
    ik = ArmIk(kin, LEFT, "arm_l_end_link", max_iter=60)
    home = dict(zip(LEFT, (math.radians(d) for d in HOME_DEG)))
    assert ik.solve((3.0, 3.0, 1.0), (1.0, 0.0, 0.0, 0.0), STATE, [home]) is None
    assert ik.stats["stage2"] == 1 and ik.stats["fail"] == 1


@needs
def test_ik_respects_moveit_coupled_constraints():
    """genie_sim_moveit_plugins' KDL rejects solutions with A.q > b - 0.1 on arm joints 6/7; the local IK
    must not return such solutions (MoveIt would re-randomise to another branch)."""
    from g2cell_motion.robot import ARM_COUPLED
    A, b = (np.array(ARM_COUPLED["A"]), np.array(ARM_COUPLED["b"]))
    kin = UrdfKinematics(str(URDF))
    ik = ArmIk(kin, LEFT, "arm_l_end_link", coupled=[([LEFT[5], LEFT[6]], A, b)], coupled_margin=0.1)
    home = dict(zip(LEFT, (math.radians(d) for d in HOME_DEG)))
    rng = np.random.default_rng(7)
    n_ok = 0
    for _ in range(15):
        goal = {n: home[n] + d for n, d in zip(LEFT, rng.uniform(-0.6, 0.6, 7))}
        goal[LEFT[5]], goal[LEFT[6]] = 0.9, -1.4          # violates the constraint (row 4)
        T = kin.link_poses({**STATE, **goal}, ["arm_l_end_link"])["arm_l_end_link"]
        sol = ik.solve(T[:3, 3], _mat_to_quat(T[:3, :3]), STATE, [home])
        if sol is not None:
            n_ok += 1
            q = np.array([sol[LEFT[5]], sol[LEFT[6]]])
            assert (A @ q <= b - 0.1 + 1e-9).all()
    assert ik.feasible(dict(zip(LEFT, [0, 0, 0, 0, 0, 0.0, 0.0])))
    assert not ik.feasible({**dict(zip(LEFT, [0.0] * 7)), LEFT[5]: 0.9, LEFT[6]: -1.4})
