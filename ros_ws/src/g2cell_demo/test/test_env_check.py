import json
import math
import pathlib

import numpy as np
import pytest

from g2cell_demo.env_check import EnvChecker, head_clearance

_here = pathlib.Path(__file__).resolve()
ASSETS = next((p for p in [pathlib.Path("/geniesim_assets/scenes")] +
               [q / "geniesim_assets" / "scenes" for q in _here.parents] if p.is_dir()),
              pathlib.Path("/nonexistent"))
URDF = ASSETS / "scene_g2cell_mobile" / "robot.urdf"
COLL = ASSETS / "g2cell" / "cell_colliders.json"
needs = pytest.mark.skipif(not (URDF.is_file() and COLL.is_file()), reason="staged robot / cell colliders missing")


def _box_checker(tmp_path, boxes):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(boxes))
    chk = EnvChecker.__new__(EnvChecker)
    chk.box_names = [b["name"] for b in boxes]
    chk.c = np.array([b["c"] for b in boxes])
    chk.ax = np.array([b["axes"] for b in boxes])
    chk.h = np.array([b["h"] for b in boxes])
    return chk


def test_signed_distance_to_oriented_box(tmp_path):
    s = math.sqrt(0.5)
    chk = _box_checker(tmp_path, [{"name": "b", "c": [0, 0, 0], "axes": [[s, s, 0], [-s, s, 0], [0, 0, 1]],
                                   "h": [0.1, 0.1, 0.1]}])
    d = chk.distances(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [s * 0.2, s * 0.2, 0.0]]))
    assert d[0] == pytest.approx(-0.1)          # min over points: the centre is 0.1 inside


@needs
def test_tuck_pose_clear_of_cell_and_head():
    chk = EnvChecker(str(URDF), str(COLL))
    if not any(n.startswith("arm_") for n in chk.points):
        pytest.skip("robot meshes not resolvable here (pkg/ links point into the container); run in the container")
    j = {"idx03_body_joint3": math.radians(20)}
    for side, base in (("l", 20), ("r", 60)):
        for k, deg in ((2, -90), (4, -110), (6, -30)):
            j[f"idx{base + k}_arm_{side}_joint{k}"] = math.radians(deg)
    for arm in ("left", "right"):
        d, _ = chk.clearance(j, arm, base=(0.0, 0.0, 0.0))
        assert d > 0.05
        assert head_clearance(chk, j, arm) > 0.2
