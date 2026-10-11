from pathlib import Path

import pytest

from g2cell_datagen import scene_variant as V

BASE_YAML = """# comment
robot:
  robot_name: G2
lidars: []
scene: scenes/g2cell/cell.usda
"""


def test_override_usda_layers_cell_and_overrides_bag_xforms():
    txt = V.override_usda("../../cell.usda", {"Bag_02": (-0.1, 0.6, 30.0, 0.87), "Bag_01": (0.0, 1.4, 0.0, 0.0)},
                          comment="seed 7")
    assert txt.startswith("#usda 1.0")
    assert "subLayers = [\n        @../../cell.usda@\n    ]" in txt
    assert 'defaultPrim = "World"' in txt
    assert 'over "World"' in txt and 'over "Bags"' in txt
    i1, i2 = txt.index('over "Bag_01"'), txt.index('over "Bag_02"')
    assert i1 < i2                                       # stable order
    assert "double3 xformOp:translate = (-0.1, 0.6, 0.87)" in txt
    assert "float xformOp:rotateZ = 30" in txt
    assert "double3 xformOp:translate = (0, 1.4, 0)" in txt
    assert "seed 7" in txt


def test_override_usda_parses_with_usd_if_available(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd
    cell = tmp_path / "cell.usda"
    cell.write_text('#usda 1.0\n(\n    defaultPrim = "World"\n)\n\ndef Xform "World"\n{\n    def Xform "Bags"\n    {\n'
                    '        def Xform "Bag_01"\n        {\n            float xformOp:rotateZ = 0\n'
                    '            double3 xformOp:translate = (0.05, 0.62, 0.87)\n'
                    '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]\n'
                    '        }\n    }\n}\n')
    sub = tmp_path / "datagen" / "b1"
    sub.mkdir(parents=True)
    ov = sub / "ep.usda"
    ov.write_text(V.override_usda("../../cell.usda", {"Bag_01": (-0.2, 0.7, 45.0, 0.87)}))
    stage = Usd.Stage.Open(str(ov))
    prim = stage.GetPrimAtPath("/World/Bags/Bag_01")
    assert tuple(prim.GetAttribute("xformOp:translate").Get()) == pytest.approx((-0.2, 0.7, 0.87))
    assert prim.GetAttribute("xformOp:rotateZ").Get() == pytest.approx(45.0)


def test_scene_yaml_replaces_scene_line_only():
    out = V.scene_yaml(BASE_YAML, "/geniesim_assets/scenes/g2cell/datagen/b1/ep.usda", comment="seed 3")
    assert "scene: /geniesim_assets/scenes/g2cell/datagen/b1/ep.usda" in out
    assert "scene: scenes/g2cell/cell.usda" not in out
    assert "robot_name: G2" in out and "seed 3" in out.splitlines()[0]


def test_scene_yaml_without_scene_key_raises():
    with pytest.raises(ValueError):
        V.scene_yaml("robot: {}\n", "x.usda")


def test_write_variant(tmp_path):
    base = tmp_path / "scene_g2cell.yaml"
    base.write_text(BASE_YAML)
    cell = tmp_path / "assets" / "scenes" / "g2cell" / "cell.usda"
    cell.parent.mkdir(parents=True)
    cell.write_text("#usda 1.0\n")
    out_dir = tmp_path / "assets" / "scenes" / "g2cell" / "datagen" / "batchA"
    usda, yml = V.write_variant(base, cell, out_dir, "ep_s00003", {"Bag_01": (0.0, 0.6, 0.0, 0.87)})
    assert usda == out_dir / "ep_s00003.usda" and yml == out_dir / "ep_s00003.yaml"
    assert "@../../cell.usda@" in usda.read_text()
    assert f"scene: {usda}" in yml.read_text()
    assert Path(base).read_text() == BASE_YAML


REAL_CELL = Path(__file__).resolve().parents[5] / "geniesim_assets" / "scenes" / "g2cell" / "cell.usda"


def test_override_on_the_real_cell_usda(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd
    if not REAL_CELL.is_file():
        pytest.skip(f"no {REAL_CELL}")
    usda, _ = V.write_variant(_base(tmp_path), REAL_CELL, tmp_path / "variants", "ep",
                              {"Bag_01": (-0.2, 0.7, 45.0, 0.87), "Bag_02": (-0.4, 1.4, 0.0, 0.0)})
    stage = Usd.Stage.Open(str(usda))
    b1 = stage.GetPrimAtPath("/World/Bags/Bag_01")
    assert b1.IsValid() and len(b1.GetChildren()) > 0               # the bag asset reference still resolves
    assert tuple(b1.GetAttribute("xformOp:translate").Get()) == pytest.approx((-0.2, 0.7, 0.87))
    assert tuple(stage.GetPrimAtPath("/World/Bags/Bag_02").GetAttribute("xformOp:translate").Get()) == \
        pytest.approx((-0.4, 1.4, 0.0))
    assert tuple(stage.GetPrimAtPath("/World/Bags/Bag_03").GetAttribute("xformOp:translate").Get()) == \
        pytest.approx((-0.1, 0.9, 0.87))                               # untouched bag keeps its pose
    assert stage.GetPrimAtPath("/PhysicsScene").IsValid()


def _base(tmp_path):
    p = tmp_path / "scene.yaml"
    p.write_text(BASE_YAML)
    return p
