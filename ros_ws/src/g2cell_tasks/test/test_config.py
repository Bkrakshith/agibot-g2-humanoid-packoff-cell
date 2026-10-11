import pathlib

import yaml

from g2cell_tasks.config import load_task_config

CFG = pathlib.Path(__file__).parents[1] / "config" / "tasks.yaml"


def test_shipped_config_loads_and_is_consistent():
    c = load_task_config(yaml.safe_load(CFG.read_text()))
    assert [b.id for b in c.bags.on_table()] == ["Bag_01", "Bag_02", "Bag_03"]
    assert c.sort.target_for("good") == ("crate", "")
    assert set(c.targets.lanes) == {"L1_1", "L2_1"}  # Layout v2: 1 column x 2 levels
    assert c.reach.left_table_max_y == 0.85
    # every lane the sort rules can produce has a drop point
    for label in ("reject", "sample", "SKU_X"):
        _, lane = c.sort.target_for(label)
        assert lane in c.targets.lanes
