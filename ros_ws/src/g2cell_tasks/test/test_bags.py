import pytest

from g2cell_tasks.bags import Bag, BagCatalog

BAGS = [
    {"id": "Bag_01", "variant": "standard", "label": "good", "xy": [0.10, 0.56], "top_z": 0.94},
    {"id": "Bag_02", "variant": "small", "label": "reject", "xy": [-0.16, 0.60], "top_z": 0.94},
    {"id": "Bag_03", "variant": "large", "label": "good", "xy": [0.02, 0.98], "top_z": 0.94},
]


def test_parse_and_get():
    c = BagCatalog.from_list(BAGS)
    b = c.get("Bag_02")
    assert b == Bag("Bag_02", "small", "reject", (-0.16, 0.60), 0.94)


def test_choose_any_skips_unpickable_and_removed():
    c = BagCatalog.from_list(BAGS)
    near = lambda b: b.xy[1] <= 0.85
    assert c.choose("any", near).id == "Bag_01"
    c.mark_removed("Bag_01")
    assert c.choose("any", near).id == "Bag_02"
    c.mark_removed("Bag_02")
    assert c.choose("any", near) is None
    assert [b.id for b in c.on_table()] == ["Bag_03"]


def test_choose_by_id():
    c = BagCatalog.from_list(BAGS)
    assert c.choose("Bag_03").id == "Bag_03"
    with pytest.raises(KeyError):
        c.choose("Bag_99")
    c.mark_removed("Bag_03")
    assert c.choose("Bag_03") is None


@pytest.mark.parametrize("bad", [
    [{"id": "a", "variant": "huge", "label": "good", "xy": [0, 0], "top_z": 1}],
    [{"id": "a", "variant": "small", "label": "", "xy": [0, 0], "top_z": 1}],
    [{"id": "a", "variant": "small", "label": "good", "xy": [0], "top_z": 1}],
    [{"id": "a", "variant": "small", "label": "good", "xy": [0, 0], "top_z": 1}] * 2,
])
def test_invalid(bad):
    with pytest.raises(ValueError):
        BagCatalog.from_list(bad)
