from g2cell_tasks.reachability import ReachTable, select_arm


def test_table_pick_uses_left_arm_inside_reach():
    c = select_arm(ReachTable(), "table", (0.10, 0.56), "crate")
    assert c.arm == "left" and c.verified


def test_table_pick_beyond_left_reach_is_unreachable():
    c = select_arm(ReachTable(), "table", (0.02, 0.98), "crate")
    assert c.arm is None and "0.85" in c.reason


def test_right_arm_never_picks_from_table():
    c = select_arm(ReachTable(), "table", (0.10, 0.56), "crate", requested="right")
    assert c.arm is None and "right" in c.reason


def test_crate_pick_prefers_arm_on_that_side():
    assert select_arm(ReachTable(), "crate", (0.58, -0.08), "crate").arm == "right"
    assert select_arm(ReachTable(), "crate", (0.62, 0.09), "crate").arm == "left"
    assert select_arm(ReachTable(), "crate", (0.58, -0.08), "crate", requested="left").arm == "left"


def test_rack_lane_target_is_flagged_unverified():
    c = select_arm(ReachTable(), "table", (0.10, 0.56), "rack_lane")
    assert c.arm == "left" and not c.verified


def test_from_dict_overrides():
    t = ReachTable.from_dict({"left_table_max_y": 1.0, "right_reaches_table": True})
    assert select_arm(t, "table", (0.0, 0.95), "crate").arm == "left"
    assert select_arm(t, "table", (0.0, 0.5), "crate", requested="right").arm == "right"


def test_unknown_source_or_target():
    assert select_arm(ReachTable(), "floor", (0, 0), "crate").arm is None
    assert select_arm(ReachTable(), "table", (0, 0.5), "bin").arm is None


# ---- waist-first (Layout v2: table left +Y, rack right -Y) ------------------------------------
from g2cell_tasks.reachability import waist_for  # noqa: E402

V2 = ReachTable.from_dict({"waist_first": True, "left_table_max_y": 0.85, "right_reaches_table": False,
                           "rack_lane_arms": {"L1_1": ["right", "left"], "L2_1": ["right"], "L1_2": [], "L2_2": []}})


def test_waist_plan_per_target():
    assert waist_for("table") == ("waist_first", 0.0)        # turn left toward the table
    assert waist_for("rack_lane") == ("waist_first", 0.0)    # turn right toward the rack
    assert waist_for("crate") == ("absolute", 0.0)           # pack-position carton: waist ~0
    assert waist_for("crate", waist_first=False) == ("fixed", 0.0)
    assert waist_for("table", waist_first=False) == ("fixed", 0.0)


def test_table_pick_to_rack_uses_left_arm_with_waist_when_lane_reachable():
    c = select_arm(V2, "table", (0.10, 0.62), "rack_lane", lane="L1_1")
    assert c.arm == "left" and "waist" in c.reason


def test_rack_lane_beyond_reach_is_unreachable():
    c = select_arm(V2, "table", (0.10, 0.62), "rack_lane", lane="L1_2")
    assert c.arm is None and "L1_2" in c.reason
    # L2_1: only the right arm reaches it (waist yawed right), and the right arm cannot pick from the table
    assert select_arm(V2, "table", (0.10, 0.62), "rack_lane", lane="L2_1").arm is None


def test_crate_pick_to_rack_prefers_right_arm():
    assert select_arm(V2, "crate", (0.62, 0.05), "rack_lane", lane="L1_1").arm == "right"


def test_pack_carton_either_arm():
    assert select_arm(V2, "crate", (0.58, -0.08), "crate").arm == "right"
    assert select_arm(V2, "crate", (0.62, 0.08), "crate").arm == "left"


def test_lane_without_entry_falls_back_to_rack_arms():
    c = select_arm(V2, "table", (0.10, 0.62), "rack_lane", lane="L9_9")
    assert c.arm == "left" and not c.verified


def test_shipped_config_reach_is_waist_first():
    import pathlib
    import yaml
    from g2cell_tasks.config import load_task_config
    doc = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config" / "tasks.yaml").read_text())
    cfg = load_task_config(doc)
    assert cfg.reach.waist_first
    assert select_arm(cfg.reach, "table", (0.10, 0.62), "crate").arm == "left"
