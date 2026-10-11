import pathlib
import py_compile

from g2cell_bringup.check_stack import REQUIRED, Graph, evaluate, format_report


def full_graph():
    topics = {t["name"]: t.get("publishers", 1) for t in REQUIRED["topics"]}
    services = {s["name"] for s in REQUIRED["services"]}
    actions = {a["name"] for a in REQUIRED["actions"]}
    return Graph(topics=topics, services=services, actions=actions)


def test_everything_present_is_ok():
    rows, missing = evaluate(REQUIRED, full_graph())
    assert missing == [] and all(r.ok for r in rows)


def test_missing_service_and_action_reported():
    g = full_graph()
    g.services.discard("/compute_ik")
    g.actions.discard("/g2cell/pick_bag")
    _, missing = evaluate(REQUIRED, g)
    assert {"/compute_ik", "/g2cell/pick_bag"} <= {m.name for m in missing}


def test_joint_command_needs_exactly_one_publisher():
    g = full_graph()
    g.topics["/joint_command"] = 2
    _, missing = evaluate(REQUIRED, g)
    bad = [m for m in missing if m.name == "/joint_command"]
    assert bad and "2 publishers" in bad[0].detail


def test_topic_with_zero_publishers_is_missing():
    g = full_graph()
    g.topics["/joint_states"] = 0
    _, missing = evaluate(REQUIRED, g)
    assert "/joint_states" in {m.name for m in missing}


def test_report_mentions_what_provides_it():
    g = full_graph()
    g.services.discard("/compute_ik")
    rows, missing = evaluate(REQUIRED, g)
    text = format_report(rows, missing)
    assert "MISSING" in text and "move_group" in text


def test_required_covers_the_stack():
    names = {x["name"] for k in ("topics", "services", "actions") for x in REQUIRED[k]}
    for n in ("/joint_states", "/joint_command", "/compute_ik", "/compute_fk", "/g2cell/move_arm",
              "/g2cell/pick_bag", "/g2cell/cell_status", "/g2cell/set_line_state", "/g2cell/spawn_bags",
              "/g2cell/episode_control", "/genie_sim/head_front_camera_rgb/image_raw"):
        assert n in names


def test_launch_files_compile():
    for f in (pathlib.Path(__file__).parents[1] / "launch").glob("*.launch.py"):
        py_compile.compile(str(f), doraise=True)
