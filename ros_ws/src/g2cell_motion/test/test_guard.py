from g2cell_motion.guard import publisher_conflicts


def test_no_conflict_when_alone():
    assert publisher_conflicts([], own=("/", "g2cell_motion_server")) == []
    assert publisher_conflicts([("/", "g2cell_motion_server")], own=("/", "g2cell_motion_server")) == []


def test_other_publishers_reported():
    infos = [("/", "g2cell_keyboard_teleop"), ("/", "g2cell_motion_server"), ("/ns", "x")]
    assert publisher_conflicts(infos, own=("/", "g2cell_motion_server")) == ["/g2cell_keyboard_teleop", "/ns/x"]


def test_same_name_in_other_namespace_is_a_conflict():
    assert publisher_conflicts([("/a", "n")], own=("/", "n")) == ["/a/n"]
