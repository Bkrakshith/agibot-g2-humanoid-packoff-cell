from g2cell_tasks.cell_state import NOT_IMPLEMENTED, OK, REJECTED, CellState


def state():
    return CellState(crate_fill_target=10, bags_on_table=3, crate_at_pack_position=True)


def test_line_and_spawn_are_not_implemented():
    s = state()
    for cmd in (0, 1, 2):
        st, msg = s.line_command(cmd)
        assert st == NOT_IMPLEMENTED and "Stage 3" in msg
    st, msg = s.spawn_bags(5, 0)
    assert st == NOT_IMPLEMENTED and "Stage 5" in msg
    assert not s.line_running


def test_episode_lifecycle():
    s = state()
    assert s.episode(1, "")[0] == REJECTED              # stop before start
    assert s.episode(2, "")[0] == REJECTED              # mark before any episode
    st, _, eid = s.episode(0, "", now="20261011_120000")
    assert st == OK and eid == "ep_20261011_120000" and s.episode_active
    assert s.episode(0, "x")[0] == REJECTED             # already running
    assert s.episode(2, "")[0] == OK and s.mark == "success"
    assert s.episode(1, "")[0] == OK and not s.episode_active
    assert s.episode(3, "")[0] == OK and s.mark == "failure"   # mark after stop is fine
    st, _, eid = s.episode(0, "ep_b")
    assert eid == "ep_b" and s.mark is None and s.picks_done == 0


def test_unknown_episode_command():
    assert state().episode(9, "")[0] == REJECTED


def test_pick_results_update_counts():
    s = state()
    s.episode(0, "e")
    s.on_pick_result("success", True, "crate")
    s.on_pick_result("unverified", False, "crate")      # not counted in the crate
    s.on_pick_result("plan_failed", False, "crate")     # bag never left the table
    s.on_pick_result("success", True, "rack_lane")
    assert s.crate_fill_count == 1 and s.picks_done == 4 and s.bags_on_table == 0


def test_crate_full():
    s = CellState(crate_fill_target=1, bags_on_table=5)
    assert not s.crate_full
    s.on_pick_result("overflow", True, "crate")
    assert s.crate_full


def test_not_implemented_list_mentions_stages():
    joined = " ".join(state().not_implemented())
    assert "Stage 3" in joined and "Stage 5" in joined
