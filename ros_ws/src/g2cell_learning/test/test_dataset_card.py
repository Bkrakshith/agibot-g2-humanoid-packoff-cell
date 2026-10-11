import json

import pytest

from g2cell_learning.dataset_card import build_card, write_card


def test_card_fields(tmp_path, episode_paths, no_action_episode):
    card = build_card(episode_paths + [no_action_episode], batch_name="b0")
    assert card["batch_name"] == "b0"
    assert card["num_episodes"] == 4
    assert card["num_episodes_with_actions"] == 3
    # success attr: true, false, true, unknown -> 2/3 of labelled episodes
    assert card["success"]["labelled"] == 3
    assert card["success"]["rate"] == pytest.approx(2 / 3)
    assert card["duration_s"]["mean"] == pytest.approx(
        sum(e["duration_s"] for e in card["episodes"]) / 4)
    assert card["scenes"] == {"scene_a": 3, "scene_b": 1}
    assert sorted(card["seeds"]) == [0, 0, 1, 2]
    assert card["aligned_rate_hz"] == [30.0]
    assert "joint_states" in card["stream_rates_hz"]["mean"]
    assert card["cameras"]["head_front_rgb"]["shape"] == [24, 32, 3]
    assert card["action_dim"] == 6 and len(card["action_joint_names"]) == 6
    assert card["total_aligned_frames"] > 0

    md = write_card(card, tmp_path / "card")
    assert (tmp_path / "card.json").exists() and md.exists()
    assert json.loads((tmp_path / "card.json").read_text())["num_episodes"] == 4
    text = md.read_text()
    assert "scene_a" in text and "Success rate" in text
