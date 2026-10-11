import json

import h5py
import numpy as np
import pytest
import torch

from conftest import CMD_NAMES, cmd_value
from g2cell_learning.dataset import (
    DatasetConfig,
    EpisodeDataset,
    NormStats,
    compute_norm_stats,
    load_episode_arrays,
)


def test_load_arrays_selects_commanded_joints_from_states_by_name(episode_path):
    arr = load_episode_arrays(episode_path)
    assert arr.joint_names == CMD_NAMES
    assert arr.state.shape == arr.action.shape
    assert arr.state.shape[1] == len(CMD_NAMES)
    with h5py.File(episode_path) as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["joint_states"].attrs["names"]]
        pos = f["aligned/joint_states/positions"][:]
    for j, n in enumerate(CMD_NAMES):
        np.testing.assert_allclose(arr.state[:, j], pos[:, names.index(n)])


def test_load_arrays_has_no_nans_leading_nan_actions_filled_from_state(episode_path):
    arr = load_episode_arrays(episode_path)
    assert np.isfinite(arr.state).all() and np.isfinite(arr.action).all()
    with h5py.File(episode_path) as f:
        raw = f["aligned/joint_command/actions"][:]
    # the fixture leaves the right arm uncommanded for the first aligned step(s)
    j = CMD_NAMES.index("right_j1")
    lead = np.isnan(raw[:, j])
    assert lead[0]
    np.testing.assert_allclose(arr.action[lead, j], arr.state[lead, j])


def test_episode_without_actions_raises_clear_error(no_action_episode):
    with pytest.raises(ValueError, match="joint_command"):
        load_episode_arrays(no_action_episode)


def test_item_shapes_and_keys(episode_path):
    cfg = DatasetConfig(image_size=(16, 20), chunk_size=5, extra_cameras=("free_rgb",))
    ds = EpisodeDataset([episode_path], cfg)
    item = ds[0]
    assert item["observation.images.head_front_rgb"].shape == (3, 16, 20)
    assert item["observation.images.free_rgb"].shape == (3, 16, 20)
    img = item["observation.images.head_front_rgb"]
    assert img.dtype == torch.float32 and 0.0 <= img.min() and img.max() <= 1.0
    assert item["observation.state"].shape == (len(CMD_NAMES),)
    assert item["action"].shape == (5, len(CMD_NAMES))
    assert item["action_is_pad"].shape == (5,) and item["action_is_pad"].dtype == torch.bool
    assert int(item["episode_index"]) == 0 and int(item["frame_index"]) == 0


def test_len_is_total_aligned_frames(episode_paths):
    ds = EpisodeDataset(episode_paths, DatasetConfig(image_size=(8, 8), chunk_size=3))
    lens = [len(load_episode_arrays(p).t_ns) for p in episode_paths]
    assert len(ds) == sum(lens)
    last = ds[len(ds) - 1]
    assert int(last["episode_index"]) == 2 and int(last["frame_index"]) == lens[2] - 1


def test_action_chunk_is_next_k_commands_and_padded_at_end(episode_path):
    k = 4
    cfg = DatasetConfig(image_size=(8, 8), chunk_size=k, normalize=False, action_offset=1)
    ds = EpisodeDataset([episode_path], cfg)
    arr = load_episode_arrays(episode_path)
    n = len(arr.t_ns)
    item = ds[3]
    np.testing.assert_allclose(item["action"].numpy(), arr.action[4:8], rtol=1e-5, atol=1e-6)
    assert not item["action_is_pad"].any()
    end = ds[n - 2]  # only one future step left
    assert end["action_is_pad"].tolist() == [False, True, True, True]
    np.testing.assert_allclose(end["action"][0].numpy(), arr.action[n - 1], rtol=1e-5, atol=1e-6)
    # padded rows repeat the last real action (harmless if the mask is ignored)
    np.testing.assert_allclose(end["action"][3].numpy(), arr.action[n - 1], rtol=1e-5, atol=1e-6)
    last = ds[n - 1]
    assert last["action_is_pad"].all()


def test_offset_zero_starts_at_current_step(episode_path):
    cfg = DatasetConfig(image_size=(8, 8), chunk_size=2, normalize=False, action_offset=0)
    ds = EpisodeDataset([episode_path], cfg)
    arr = load_episode_arrays(episode_path)
    np.testing.assert_allclose(ds[5]["action"].numpy(), arr.action[5:7], rtol=1e-5, atol=1e-6)


def test_commanded_values_match_ground_truth(episode_path):
    # sanity check that the aligned command is the latest command (nearest-previous)
    arr = load_episode_arrays(episode_path)
    t = (arr.t_ns - arr.t_ns[0]) / 1e9
    j = CMD_NAMES.index("left_j1")
    # commands are 20 Hz, so the held value lags truth by < 0.05 s + alignment offset
    truth = np.array([cmd_value("left_j1", x) for x in t + (arr.t_ns[0] - 1e9) / 1e9])
    assert np.max(np.abs(arr.action[:, j] - truth)) < 0.02


def test_norm_stats_roundtrip_and_normalisation(tmp_path, episode_paths):
    stats = compute_norm_stats(episode_paths)
    assert stats.state_mean.shape == (len(CMD_NAMES),)
    assert (stats.action_std > 0).all() and (stats.state_std > 0).all()
    p = tmp_path / "norm.json"
    stats.save(p)
    data = json.loads(p.read_text())
    assert data["joint_names"] == CMD_NAMES
    back = NormStats.load(p)
    np.testing.assert_allclose(back.action_mean, stats.action_mean)

    ds = EpisodeDataset(episode_paths, DatasetConfig(image_size=(8, 8), chunk_size=1,
                                                     action_offset=0), stats=back)
    acts = torch.stack([ds[i]["action"][0] for i in range(len(ds))])
    np.testing.assert_allclose(acts.mean(0).numpy(), 0, atol=1e-4)
    np.testing.assert_allclose(acts.std(0, unbiased=False).numpy(), 1, atol=1e-3)
    raw = back.unnormalize_action(acts.numpy())
    assert raw.shape == acts.shape


def test_constant_joint_gets_floor_std(tmp_path, episode_path):
    stats = compute_norm_stats([episode_path], std_floor=1e-2)
    assert (stats.action_std >= 1e-2).all()


def test_cached_images_match_uncached(episode_path):
    a = EpisodeDataset([episode_path], DatasetConfig(image_size=(12, 16), cache_images=True))
    b = EpisodeDataset([episode_path], DatasetConfig(image_size=(12, 16), cache_images=False))
    for i in (0, 7, len(a) - 1):
        torch.testing.assert_close(a[i]["observation.images.head_front_rgb"],
                                   b[i]["observation.images.head_front_rgb"])


def test_dataloader_batches(episode_paths):
    ds = EpisodeDataset(episode_paths, DatasetConfig(image_size=(8, 8), chunk_size=3))
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    assert batch["action"].shape == (4, 3, len(CMD_NAMES))
    assert batch["observation.images.head_front_rgb"].shape == (4, 3, 8, 8)
