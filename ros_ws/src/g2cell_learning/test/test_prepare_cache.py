import json
import os

import numpy as np
import pytest
import torch

from conftest import CMD_NAMES, write_episode
from g2cell_learning.dataset import (
    CachedEpisodeDataset,
    DatasetConfig,
    EpisodeDataset,
    compute_norm_stats,
)
from g2cell_learning.prepare_cache import (
    cache_dir_for,
    load_cached_stats,
    main as prepare_main,
    prepare_cache,
)

CAMS = ("head_front_rgb", "free_rgb")


def test_cache_layout_and_dtypes(tmp_path, episode_paths):
    dirs = prepare_cache(episode_paths, tmp_path / "cache", image_size=(12, 16), cameras=CAMS,
                         workers=0)
    assert len(dirs) == 3
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["joint_names"] == CMD_NAMES and meta["image_size"] == [12, 16]
    assert meta["source"] == str(episode_paths[0])
    for cam in CAMS:
        frames = np.load(dirs[0] / f"{cam}.npy", mmap_mode="r")
        lut = np.load(dirs[0] / f"{cam}_lut.npy")
        assert frames.dtype == np.uint8 and frames.shape[1:] == (3, 12, 16)
        assert lut.shape == (meta["length"],) and lut.max() < frames.shape[0]
        assert frames.shape[0] <= meta["length"]
    # unique frames only: the 15 Hz free camera has about half as many rows as aligned steps
    assert np.load(dirs[0] / "free_rgb.npy", mmap_mode="r").shape[0] < meta["length"] * 0.6
    low = np.load(dirs[0] / "lowdim.npz")
    assert low["state"].shape == low["action"].shape == (meta["length"], len(CMD_NAMES))


def test_cached_items_match_the_hdf5_dataset(tmp_path, episode_paths):
    cfg = DatasetConfig(image_size=(12, 16), chunk_size=4, extra_cameras=("free_rgb",))
    ref = EpisodeDataset(episode_paths, cfg)
    dirs = prepare_cache(episode_paths, tmp_path / "c", image_size=(12, 16), cameras=CAMS,
                         workers=2)
    ds = CachedEpisodeDataset(dirs, cfg, ref.stats)
    assert len(ds) == len(ref) and ds.joint_names == ref.joint_names
    for i in (0, 5, len(ds) // 2, len(ds) - 1):
        a, b = ref[i], ds[i]
        assert set(a) == set(b)
        for k in ("observation.state", "action", "action_is_pad", "episode_index",
                  "frame_index"):
            torch.testing.assert_close(a[k], b[k])
        for cam in CAMS:
            img = b[f"observation.images.{cam}"]
            assert img.dtype == torch.uint8 and img.shape == (3, 12, 16)
            torch.testing.assert_close(img.float() / 255, a[f"observation.images.{cam}"],
                                       atol=1 / 255 + 1e-6, rtol=0)


def test_workers_do_not_change_the_cache(tmp_path, episode_paths):
    d0 = prepare_cache(episode_paths[:1], tmp_path / "a", (12, 16), CAMS, workers=0)[0]
    d2 = prepare_cache(episode_paths[:1], tmp_path / "b", (12, 16), CAMS, workers=2)[0]
    for cam in CAMS:
        np.testing.assert_array_equal(np.load(d0 / f"{cam}.npy"), np.load(d2 / f"{cam}.npy"))


def test_stats_from_cache_equal_compute_norm_stats(tmp_path, episode_paths):
    dirs = prepare_cache(episode_paths, tmp_path / "c", (8, 8), ("head_front_rgb",), workers=0)
    want = compute_norm_stats(episode_paths[:2])
    got = load_cached_stats(dirs[:2])
    for k in ("state_mean", "state_std", "action_mean", "action_std", "state_min", "action_max"):
        np.testing.assert_allclose(getattr(got, k), getattr(want, k), rtol=1e-9, atol=1e-12)
    assert got.joint_names == want.joint_names


def test_up_to_date_cache_is_reused_and_stale_one_rebuilt(tmp_path, episode_path):
    root = tmp_path / "c"
    d = prepare_cache([episode_path], root, (8, 8), ("head_front_rgb",), workers=0)[0]
    stamp = (d / "meta.json").stat().st_mtime_ns
    assert prepare_cache([episode_path], root, (8, 8), ("head_front_rgb",))[0] == d
    assert (d / "meta.json").stat().st_mtime_ns == stamp
    # another image size -> another cache dir
    assert cache_dir_for(episode_path, root, (8, 10), ("head_front_rgb",)) != d
    # the source changes (re-converted episode) -> new key
    os.utime(episode_path, ns=(1, 1))
    assert cache_dir_for(episode_path, root, (8, 8), ("head_front_rgb",)) != d


def test_missing_camera_raises(tmp_path, episode_path):
    with pytest.raises(ValueError, match="camera"):
        prepare_cache([episode_path], tmp_path / "c", (8, 8), ("nope",), workers=0)


def test_dataloader_workers_with_memmaps(tmp_path, episode_paths):
    cfg = DatasetConfig(image_size=(8, 8), chunk_size=3)
    dirs = prepare_cache(episode_paths, tmp_path / "c", (8, 8), ("head_front_rgb",), workers=0)
    ds = CachedEpisodeDataset(dirs, cfg, load_cached_stats(dirs))
    dl = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True, num_workers=2)
    n = sum(b["action"].shape[0] for b in dl)
    assert n == len(ds)


def test_cli_writes_norm_stats(tmp_path, episode_paths):
    out = tmp_path / "c"
    rc = prepare_main([*map(str, episode_paths), "--cache-dir", str(out), "--image-size", "8",
                       "8", "--extra-cameras", "free_rgb", "--workers", "0"])
    assert rc == 0
    stats = json.loads((out / "norm_stats.json").read_text())
    assert stats["joint_names"] == CMD_NAMES
    assert len(list(out.glob("*/meta.json"))) == 3


def test_episode_without_actions_is_rejected(tmp_path, no_action_episode):
    with pytest.raises(ValueError, match="joint_command"):
        prepare_cache([no_action_episode], tmp_path / "c", (8, 8), ("head_front_rgb",))


def test_policy_accepts_uint8_images():
    from g2cell_learning.policy import build_policy
    torch.manual_seed(0)
    pol = build_policy("act", 6, 6, 3, d_model=32).eval()
    img = torch.randint(0, 256, (2, 3, 16, 20), dtype=torch.uint8)
    st = torch.randn(2, 6)
    a = pol({"observation.state": st, "observation.images.head_front_rgb": img})
    b = pol({"observation.state": st, "observation.images.head_front_rgb": img.float() / 255})
    torch.testing.assert_close(a, b)


def test_write_episode_fixture_still_works(tmp_path):
    assert write_episode(tmp_path / "x.hdf5").exists()
