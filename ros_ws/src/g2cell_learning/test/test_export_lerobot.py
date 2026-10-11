import json

import av
import numpy as np
import pyarrow.parquet as pq

from conftest import CMD_NAMES
from g2cell_learning.dataset import load_episode_arrays
from g2cell_learning.export_lerobot import export_lerobot


def test_export_layout(tmp_path, episode_paths):
    root = tmp_path / "lr"
    info = export_lerobot(episode_paths, root, task="pack salad bags", fps=30,
                          cameras={"head_front_rgb": "head", "free_rgb": "free"})
    meta = json.loads((root / "meta/info.json").read_text())
    assert meta == info
    assert meta["codebase_version"] == "v2.0"
    assert meta["total_episodes"] == 3 and meta["total_tasks"] == 1 and meta["fps"] == 30
    assert meta["total_videos"] == 6 and meta["total_chunks"] == 1
    assert meta["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert meta["video_path"] == ("videos/chunk-{episode_chunk:03d}/{video_key}/"
                                  "episode_{episode_index:06d}.mp4")
    feats = meta["features"]
    for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index",
              "index", "task_index", "observation.images.head", "observation.images.free"):
        assert k in feats, k
    assert feats["action"]["shape"] == [6]
    assert feats["action"]["names"]["motors"] == CMD_NAMES
    assert feats["observation.images.head"]["dtype"] == "video"
    assert feats["observation.images.head"]["shape"][:2] == [24, 32]

    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l) for l in (root / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "pack salad bags"}]
    assert [e["episode_index"] for e in eps] == [0, 1, 2]
    assert all(e["tasks"] == ["pack salad bags"] for e in eps)
    assert sum(e["length"] for e in eps) == meta["total_frames"]
    assert (root / "meta/stats.json").exists()

    total = 0
    for e, p in zip(eps, episode_paths):
        i = e["episode_index"]
        t = pq.read_table(root / f"data/chunk-000/episode_{i:06d}.parquet")
        assert t.num_rows == e["length"]
        cols = set(t.column_names)
        assert {"observation.state", "action", "timestamp", "frame_index", "episode_index",
                "index", "task_index"} <= cols
        d = t.to_pydict()
        arr = load_episode_arrays(p)
        np.testing.assert_allclose(np.array(d["action"]), arr.action, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.array(d["observation.state"]), arr.state, rtol=1e-6,
                                   atol=1e-6)
        assert d["frame_index"] == list(range(e["length"]))
        assert d["episode_index"] == [i] * e["length"]
        assert d["index"] == list(range(total, total + e["length"]))
        np.testing.assert_allclose(d["timestamp"], np.arange(e["length"]) / 30, atol=1e-4)
        total += e["length"]
        for key in ("head", "free"):
            mp4 = root / f"videos/chunk-000/observation.images.{key}/episode_{i:06d}.mp4"
            with av.open(str(mp4)) as c:
                n = sum(1 for _ in c.decode(video=0))
            assert n == e["length"]


def test_export_skips_episode_without_actions(tmp_path, episode_path, no_action_episode):
    info = export_lerobot([episode_path, no_action_episode], tmp_path / "lr2", task="t")
    assert info["total_episodes"] == 1
