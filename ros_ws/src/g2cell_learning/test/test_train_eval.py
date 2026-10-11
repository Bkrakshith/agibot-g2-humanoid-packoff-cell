import csv
import json

import numpy as np
import pytest
import torch

from g2cell_learning.train import TrainConfig, split_episodes, train
from g2cell_learning.eval_offline import evaluate


def test_split_by_episode_is_disjoint_and_deterministic(episode_paths):
    tr, va = split_episodes(episode_paths, val_fraction=0.34, seed=0)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 3 and len(va) == 1
    assert split_episodes(episode_paths, 0.34, seed=0) == (tr, va)


def test_split_train_equals_val(episode_paths):
    tr, va = split_episodes(episode_paths[:1], val_fraction=0.5, train_equals_val=True)
    assert tr == va == [str(episode_paths[0])]


def test_split_needs_two_episodes_unless_train_equals_val(episode_paths):
    with pytest.raises(ValueError):
        split_episodes(episode_paths[:1], val_fraction=0.5)


def test_config_from_cli_roundtrip():
    cfg = TrainConfig.from_args(["--episodes", "a.hdf5", "b.hdf5", "--policy", "bc",
                                 "--steps", "7", "--image-size", "16", "20",
                                 "--extra-cameras", "free_rgb", "--train-equals-val"])
    assert cfg.episodes == ["a.hdf5", "b.hdf5"] and cfg.policy == "bc" and cfg.steps == 7
    assert cfg.image_size == (16, 20) and cfg.extra_cameras == ("free_rgb",)
    assert cfg.train_equals_val is True


@pytest.mark.parametrize("policy", ["act", "bc"])
def test_train_then_eval(tmp_path, episode_paths, policy):
    out = tmp_path / f"run_{policy}"
    cfg = TrainConfig(episodes=[str(p) for p in episode_paths], out_dir=str(out), policy=policy,
                      steps=40, batch_size=8, image_size=(16, 20), chunk_size=4, d_model=32,
                      backbone="small", eval_every=20, ckpt_every=20, val_fraction=0.34, lr=1e-3,
                      num_workers=0)
    res = train(cfg)
    assert (out / "checkpoints" / "last.pt").exists()
    assert (out / "checkpoints" / "step_000020.pt").exists()
    assert (out / "norm_stats.json").exists() and (out / "config.json").exists()
    split = json.loads((out / "split.json").read_text())
    assert set(split["train"]).isdisjoint(split["val"])
    rows = list(csv.DictReader((out / "loss.csv").open()))
    assert len(rows) >= 2 and {"step", "train_loss", "val_loss"} <= set(rows[0])
    assert res["final_train_loss"] < res["first_train_loss"]

    metrics = evaluate(out / "checkpoints" / "last.pt", split["val"], out_dir=tmp_path / "ev",
                       plot_joints=None)
    assert np.isfinite(metrics["action_mse"]) and np.isfinite(metrics["chunk_mse"])
    assert len(metrics["per_joint_mse"]) == 6
    assert (tmp_path / "ev" / "metrics.json").exists()
    pngs = list((tmp_path / "ev").glob("replay_*.png"))
    assert len(pngs) == len(split["val"])
    assert pngs[0].stat().st_size > 1000


def test_resume_from_checkpoint(tmp_path, episode_paths):
    out = tmp_path / "r"
    base = dict(episodes=[str(p) for p in episode_paths], out_dir=str(out), steps=10,
                batch_size=4, image_size=(8, 8), chunk_size=2, d_model=16, eval_every=10,
                ckpt_every=10, num_workers=0)
    train(TrainConfig(**base))
    res = train(TrainConfig(**{**base, "steps": 15, "resume": str(out / "checkpoints/last.pt")}))
    assert res["start_step"] == 10 and res["step"] == 15
    ck = torch.load(out / "checkpoints/last.pt", weights_only=False)
    assert ck["step"] == 15
