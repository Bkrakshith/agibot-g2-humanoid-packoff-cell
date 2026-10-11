"""Server-oriented training features: config files, cache, AMP choice, accumulation,
time-based checkpoints + resume, stop requests, early stopping and the dry run."""
import csv
import json
from pathlib import Path

import pytest
import torch

from g2cell_learning.train import (
    TrainConfig,
    load_config_file,
    parse_simple_yaml,
    resolve_amp,
    train,
)
from g2cell_learning import train as train_mod

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _base(tmp_path, episode_paths, **kw):
    d = dict(episodes=[str(p) for p in episode_paths], out_dir=str(tmp_path / "run"), steps=12,
             batch_size=4, image_size=(8, 8), chunk_size=2, d_model=16, eval_every=4,
             ckpt_every=100, val_fraction=0.34, num_workers=0, lr=1e-3)
    d.update(kw)
    return TrainConfig(**d)


# ---------------------------------------------------------------- config files

def test_simple_yaml_subset():
    text = """
# comment
policy: act          # trailing comment
steps: 30000
lr: 3.0e-4
compile: false
amp: "bf16"
image_size: [160, 256]
episodes:
  - output/episodes/*/*.hdf5
  - extra.hdf5
extra_cameras: []
"""
    d = parse_simple_yaml(text)
    assert d == {"policy": "act", "steps": 30000, "lr": 3e-4, "compile": False, "amp": "bf16",
                 "image_size": [160, 256],
                 "episodes": ["output/episodes/*/*.hdf5", "extra.hdf5"], "extra_cameras": []}


def test_config_file_then_cli_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("steps: 50\nbatch_size: 64\nimage_size: [16, 20]\npolicy: bc\n")
    cfg = TrainConfig.from_args(["--config", str(p), "--steps", "7", "--episodes", "a.hdf5"])
    assert cfg.steps == 7 and cfg.batch_size == 64 and cfg.policy == "bc"
    assert cfg.image_size == (16, 20) and cfg.episodes == ["a.hdf5"]


def test_unknown_config_key_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stepz: 5\n")
    with pytest.raises(ValueError, match="stepz"):
        load_config_file(p)


def test_server_config_loads_with_server_defaults():
    cfg = TrainConfig.from_args(["--config", str(CONFIGS / "server_a6000.yaml")])
    assert cfg.device == "auto" and cfg.amp == "auto"
    assert cfg.batch_size * cfg.grad_accum >= 128
    assert cfg.num_workers >= 4 and cfg.pin_memory and cfg.persistent_workers
    assert cfg.cache_dir and cfg.ckpt_minutes > 0 and cfg.early_stop_patience > 0
    assert cfg.resume == "auto"


def test_episode_globs_are_expanded(tmp_path, episode_paths):
    cfg = _base(tmp_path, [], episodes=[str(Path(episode_paths[0]).parent / "ep*.hdf5")])
    assert sorted(train_mod.expand_episodes(cfg.episodes)) == sorted(map(str, episode_paths))
    with pytest.raises(FileNotFoundError):
        train_mod.expand_episodes([str(tmp_path / "none*.hdf5")])


# ---------------------------------------------------------------- AMP choice

def test_resolve_amp():
    assert resolve_amp("auto", "cpu") is None                       # fp32 fallback on CPU
    assert resolve_amp("bf16", "cpu") is None
    assert resolve_amp("auto", "cuda", bf16_ok=True) == torch.bfloat16
    assert resolve_amp("auto", "cuda", bf16_ok=False) == torch.float16
    assert resolve_amp("fp16", "cuda", bf16_ok=True) == torch.float16
    assert resolve_amp("off", "cuda", bf16_ok=True) is None
    with pytest.raises(ValueError):
        resolve_amp("fp8", "cuda")


# ---------------------------------------------------------------- training loop

def test_cached_training_with_workers_accum_and_channels_last(tmp_path, episode_paths):
    cfg = _base(tmp_path, episode_paths, cache_dir=str(tmp_path / "cache"), num_workers=2,
                grad_accum=2, channels_last=True, steps=30, eval_every=10)
    res = train(cfg)
    out = Path(cfg.out_dir)
    assert res["final_train_loss"] < res["first_train_loss"]
    assert (out / "checkpoints" / "last.pt").exists()
    rows = list(csv.DictReader((out / "loss.csv").open()))
    assert {"step", "train_loss", "val_loss", "samples_per_s"} <= set(rows[0])
    assert res["samples_seen"] == 30 * 4 * 2
    assert list((tmp_path / "cache").glob("*/meta.json"))


def test_time_based_checkpoint_and_auto_resume(tmp_path, episode_paths):
    cfg = _base(tmp_path, episode_paths, ckpt_minutes=1e-9, steps=6)
    res = train(cfg)
    assert res["time_checkpoints"] >= 1
    res2 = train(_base(tmp_path, episode_paths, steps=9, resume="auto"))
    assert res2["start_step"] == 6 and res2["step"] == 9
    # resume="auto" with nothing to resume starts fresh
    res3 = train(_base(tmp_path, episode_paths, out_dir=str(tmp_path / "fresh"), steps=2,
                       resume="auto"))
    assert res3["start_step"] == 0


def test_stop_request_saves_and_resumes(tmp_path, episode_paths):
    res = train(_base(tmp_path, episode_paths, steps=50), should_stop=lambda step: step >= 7)
    assert res["stopped"] and res["step"] == 7
    ck = torch.load(Path(res["out_dir"]) / "checkpoints" / "last.pt", weights_only=False)
    assert ck["step"] == 7 and "rng" in ck
    res2 = train(_base(tmp_path, episode_paths, steps=10, resume="auto"))
    assert res2["start_step"] == 7 and not res2["stopped"]


def test_early_stopping_on_val_loss(tmp_path, episode_paths):
    res = train(_base(tmp_path, episode_paths, lr=0.0, steps=100, eval_every=2,
                      early_stop_patience=2))
    assert res["early_stopped"] and res["step"] == 6       # best at 2, no gain at 4 and 6
    assert (Path(res["out_dir"]) / "checkpoints" / "best.pt").exists()


def test_dry_run_estimates_without_training(tmp_path, episode_paths, capsys):
    out = tmp_path / "dry"
    train_mod.main(["--episodes", *map(str, episode_paths), "--out-dir", str(out),
                    "--image-size", "8", "8", "--chunk-size", "2", "--d-model", "16",
                    "--batch-size", "4", "--steps", "1000", "--dry-run", "--bench-steps", "3",
                    "--cache-dir", str(tmp_path / "cache"), "--price-per-hour", "0.8"])
    rep = json.loads((out / "dry_run.json").read_text())
    for k in ("samples_per_s", "loader_samples_per_s", "step_s", "train_frames",
              "est_epoch_s", "est_total_s", "est_cost", "device", "amp"):
        assert k in rep, k
    assert rep["est_total_s"] == pytest.approx(rep["step_s"] * 1000, rel=1e-6)
    assert not (out / "checkpoints" / "last.pt").exists()
    assert "est" in capsys.readouterr().out
