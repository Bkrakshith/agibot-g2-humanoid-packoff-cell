"""Train ACT-lite or BC-MLP on episode HDF5 files (CPU or GPU).

    python -m g2cell_learning.train --episodes ep*.hdf5 --out-dir output/train/run1 --policy act
    python -m g2cell_learning.train --config ros_ws/src/g2cell_learning/configs/server_a6000.yaml
    python -m g2cell_learning.train --config ... --dry-run      # estimate before paying GPU hours

Outputs in ``out_dir``: config.json, split.json (episodes per split), norm_stats.json,
loss.csv (step, train_loss, val_loss, lr, elapsed_s, samples_per_s),
checkpoints/step_XXXXXX.pt, last.pt (also every --ckpt-minutes and on SIGTERM) and best.pt
(lowest val loss). --resume auto continues from out_dir/checkpoints/last.pt if it exists.

GPU path: --cache-dir (uint8 memmap frames, see prepare_cache.py), DataLoader workers with
pinned memory / persistent workers / prefetch, autocast bf16 (fp16 + GradScaler where bf16 is
unsupported; fp32 on CPU), channels_last, optional torch.compile, gradient accumulation.
Config files (--config) hold any of the flags below as ``key: value`` (YAML; a flat subset is
parsed without PyYAML); command-line flags override them. Relative paths are relative to the
current directory.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import glob
import json
import os
import random
import signal
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .dataset import (
    CachedEpisodeDataset,
    DatasetConfig,
    EpisodeDataset,
    NormStats,
    compute_norm_stats,
)
from .policy import build_policy, count_parameters, masked_loss

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


@dataclass
class TrainConfig:
    episodes: list[str] = field(default_factory=list)   # paths or glob patterns
    out_dir: str = "output/train/run"
    config: str = ""                    # YAML file with any of these fields (CLI overrides)
    policy: str = "act"                 # act | bc
    backbone: str = "small"             # small | resnet18 (weights=None)
    d_model: int = 128
    chunk_size: int = 20
    action_offset: int = 1
    image_size: tuple[int, int] = (128, 160)
    extra_cameras: tuple[str, ...] = ()
    steps: int = 2000                   # optimizer steps
    batch_size: int = 16                # per micro-batch; effective = batch_size * grad_accum
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    loss: str = "l1"                    # l1 (ACT) | mse
    grad_clip: float = 10.0
    val_fraction: float = 0.2
    val_episodes: list[str] = field(default_factory=list)  # explicit val set (overrides fraction)
    train_equals_val: bool = False      # smoke tests with a single episode
    eval_every: int = 100
    val_batches: int = 10               # val loss over a fixed subset of val_batches batches
    ckpt_every: int = 500               # numbered checkpoints every N steps
    ckpt_minutes: float = 10.0          # last.pt at least this often (0 = off)
    early_stop_patience: int = 0        # evals without val improvement before stopping (0 = off)
    early_stop_min_delta: float = 0.0
    seed: int = 0
    device: str = "auto"                # auto | cpu | cuda | cuda:N
    amp: str = "auto"                   # auto (bf16, else fp16 on CUDA; fp32 on CPU) | bf16 | fp16 | off
    compile: bool = False               # torch.compile the policy
    channels_last: bool = True
    cache_dir: str = ""                 # uint8 frame cache (prepare_cache.py); "" = read HDF5
    cache_workers: int = 0              # processes for building the cache (0 = in-process)
    num_workers: int = 0
    pin_memory: bool = True             # only used on CUDA
    persistent_workers: bool = True     # only with num_workers > 0
    prefetch_factor: int = 4            # only with num_workers > 0
    threads: int = 0                    # torch CPU threads (0 = torch default)
    resume: str = ""                    # checkpoint path | auto (out_dir/checkpoints/last.pt)
    dry_run: bool = False               # benchmark a few steps, estimate epoch/run time, exit
    bench_steps: int = 20
    price_per_hour: float = 0.0         # for the dry-run cost estimate

    @classmethod
    def _parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        for f in dataclasses.fields(cls):
            flag = "--" + f.name.replace("_", "-")
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            if f.type in ("bool", bool):
                p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
            elif f.name in ("episodes", "val_episodes"):
                p.add_argument(flag, nargs="+", default=default)
            elif f.name == "image_size":
                p.add_argument(flag, nargs=2, type=int, default=default, metavar=("H", "W"))
            elif f.name == "extra_cameras":
                p.add_argument(flag, nargs="*", default=default)
            else:
                typ = {"int": int, "float": float, "str": str}.get(str(f.type), type(default))
                p.add_argument(flag, type=typ, default=default)
        return p

    @classmethod
    def from_args(cls, argv=None) -> "TrainConfig":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", default="")
        known, _ = pre.parse_known_args(argv)
        p = cls._parser()
        if known.config:
            p.set_defaults(**load_config_file(known.config))
        d = vars(p.parse_args(argv))
        d["image_size"] = tuple(d["image_size"])
        d["extra_cameras"] = tuple(d["extra_cameras"])
        return cls(**d)

    def dataset_config(self, normalize=True) -> DatasetConfig:
        return DatasetConfig(image_size=tuple(self.image_size),
                             extra_cameras=tuple(self.extra_cameras),
                             chunk_size=self.chunk_size, action_offset=self.action_offset,
                             normalize=normalize)


# ---------------------------------------------------------------- config files

def _scalar(v: str):
    v = v.strip()
    if not v:
        return ""
    if v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_simple_yaml(text: str) -> dict:
    """Flat YAML subset: ``key: scalar``, ``key: [a, b]`` and ``key:`` + ``  - item`` lines."""
    out: dict = {}
    key = None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("- "):
            if key is None or not isinstance(out.get(key), list):
                raise ValueError(f"list item without a key: {raw!r}")
            out[key].append(_scalar(s[2:]))
            continue
        if ":" not in s:
            raise ValueError(f"cannot parse config line {raw!r}")
        key, val = s.split(":", 1)
        key, val = key.strip(), val.strip()
        if val.startswith("["):
            if not val.endswith("]"):
                raise ValueError(f"unterminated list for {key}")
            inner = val[1:-1].strip()
            out[key] = [_scalar(x) for x in inner.split(",")] if inner else []
        elif val == "":
            out[key] = []
        else:
            out[key] = _scalar(val)
    return out


def load_config_file(path) -> dict:
    text = Path(path).read_text()
    try:
        import yaml
        d = yaml.safe_load(text) or {}
    except ImportError:
        d = parse_simple_yaml(text)
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}")
    if "image_size" in d:
        d["image_size"] = tuple(d["image_size"])
    if "extra_cameras" in d:
        d["extra_cameras"] = tuple(d["extra_cameras"] or ())
    for k in ("episodes", "val_episodes"):
        if k in d and isinstance(d[k], str):
            d[k] = [d[k]]
    d["config"] = str(path)
    return d


def expand_episodes(patterns) -> list[str]:
    out = []
    for p in patterns:
        if any(c in str(p) for c in "*?["):
            hits = sorted(glob.glob(str(p)))
            if not hits:
                raise FileNotFoundError(f"no episode files match {p}")
            out += hits
        else:
            out.append(str(p))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- helpers

def split_episodes(paths, val_fraction: float = 0.2, seed: int = 0,
                   train_equals_val: bool = False, val_episodes=()):
    """Split by whole episode (never by frame, which would leak near-identical frames)."""
    paths = [str(p) for p in paths]
    if train_equals_val:
        return list(paths), list(paths)
    if val_episodes:
        val = [str(p) for p in val_episodes]
        return [p for p in paths if p not in val], val
    if len(paths) < 2:
        raise ValueError("need >= 2 episodes for a train/val split "
                         "(use --train-equals-val for a smoke test)")
    order = sorted(paths)
    random.Random(seed).shuffle(order)
    n_val = min(len(order) - 1, max(1, round(len(order) * val_fraction)))
    val = sorted(order[:n_val])
    return [p for p in paths if p not in val], val


def resolve_amp(amp: str, device_type: str, bf16_ok: bool | None = None):
    """autocast dtype for this device, or None for fp32."""
    if amp not in ("auto", "bf16", "fp16", "off"):
        raise ValueError(f"amp must be auto, bf16, fp16 or off, got {amp!r}")
    if device_type != "cuda" or amp == "off":
        return None
    if bf16_ok is None:
        bf16_ok = torch.cuda.is_bf16_supported()
    if amp == "fp16" or (amp == "auto" and not bf16_ok):
        return torch.float16
    if amp == "bf16" and not bf16_ok:
        warnings.warn("bf16 not supported on this GPU; using fp16")
        return torch.float16
    return torch.bfloat16


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _infinite(loader):
    while True:
        yield from loader


def _worker_init(_):
    torch.set_num_threads(1)


def build_from_checkpoint(ckpt: dict) -> torch.nn.Module:
    c = ckpt["config"]
    m = build_policy(c["policy"], ckpt["state_dim"], ckpt["action_dim"], c["chunk_size"],
                     cameras=ckpt["cameras"], backbone=c["backbone"], d_model=c["d_model"])
    m.load_state_dict(ckpt["model"])
    return m


def _to_device(b: dict, device, channels_last: bool) -> dict:
    out = {}
    for k, v in b.items():
        v = v.to(device, non_blocking=True)
        if channels_last and v.dim() == 4:
            v = v.contiguous(memory_format=torch.channels_last)
        out[k] = v
    return out


def _autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def _val_loss(model, loader, n_batches, loss_kind, device, amp_dtype=None,
              channels_last=False) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        b = _to_device(b, device, channels_last)
        with _autocast(device, amp_dtype):
            tot += masked_loss(model(b), b["action"], b["action_is_pad"], loss_kind).float().item()
        n += 1
    model.train()
    return tot / max(n, 1)


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _make_loader(ds, cfg: TrainConfig, device, shuffle: bool, generator, drop_last=False):
    nw = cfg.num_workers
    kw = dict(batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=nw,
              generator=generator, pin_memory=bool(cfg.pin_memory and device.type == "cuda"))
    if nw > 0:
        kw.update(persistent_workers=cfg.persistent_workers, prefetch_factor=cfg.prefetch_factor,
                  worker_init_fn=_worker_init)
    return torch.utils.data.DataLoader(ds, **kw)


def _resume_path(cfg: TrainConfig) -> str:
    if cfg.resume == "auto":
        p = Path(cfg.out_dir) / "checkpoints" / "last.pt"
        return str(p) if p.exists() else ""
    return cfg.resume


def _setup(cfg: TrainConfig, write_files: bool = True) -> SimpleNamespace:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    if cfg.grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    device = resolve_device(cfg.device)
    amp_dtype = resolve_amp(cfg.amp, device.type)
    out = Path(cfg.out_dir)
    if write_files:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    rpath = _resume_path(cfg)
    resume = torch.load(rpath, map_location="cpu", weights_only=False) if rpath else None
    dcfg = cfg.dataset_config()
    if resume:
        tr, va = resume["split"]["train"], resume["split"]["val"]
        stats = NormStats.from_dict(resume["norm_stats"])
    else:
        tr, va = split_episodes(expand_episodes(cfg.episodes), cfg.val_fraction, cfg.seed,
                                cfg.train_equals_val, expand_episodes(cfg.val_episodes))
        stats = None
    if cfg.cache_dir:
        from .prepare_cache import load_cached_stats, prepare_cache
        t0 = time.time()
        tr_dirs = prepare_cache(tr, cfg.cache_dir, dcfg.image_size, dcfg.cameras,
                                cfg.cache_workers)
        va_dirs = tr_dirs if va == tr else prepare_cache(va, cfg.cache_dir, dcfg.image_size,
                                                         dcfg.cameras, cfg.cache_workers)
        print(f"[train] cache ready in {time.time() - t0:.1f}s: {cfg.cache_dir}", flush=True)
        if stats is None:
            stats = load_cached_stats(tr_dirs)            # train split only
        train_ds = CachedEpisodeDataset(tr_dirs, dcfg, stats)
        val_ds = train_ds if va == tr else CachedEpisodeDataset(va_dirs, dcfg, stats)
    else:
        if stats is None:
            stats = compute_norm_stats(tr)                # train split only
        train_ds = EpisodeDataset(tr, dcfg, stats)
        val_ds = EpisodeDataset(va, dcfg, stats) if va != tr else train_ds
    if write_files:
        stats.save(out / "norm_stats.json")
        (out / "split.json").write_text(json.dumps({"train": tr, "val": va,
                                                    "train_equals_val": cfg.train_equals_val},
                                                   indent=2))
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    g = torch.Generator().manual_seed(cfg.seed)
    loader = _make_loader(train_ds, cfg, device, True, g,
                          drop_last=len(train_ds) > cfg.batch_size)
    # fixed val subset: the same frames every eval, so val loss is comparable (early stopping)
    n_val = min(len(val_ds), cfg.val_batches * cfg.batch_size)
    val_idx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(1))[:n_val]
    val_loader = _make_loader(torch.utils.data.Subset(val_ds, sorted(val_idx.tolist())), cfg,
                              device, False, None)

    J = len(train_ds.joint_names)
    raw_model = build_policy(cfg.policy, J, J, cfg.chunk_size, cameras=dcfg.cameras,
                             backbone=cfg.backbone, d_model=cfg.d_model).to(device)
    if cfg.channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    if resume:
        raw_model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        if scaler is not None and resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
    model = torch.compile(raw_model) if cfg.compile else raw_model
    meta = {"config": dataclasses.asdict(cfg), "state_dim": J, "action_dim": J,
            "joint_names": train_ds.joint_names, "cameras": list(dcfg.cameras),
            "norm_stats": stats.to_dict(), "split": {"train": tr, "val": va}}
    return SimpleNamespace(out=out, device=device, amp_dtype=amp_dtype, train_ds=train_ds,
                           val_ds=val_ds, loader=loader, val_loader=val_loader, model=model,
                           raw_model=raw_model, opt=opt, scaler=scaler, resume=resume, meta=meta,
                           tr=tr, va=va)


def _train_step(s, cfg, it) -> torch.Tensor:
    """One optimizer step over grad_accum micro-batches; returns the mean loss (on device)."""
    s.opt.zero_grad(set_to_none=True)
    tot = torch.zeros((), device=s.device)
    for _ in range(cfg.grad_accum):
        b = _to_device(next(it), s.device, cfg.channels_last)
        with _autocast(s.device, s.amp_dtype):
            loss = masked_loss(s.model(b), b["action"], b["action_is_pad"], cfg.loss)
        loss = loss.float() / cfg.grad_accum
        if s.scaler is not None:
            s.scaler.scale(loss).backward()
        else:
            loss.backward()
        tot += loss.detach()
    if s.scaler is not None:
        s.scaler.unscale_(s.opt)
    if cfg.grad_clip:
        torch.nn.utils.clip_grad_norm_(s.raw_model.parameters(), cfg.grad_clip)
    if s.scaler is not None:
        s.scaler.step(s.opt)
        s.scaler.update()
    else:
        s.opt.step()
    return tot


# ---------------------------------------------------------------- training

def train(cfg: TrainConfig, should_stop=None) -> dict:
    """Train; ``should_stop(step) -> bool`` (and SIGTERM/SIGINT) stop after saving last.pt."""
    s = _setup(cfg)
    out, model = s.out, s.model
    start = int(s.resume["step"]) if s.resume else 0
    eff = cfg.batch_size * cfg.grad_accum
    print(f"[train] {cfg.policy}/{cfg.backbone}: {count_parameters(s.raw_model)/1e6:.2f} M params, "
          f"{len(s.train_ds)} train / {len(s.val_ds)} val frames, steps {start}->{cfg.steps}, "
          f"batch {cfg.batch_size}x{cfg.grad_accum}, {s.device}, amp {s.amp_dtype}", flush=True)

    r = s.resume or {}
    best_val = r.get("best_val", float("inf"))
    bad_evals = int(r.get("bad_evals", 0))
    samples = int(r.get("samples_seen", 0))
    elapsed0 = float(r.get("elapsed_s", 0.0))
    if s.resume and "rng" in s.resume:
        _set_rng_state(s.resume["rng"])

    stop_signal = {"sig": None}

    def on_signal(signum, _frame):
        stop_signal["sig"] = signum

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, on_signal)

    t0 = time.time()

    def save(step, name):
        ckpt = {**s.meta, "step": step, "model": s.raw_model.state_dict(),
                "optimizer": s.opt.state_dict(),
                "scaler": s.scaler.state_dict() if s.scaler is not None else None,
                "rng": _rng_state(), "best_val": best_val, "bad_evals": bad_evals,
                "samples_seen": samples, "elapsed_s": elapsed0 + time.time() - t0}
        tmp = out / "checkpoints" / (name + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, out / "checkpoints" / name)

    csv_path = out / "loss.csv"
    new_csv = not (s.resume and csv_path.exists())
    fh = csv_path.open("w" if new_csv else "a", newline="")
    w = csv.writer(fh)
    if new_csv:
        w.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s", "samples_per_s"])
    model.train()
    it = _infinite(s.loader)
    first, last = None, float("nan")
    running, n_run = torch.zeros((), device=s.device), 0
    step = start
    last_save, n_time_ckpt = time.time(), 0
    win_t, win_samples = time.time(), 0
    stopped = early = False
    try:
        while step < cfg.steps:
            running += _train_step(s, cfg, it)
            n_run += 1
            step += 1
            samples += eff
            win_samples += eff
            if step % cfg.eval_every == 0 or step == cfg.steps:
                tl = (running / n_run).item()
                first = tl if first is None else first
                last = tl
                vl = _val_loss(model, s.val_loader, cfg.val_batches, cfg.loss, s.device,
                               s.amp_dtype, cfg.channels_last)
                sps = win_samples / max(time.time() - win_t, 1e-9)
                w.writerow([step, f"{tl:.6f}", f"{vl:.6f}", cfg.lr,
                            f"{elapsed0 + time.time() - t0:.1f}", f"{sps:.1f}"])
                fh.flush()
                print(f"[train] step {step:6d}  train {tl:.4f}  val {vl:.4f}  {sps:7.0f} samples/s"
                      f"  {time.time() - t0:.0f}s", flush=True)
                running, n_run = torch.zeros((), device=s.device), 0
                win_t, win_samples = time.time(), 0
                if vl < best_val - cfg.early_stop_min_delta:
                    best_val, bad_evals = vl, 0
                    save(step, "best.pt")
                else:
                    bad_evals += 1
                    if cfg.early_stop_patience and bad_evals >= cfg.early_stop_patience:
                        print(f"[train] early stop: no val improvement in {bad_evals} evals "
                              f"(best {best_val:.4f})", flush=True)
                        early = True
                        break
            if step % cfg.ckpt_every == 0:
                save(step, f"step_{step:06d}.pt")
            if cfg.ckpt_minutes and time.time() - last_save >= cfg.ckpt_minutes * 60:
                save(step, "last.pt")
                last_save, n_time_ckpt = time.time(), n_time_ckpt + 1
            if stop_signal["sig"] is not None or (should_stop and should_stop(step)):
                print(f"[train] stop requested at step {step}; saving last.pt", flush=True)
                stopped = True
                break
        save(step, "last.pt")
    finally:
        fh.close()
        for sig, h in old_handlers.items():
            signal.signal(sig, h)
    return {"start_step": start, "step": step, "first_train_loss": first,
            "final_train_loss": last, "out_dir": str(out), "samples_seen": samples,
            "time_checkpoints": n_time_ckpt, "stopped": stopped, "early_stopped": early,
            "best_val": best_val}


# ---------------------------------------------------------------- dry run

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dry_run(cfg: TrainConfig) -> dict:
    """Time the loader and a few train steps; estimate epoch and run time (and cost)."""
    t_setup = time.time()
    s = _setup(cfg, write_files=False)
    setup_s = time.time() - t_setup
    eff = cfg.batch_size * cfg.grad_accum
    n = max(1, cfg.bench_steps)

    it = _infinite(s.loader)
    next(it)                                   # worker start-up not counted
    t = time.time()
    for _ in range(n * cfg.grad_accum):
        next(it)
    loader_sps = n * eff / max(time.time() - t, 1e-9)

    if s.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(s.device)
    s.model.train()
    t = time.time()
    for _ in range(5 if cfg.compile else 2):   # warm-up (cuDNN autotune, torch.compile)
        _train_step(s, cfg, it)
    _sync(s.device)
    warm_s = time.time() - t
    t = time.time()
    for _ in range(n):
        loss = _train_step(s, cfg, it)
    loss.item()
    _sync(s.device)
    step_s = (time.time() - t) / n
    sps = eff / step_s
    frames = len(s.train_ds)
    rep = {
        "device": str(s.device), "amp": str(s.amp_dtype), "compile": cfg.compile,
        "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum, "num_workers": cfg.num_workers,
        "cache": bool(cfg.cache_dir), "train_frames": frames, "val_frames": len(s.val_ds),
        "params_m": count_parameters(s.raw_model) / 1e6, "setup_s": setup_s, "warmup_s": warm_s,
        "loader_samples_per_s": loader_sps, "samples_per_s": sps, "step_s": step_s,
        "est_epoch_s": frames / sps, "steps": cfg.steps, "est_total_s": step_s * cfg.steps,
        "est_epochs": cfg.steps * eff / max(frames, 1),
        "loader_bound": loader_sps < 1.2 * sps,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated(s.device) / 1e9
                            if s.device.type == "cuda" else None),
    }
    rep["est_cost"] = rep["est_total_s"] / 3600 * cfg.price_per_hour
    (s.out / "dry_run.json").write_text(json.dumps(rep, indent=2))
    print(f"[dry-run] {rep['device']} amp={rep['amp']} batch {cfg.batch_size}x{cfg.grad_accum}: "
          f"{sps:.0f} samples/s train, {loader_sps:.0f} samples/s loader"
          f"{' (LOADER-BOUND: add workers / use the cache)' if rep['loader_bound'] else ''}\n"
          f"[dry-run] {frames} train frames -> est epoch {rep['est_epoch_s']:.1f}s; "
          f"{cfg.steps} steps ({rep['est_epochs']:.1f} epochs) -> est total "
          f"{rep['est_total_s'] / 3600:.2f} h"
          + (f", est cost {rep['est_cost']:.2f} at {cfg.price_per_hour}/h"
             if cfg.price_per_hour else "")
          + (f"; peak GPU mem {rep['peak_gpu_mem_gb']:.1f} GB" if rep["peak_gpu_mem_gb"] else ""),
          flush=True)
    return rep


def main(argv=None):
    cfg = TrainConfig.from_args(argv)
    if not cfg.episodes and not _resume_path(cfg):
        raise SystemExit("--episodes is required")
    if cfg.dry_run:
        dry_run(cfg)
        return
    print(json.dumps(train(cfg), indent=2))


if __name__ == "__main__":
    main()
