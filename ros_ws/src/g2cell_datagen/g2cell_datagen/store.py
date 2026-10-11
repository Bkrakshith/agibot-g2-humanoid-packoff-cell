"""Batch folder bookkeeping (pure): seeds, episode names, per-episode JSON, resume.

Layout (the recorder writes into the same folder, EPISODES_DIR=<batch_dir>):
    <batch_dir>/batch.json                    run arguments + config snapshot
    <batch_dir>/ep_s00007/episode.json        seed, randomisation, outcome, timings, status
    <batch_dir>/ep_s00007/bag/                MCAP (g2cell_recorder record_episode.sh)
    <batch_dir>/ep_s00007/ep_s00007.hdf5
    <batch_dir>/_crashed/ep_s00007.1/         leftovers of an interrupted attempt (kept, never reused)

status: running (written before anything moves; still there after a crash) -> done | error | discarded.
Resume: a seed is skipped only when its episode.json says done.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

EPISODE_JSON = "episode.json"
DONE = "done"
_NAME_RE = re.compile(r"^ep_s(\d{5,})$")


def parse_seeds(text: str) -> list:
    """'0-9,20,30-32' -> [0..9, 20, 30, 31, 32] (order kept, duplicates dropped)."""
    out, seen = [], set()
    if not text or not text.strip():
        raise ValueError("empty seed list")
    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"bad seed range {part!r} (use e.g. 0-9,20)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if b < a:
            raise ValueError(f"bad seed range {part!r}: end before start")
        for s in range(a, b + 1):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def episode_name(seed: int) -> str:
    return f"ep_s{int(seed):05d}"


def seed_of(name: str):
    m = _NAME_RE.match(name)
    return int(m.group(1)) if m else None


class EpisodeStore:
    def __init__(self, batch_dir):
        self.dir = Path(batch_dir)

    def episode_dir(self, name: str) -> Path:
        return self.dir / name

    def json_path(self, name: str) -> Path:
        return self.episode_dir(name) / EPISODE_JSON

    def hdf5_path(self, name: str) -> Path:
        return self.episode_dir(name) / f"{name}.hdf5"

    def load(self, name: str):
        p = self.json_path(name)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {"status": "corrupt"}

    def status(self, name: str):
        rec = self.load(name)
        return None if rec is None else rec.get("status")

    def write(self, name: str, rec: dict) -> Path:
        p = self.json_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=False))
        os.replace(tmp, p)
        return p

    def pending(self, seeds, retry_errors: bool = True) -> list:
        out = []
        for s in seeds:
            st = self.status(episode_name(s))
            if st == DONE or (st == "error" and not retry_errors):
                continue
            out.append(s)
        return out

    def quarantine(self, name: str):
        """Move an unfinished attempt aside (the recorder refuses to overwrite an existing bag)."""
        src = self.episode_dir(name)
        if not src.exists():
            return None
        crashed = self.dir / "_crashed"
        crashed.mkdir(parents=True, exist_ok=True)
        n = 1
        while (crashed / f"{name}.{n}").exists():
            n += 1
        dest = crashed / f"{name}.{n}"
        shutil.move(str(src), str(dest))
        return dest

    def records(self) -> list:
        out = []
        if not self.dir.is_dir():
            return out
        for d in sorted(self.dir.iterdir()):
            if d.is_dir() and seed_of(d.name) is not None:
                rec = self.load(d.name)
                if rec is not None:
                    out.append({"name": d.name, **rec})
        return out

    def summary(self) -> dict:
        recs = self.records()
        return {
            "episodes": len(recs),
            "status": dict(sorted(Counter(r.get("status", "?") for r in recs).items())),
            "outcome": dict(sorted(Counter(r["outcome"] for r in recs if r.get("outcome")).items())),
            "success": dict(sorted(Counter(r["success"] for r in recs if r.get("success")).items())),
        }
