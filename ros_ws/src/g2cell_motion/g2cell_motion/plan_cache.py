"""Persistent cache of solved motion plans, invalidated automatically when the cell layout changes
(no ROS imports, unit-tested).

    key = layout_key(build_py=".../assets/cell/build.py", colliders=".../cell_colliders.json")
    cache = PlanCache("/g2cell_output/plan_cache/packing_demo.json", key)
    hit = cache.get({"arm": "left", "targets": ..., "base": ...})
    cache.put({...}, {"path": [...]}); cache.save()

Layout key = sha256 over (a) the module-level UPPERCASE constants of assets/cell/build.py, read via the
AST so comments, formatting and function bodies do not matter, and (b) the content of the exported
cell_colliders.json (what the planner actually checks against; present in the container, where build.py
is not mounted). Whatever exists is hashed. A cache file written under another key (or planner version)
is discarded on load.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_DIGITS = 6


def _canon(obj, digits: int):
    if isinstance(obj, float):
        v = round(obj, digits)
        return 0.0 if v == 0 else v          # -0.0 == 0.0
    if isinstance(obj, (list, tuple)):
        return [_canon(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canon(v, digits) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    return obj


def cache_key(obj, digits: int = DEFAULT_DIGITS) -> str:
    """sha256 of the canonical JSON of obj (dicts sorted, tuples = lists, floats rounded)."""
    s = json.dumps(_canon(obj, digits), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def layout_constants_hash(build_py) -> str | None:
    """Hash of the module-level UPPERCASE assignments of a build.py (None if missing/unparsable)."""
    try:
        tree = ast.parse(Path(build_py).read_text())
    except (OSError, SyntaxError, ValueError):
        return None
    parts = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if names and all(n.isupper() for n in names):
            parts.append(ast.dump(node, annotate_fields=False, include_attributes=False))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def file_hash(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def layout_key(build_py=None, colliders=None, extra: str = "") -> str:
    parts = {"build": layout_constants_hash(build_py) if build_py else None,
             "colliders": file_hash(colliders) if colliders else None, "extra": extra}
    return cache_key(parts)


def find_build_py(start=None) -> Path | None:
    """assets/cell/build.py of the g2_packoff_cell checkout above `start` (source tree / host), or the
    G2CELL_BUILD_PY environment variable. None inside the container (not mounted)."""
    env = os.environ.get("G2CELL_BUILD_PY")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(start or __file__).resolve()
    for p in here.parents:
        c = p / "assets" / "cell" / "build.py"
        if c.is_file():
            return c
    return None


class PlanCache:
    def __init__(self, path, layout: str, version: int = 1):
        self.path = Path(path) if path else None
        self.layout, self.version = layout, version
        self.entries: dict = {}
        self.invalidated = False
        self.hits = self.misses = 0
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.invalidated = True
            return
        if data.get("layout") == layout and data.get("version") == version:
            self.entries = dict(data.get("entries", {}))
        else:
            self.invalidated = True

    def get(self, key_obj):
        v = self.entries.get(cache_key(key_obj))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key_obj, value) -> None:
        self.entries[cache_key(key_obj)] = value

    def clear(self) -> None:
        self.entries = {}

    def save(self) -> bool:
        """Atomic write; False (and keep going in memory) if the file cannot be written."""
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".plan_cache")
            with os.fdopen(fd, "w") as f:
                json.dump({"layout": self.layout, "version": self.version, "entries": self.entries}, f)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False


def goal_key(arm: str, goal: dict, current: dict, params: dict, state_digits: int = 4) -> str:
    """Cache key for one MoveArm goal: the goal fields, the start state (rounded to 1e-4 rad, so
    /joint_states noise does not miss) and the planner parameters."""
    cur = {k: round(float(v), state_digits) for k, v in sorted(current.items())}
    return cache_key({"arm": arm, "goal": goal, "current": cur, "params": params})
