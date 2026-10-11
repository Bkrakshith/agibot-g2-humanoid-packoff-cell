"""Extracts the public surface of an agibot_gdk module: names, call signatures, enum members.

`from_source` reads the training repo's fake/agibot_gdk/__init__.py with ast, so the fake's own
imports (cv2, gdk_training) are never executed. `check_module` compares a live module to it.

    python3 test/surface.py <fake __init__.py> > test/fake_surface_3.3.8.json   # refresh the snapshot
"""
from __future__ import annotations

import ast
import enum
import inspect
import json
import sys

# Helpers of the fake that are not part of the GDK API.
NOT_API = {"World"}


def _params(fn: ast.FunctionDef) -> list[dict]:
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = [{"name": a.arg, "kind": "positional", "default": d is not None} for a, d in zip(positional, defaults)]
    out += [{"name": a.arg, "kind": "keyword_only", "default": d is not None}
            for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return [p for p in out if p["name"] != "self"]


def _attributes(fn: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(fn):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            for t in (target.elts if isinstance(target, ast.Tuple) else [target]):
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    names.append(t.attr)
    return sorted(set(names))


def from_source(text: str) -> dict:
    tree = ast.parse(text)
    surface = {"functions": {}, "classes": {}, "enums": {}}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            surface["functions"][node.name] = _params(node)
        elif isinstance(node, ast.ClassDef) and node.name not in NOT_API and not node.name.startswith("_"):
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            if bases & {"Enum", "IntEnum"}:
                surface["enums"][node.name] = {
                    t.targets[0].id: ast.literal_eval(t.value) for t in node.body if isinstance(t, ast.Assign)}
                continue
            methods, attributes = {}, []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (item.name == "__init__" or not item.name.startswith("_")):
                    methods[item.name] = _params(item)
                    if item.name == "__init__":
                        attributes = _attributes(item)
            surface["classes"][node.name] = {"methods": methods, "attributes": attributes}
    return surface


def _live_params(fn) -> list[dict]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        kind = "keyword_only" if p.kind == p.KEYWORD_ONLY else "positional"
        out.append({"name": p.name, "kind": kind, "default": p.default is not p.empty})
    return out


def check_module(module, surface: dict) -> list[str]:
    """Returns one line per difference between `module` and `surface` (empty: compatible)."""
    problems = []
    for name, params in surface["functions"].items():
        fn = getattr(module, name, None)
        if not callable(fn):
            problems.append(f"missing function {name}")
        elif _live_params(fn) != params:
            problems.append(f"{name}{_live_params(fn)} != {params}")
    for name, members in surface["enums"].items():
        cls = getattr(module, name, None)
        if not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
            problems.append(f"missing enum {name}")
            continue
        for member, value in members.items():
            if member not in cls.__members__:
                problems.append(f"missing {name}.{member}")
            elif cls[member].value != value:
                problems.append(f"{name}.{member} = {cls[member].value}, expected {value}")
    for name, spec in surface["classes"].items():
        cls = getattr(module, name, None)
        if not isinstance(cls, type):
            problems.append(f"missing class {name}")
            continue
        for method, params in spec["methods"].items():
            fn = getattr(cls, method, None)
            if fn is None:
                problems.append(f"missing {name}.{method}")
            elif _live_params(fn) != params:
                problems.append(f"{name}.{method}{_live_params(fn)} != {params}")
        init = spec["methods"].get("__init__", [])
        if spec["attributes"] and all(p["default"] for p in init):
            instance = cls()
            problems += [f"{name}() has no attribute {a}" for a in spec["attributes"] if not hasattr(instance, a)]
    return problems


if __name__ == "__main__":
    print(json.dumps(from_source(open(sys.argv[1]).read()), indent=1, sort_keys=True))
