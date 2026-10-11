#!/usr/bin/env python3
"""Download the GenieSimAssets subset the G2 cell needs (~165 MB of the 29 GB dataset).

    python scripts/fetch_assets.py                      # -> ../geniesim_assets
    python scripts/fetch_assets.py --dest /data/geniesim_assets
    python scripts/fetch_assets.py --dry-run            # list matched files and total size only

Needs huggingface_hub (pip install huggingface_hub). Safe to rerun: files already present
are skipped by snapshot_download.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "agibot-world/GenieSimAssets"
REPO_TYPE = "dataset"
# The one place the subset is defined. Patterns use fnmatch, as in snapshot_download
# ("*" also matches "/", so "*.toml" picks up nested package files).
ALLOW_PATTERNS = [
    "*.toml",
    "*.py",
    "*.md",
    "VERSION",
    ".gitignore",
    "scenes/GDK/**",
    "scenes/blank/**",
    "robot/G2_omnipicker/**",
]

# Dataset commit the workstation copy was downloaded at (VERSION 3.2.0). Override with --revision.
ASSETS_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"

DEFAULT_DEST = Path(__file__).resolve().parent.parent.parent / "geniesim_assets"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dry_run(revision: str, verbose: bool) -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    siblings = info.siblings or []
    matched = list(
        filter_repo_objects(siblings, allow_patterns=ALLOW_PATTERNS, key=lambda s: s.rfilename)
    )
    total = sum(s.size or 0 for s in matched)
    repo_total = sum(s.size or 0 for s in siblings)
    if verbose:
        for s in sorted(matched, key=lambda s: s.rfilename):
            print(f"{human(s.size or 0):>10}  {s.rfilename}")
    print(f"repo {REPO_ID} @ {info.sha}")
    print(f"matched {len(matched)} of {len(siblings)} files: {human(total)} of {human(repo_total)}")
    for pat in ALLOW_PATTERNS:
        sub = list(filter_repo_objects(matched, allow_patterns=[pat], key=lambda s: s.rfilename))
        print(f"  {pat:<26} {len(sub):>5} files  {human(sum(s.size or 0 for s in sub)):>10}")
    return 0


def download(dest: Path, revision: str) -> int:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(dest),
    )
    if not (Path(path) / "pyproject.toml").is_file():
        print(f"warning: no pyproject.toml in {path}; cell.sh up will refuse it", file=sys.stderr)
        return 1
    print(f"assets ready in {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path(os.environ.get("ASSETS_DIR", DEFAULT_DEST)),
                    help=f"download directory (default: $ASSETS_DIR or {DEFAULT_DEST})")
    ap.add_argument("--revision", default=os.environ.get("GENIESIM_ASSETS_REV", ASSETS_REVISION),
                    help="dataset branch/commit (default: $GENIESIM_ASSETS_REV or the pinned commit)")
    ap.add_argument("--dry-run", action="store_true", help="list matched files and size; download nothing")
    ap.add_argument("-v", "--verbose", action="store_true", help="with --dry-run, print every file")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run(args.revision, args.verbose)
    return download(args.dest.resolve(), args.revision)


if __name__ == "__main__":
    sys.exit(main())
