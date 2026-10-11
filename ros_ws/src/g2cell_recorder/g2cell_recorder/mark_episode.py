"""Set an episode's outcome after recording: the HDF5 root `success` attr (true|false|unknown).

    mark_episode /g2cell_output/episodes/<name>/<name>.hdf5 --success true
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUTCOMES = ("true", "false", "unknown")


def mark(path, success: str) -> None:
    if success not in OUTCOMES:
        raise ValueError(f"success must be one of {OUTCOMES}, got {success!r}")
    import h5py  # noqa: PLC0415 - keep --help working without h5py

    with h5py.File(path, "r+") as f:
        f.attrs["success"] = success


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mark_episode", description=__doc__.splitlines()[0])
    ap.add_argument("hdf5", help="episode HDF5 written by bag_to_hdf5")
    ap.add_argument("--success", choices=OUTCOMES, required=True)
    a = ap.parse_args(argv)
    if not Path(a.hdf5).is_file():
        print(f"mark_episode: no such file: {a.hdf5}", file=sys.stderr)
        return 1
    mark(a.hdf5, a.success)
    print(f"{a.hdf5}: success={a.success}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
