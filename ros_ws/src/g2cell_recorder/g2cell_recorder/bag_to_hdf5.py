"""bag_to_hdf5: convert one MCAP rosbag2 episode into one HDF5 file (schema in episode.py).

    ros2 run g2cell_recorder bag_to_hdf5 /g2cell_output/episodes/ep1/bag \
        -o /g2cell_output/episodes/ep1/ep1.hdf5 --scene scene_pnp_g2_op --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from g2cell_recorder.bag_reader import read_bag
from g2cell_recorder.episode import (
    COMPRESSIONS,
    DEFAULT_COMPRESSION,
    DEFAULT_GZIP_LEVEL,
    DEFAULT_TOPICS,
    convert_messages,
)


def default_workers() -> int:
    """One worker per core, capped at 8 (the reader/writer process needs a core too)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(8, n - 1 if n > 2 else n))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="bag_to_hdf5", description=__doc__.splitlines()[0])
    ap.add_argument("bag", help="rosbag2 directory (or a single .mcap file)")
    ap.add_argument("-o", "--output", help="HDF5 path (default: <bag>.hdf5)")
    ap.add_argument("--rate", type=float, default=30.0, help="aligned clock rate in Hz (30)")
    ap.add_argument("--scene", default="", help="scene name stored in the metadata")
    ap.add_argument("--seed", type=int, default=-1, help="episode seed stored in the metadata")
    ap.add_argument("--success", choices=["true", "false", "unknown"], default="unknown")
    ap.add_argument("--aligned-images", action="store_true",
                    help="also copy frames into /aligned (default: index only)")
    ap.add_argument("--backend", choices=["auto", "rosbag2", "mcap"], default="auto")
    ap.add_argument("--compression", choices=COMPRESSIONS, default=DEFAULT_COMPRESSION,
                    help="image/depth datasets: gzip (smallest, parallel), lzf (fast, single "
                         "thread) or none")
    ap.add_argument("--level", type=int, default=DEFAULT_GZIP_LEVEL,
                    help="gzip level 0-9 (default 4; 1 is ~20%% faster and ~15%% larger)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="frame decode/compress processes (0 = in-process)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    bag = Path(a.bag).resolve()
    out = Path(a.output) if a.output else bag.with_name(bag.name + ".hdf5")
    t0 = time.monotonic()
    summary = convert_messages(
        read_bag(bag, topics=DEFAULT_TOPICS.keys(), backend=a.backend), out, source=str(bag),
        scene=a.scene, seed=a.seed, aligned_rate_hz=a.rate, aligned_images=a.aligned_images,
        success=a.success, compression=a.compression, compression_level=a.level,
        workers=a.workers)
    print(f"wrote {summary['path']} in {time.monotonic() - t0:.1f}s, "
          f"{(summary['end_time_ns'] - summary['start_time_ns']) / 1e9:.2f}s of data, "
          f"aligned {summary['aligned_len']} steps @ {a.rate:g} Hz, "
          f"{out.stat().st_size / 1e6:.1f} MB ({a.compression}, {a.workers} workers)")
    for k, n in sorted(summary["counts"].items()):
        tr = summary["transports"].get(k, "")
        print(f"  {k:28s} {n:6d} msgs  {summary['rates_hz'][k]:6.1f} Hz  {tr}")
    for t, n in summary["ignored_topics"].items():
        print(f"  ignored {t}: {n} msgs (stream already fed by another transport)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
