"""Read an MCAP rosbag2 as (topic, receive_time_ns, msg) tuples in time order.

Backends, same interface:
  rosbag2  rosbag2_py + rclpy deserialisation (inside the ROS 2 container)
  mcap     pure-python `mcap` + `mcap-ros2-support` (host fallback; no ROS install needed)
  auto     rosbag2 if importable, else mcap
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator


def find_mcap_files(path) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == ".mcap":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.mcap"))
        if files:
            return files
    raise FileNotFoundError(f"no .mcap files at {p}")


def read_bag(path, topics: Iterable[str] | None = None,
             backend: str = "auto") -> Iterator[tuple[str, int, object]]:
    wanted = set(topics) if topics is not None else None
    if backend == "auto":
        try:
            import rosbag2_py  # noqa: F401
            backend = "rosbag2"
        except ImportError:
            backend = "mcap"
    if backend == "rosbag2":
        return _read_rosbag2(path, wanted)
    if backend == "mcap":
        return _read_mcap(path, wanted)
    raise ValueError(f"unknown backend {backend!r}")


def _read_rosbag2(path, wanted):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    p = Path(path)
    uri = str(p.parent if p.is_file() else p)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in types if wanted is None or t in wanted]
    if not present:
        return
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    classes = {t: get_message(types[t]) for t in present}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, int(t_ns), deserialize_message(data, classes[topic])


def _read_mcap(path, wanted):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    for f in find_mcap_files(path):
        with open(f, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            topics = sorted(wanted) if wanted is not None else None
            for _, channel, message, msg in reader.iter_decoded_messages(topics=topics,
                                                                          log_time_order=True):
                yield channel.topic, int(message.log_time), msg
