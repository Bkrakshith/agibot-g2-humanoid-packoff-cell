"""Recorder side of a backend (ROS-free): g2cell_teleop's RecorderController around
`ros2 run g2cell_recorder record_episode.sh <name> --scene S --seed N` (compact defaults: JPEG head RGB,
compressedDepth at 5 Hz, MCAP zstd_fast), stopped with SIGINT, converted to HDF5 by the script, then
marked with mark_episode when the outcome is true/false.
"""
from __future__ import annotations

import time
from pathlib import Path

from g2cell_datagen.runner import RecordingResult
from g2cell_teleop.recorder_control import RecorderController

NO_H5PY_RC = 3     # record_episode.sh: bag written, h5py missing -> convert later (datagen_batch finalize)


def judge(rc, bag_dir: Path, hdf5: Path, allow_deferred: bool = True) -> RecordingResult:
    has_bag = (Path(bag_dir) / "metadata.yaml").is_file()
    if rc == 0 and Path(hdf5).is_file():
        return RecordingResult(True, hdf5=str(hdf5), returncode=rc)
    if rc == NO_H5PY_RC and has_bag and allow_deferred:
        return RecordingResult(True, hdf5="", returncode=rc,
                               error="bag only (no h5py): run datagen_batch finalize to convert + mark")
    if rc is None:
        return RecordingResult(False, returncode=None, error="recorder still running / converting (timeout)")
    return RecordingResult(False, returncode=rc,
                           error=f"recorder exited {rc}; bag {'present' if has_bag else 'missing'}, no hdf5")


class EpisodeRecorder:
    def __init__(self, episodes_dir, scene: str = "", record_cmd=None, mark_cmd=None,
                 warmup_s: float = 1.5, convert_timeout_s: float = 300.0, allow_deferred: bool = True, log=print):
        self.episodes_dir = Path(episodes_dir)
        self.scene, self.warmup_s = scene, warmup_s
        self.convert_timeout_s, self.allow_deferred, self.log = convert_timeout_s, allow_deferred, log
        self._kw = {}
        if record_cmd:
            self._kw["record_cmd"] = record_cmd
        if mark_cmd:
            self._kw["mark_cmd"] = mark_cmd
        self.rc: RecorderController | None = None

    def start(self, name: str, seed: int) -> None:
        self.rc = RecorderController(scene=self.scene, seed=seed, episodes_dir=str(self.episodes_dir),
                                     name_fn=lambda: name, log=self.log, **self._kw)
        self.rc.start()
        time.sleep(self.warmup_s)          # let ros2 bag discover and subscribe before anything moves
        if not self.rc.running:
            raise RuntimeError(f"recorder for {name} exited at start (rc {self.rc._proc.returncode})")

    def stop(self, name: str, success: str) -> RecordingResult:
        rc = self.rc
        if rc is None:
            return RecordingResult(False, error="recorder was never started")
        rc.stop(outcome=success)
        done = rc.wait(self.convert_timeout_s)
        d = self.episodes_dir / name
        res = judge(rc.last_returncode if done else None, d / "bag", d / f"{name}.hdf5", self.allow_deferred)
        if res.ok and rc.last_error:
            res.error = (res.error + "; " if res.error else "") + rc.last_error
        return res
