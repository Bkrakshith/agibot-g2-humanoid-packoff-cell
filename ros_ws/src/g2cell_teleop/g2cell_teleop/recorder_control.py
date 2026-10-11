"""Start/stop g2cell_recorder episodes from the teleop node (ROS-free, unit-tested).

start() launches `ros2 run g2cell_recorder record_episode.sh <name> --scene S --seed N` in its
own process group. stop() sends SIGINT to that group, exactly like Ctrl+C in a terminal, so
`ros2 bag record` closes the bag and the script converts it to HDF5. A watcher thread waits
for the conversion; when it finishes, a success/failure mark (made during or after the
recording) is written with `ros2 run g2cell_recorder mark_episode <h5> --success X`.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

OUTCOMES = ("true", "false", "unknown")


def default_episode_name(scene: str = "", now: time.struct_time | None = None) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    scene = re.sub(r"[^A-Za-z0-9_-]+", "_", scene).strip("_")
    return f"teleop_{scene}_{stamp}" if scene else f"teleop_{stamp}"


class RecorderController:
    STOP_SIGNAL = signal.SIGINT

    def __init__(self, scene: str = "", seed: int = -1,
                 episodes_dir: str = "/g2cell_output/episodes",
                 record_cmd=("ros2", "run", "g2cell_recorder", "record_episode.sh"),
                 mark_cmd=("ros2", "run", "g2cell_recorder", "mark_episode"),
                 name_fn: Callable[[], str] | None = None,
                 log: Callable[[str], None] = print):
        self.scene, self.seed = scene, int(seed)
        self.episodes_dir = episodes_dir
        self.record_cmd, self.mark_cmd = list(record_cmd), list(mark_cmd)
        self.name_fn = name_fn or (lambda: default_episode_name(self.scene))
        self.log = log
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._watcher: threading.Thread | None = None
        self._name: str | None = None
        self._outcome = "unknown"
        self._converted = False  # last episode's HDF5 exists and is final
        self.last_returncode: int | None = None
        self.last_error = ""

    # ---- state -------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def busy(self) -> bool:
        """Recording, or still converting the last episode."""
        return self._watcher is not None and self._watcher.is_alive() or self.running

    def hdf5_path(self, name: str | None = None) -> Path:
        name = name or self._name
        return Path(self.episodes_dir) / name / f"{name}.hdf5"

    # ---- control -----------------------------------------------------------------------
    def start(self) -> str:
        if self.busy:
            raise RuntimeError("an episode is still recording or converting")
        name = self.name_fn()
        cmd = self.record_cmd + [name, "--scene", self.scene, "--seed", str(self.seed)]
        env = dict(os.environ, EPISODES_DIR=self.episodes_dir)
        with self._lock:
            self._name, self._outcome, self._converted = name, "unknown", False
            self.last_returncode, self.last_error = None, ""
            self._proc = subprocess.Popen(cmd, env=env, start_new_session=True,
                                          stdin=subprocess.DEVNULL)
        self.log(f"recording episode {name} -> {self.hdf5_path()}")
        return name

    def stop(self, outcome: str | None = None) -> bool:
        """SIGINT the recorder; conversion and marking continue in the background."""
        if not self.running:
            return False
        if outcome is not None:
            self._set_outcome(outcome)
        try:
            os.killpg(self._proc.pid, self.STOP_SIGNAL)
        except ProcessLookupError:
            pass
        self._watcher = threading.Thread(target=self._finish, args=(self._proc, self._name), daemon=True)
        self._watcher.start()
        self.log(f"stopping {self._name}; converting to HDF5 in the background")
        return True

    def mark(self, outcome: str) -> None:
        """Mark the current (or, after stop, the last) episode's outcome."""
        self._set_outcome(outcome)
        with self._lock:
            ready = self._converted
        if ready:
            self._write_mark(self._name, outcome)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the last episode is converted and marked. False on timeout."""
        if self._watcher is not None:
            self._watcher.join(timeout)
            return not self._watcher.is_alive()
        return not self.running

    # ---- internals ---------------------------------------------------------------------
    def _set_outcome(self, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._lock:
            self._outcome = outcome

    def _finish(self, proc: subprocess.Popen, name: str) -> None:
        rc = proc.wait()
        h5 = self.hdf5_path(name)
        with self._lock:
            self.last_returncode = rc
            ok = rc == 0 and h5.exists()
            self._converted = ok
            outcome = self._outcome
            if not ok:
                self.last_error = f"recorder exited {rc}; no hdf5 at {h5}"
        if not ok:
            self.log(f"episode {name}: {self.last_error}")
            return
        self.log(f"episode {name} converted: {h5}")
        if outcome != "unknown":
            self._write_mark(name, outcome)

    def _write_mark(self, name: str, outcome: str) -> None:
        h5 = self.hdf5_path(name)
        res = subprocess.run(self.mark_cmd + [str(h5), "--success", outcome],
                             stdin=subprocess.DEVNULL, check=False)
        if res.returncode == 0:
            self.log(f"episode {name}: success={outcome}")
        else:
            self.last_error = f"mark_episode exited {res.returncode} for {h5}"
            self.log(self.last_error)
