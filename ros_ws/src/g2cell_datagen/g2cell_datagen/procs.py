"""Managed child processes (ROS-free): each in its own process group, stopped like Ctrl+C.

Used for the per-episode pick_bag_server (it loads the episode's tasks.yaml), the motion_server and
the sim restart commands. Never uses pkill: only the PIDs this process started are signalled.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time


class ManagedProcess:
    def __init__(self, cmd, name: str = "", log_path=None, env=None, log=print):
        self.cmd, self.name = list(cmd), name or os.path.basename(str(cmd[0]))
        self.log_path, self.env, self.log = log_path, env, log
        self.proc: subprocess.Popen | None = None
        self._fh = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> "ManagedProcess":
        if self.running:
            return self
        out = subprocess.DEVNULL
        if self.log_path:
            self._fh = open(self.log_path, "ab")
            out = self._fh
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=self.env)
        self.log(f"started {self.name} (pid {self.proc.pid})")
        return self

    def stop(self, timeout: float = 10.0) -> int | None:
        """SIGINT the group, then SIGTERM, then SIGKILL. Returns the exit code (None if never started)."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        rc = self.proc.poll()
        if self._fh:
            self._fh.close()
            self._fh = None
        self.log(f"stopped {self.name} (rc {rc})")
        return rc


def run_shell(cmd: str, timeout: float, log_path=None) -> int:
    """Run a shell command to completion in its own process group; kill the group on timeout."""
    p = ManagedProcess(["bash", "-c", cmd], name=cmd.split()[0] if cmd.split() else "bash", log_path=log_path,
                       log=lambda *_: None).start()
    deadline = time.monotonic() + timeout
    while p.running and time.monotonic() < deadline:
        time.sleep(0.2)
    if p.running:
        p.stop(timeout=5.0)
        raise TimeoutError(f"command timed out after {timeout:.0f} s: {cmd}")
    return p.proc.returncode
