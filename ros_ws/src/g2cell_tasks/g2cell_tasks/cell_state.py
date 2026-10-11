"""Cell bookkeeping behind cell_manager (no ROS imports).

What is real now: episode start/stop/marks, pick counts from /g2cell/pick_results.
What is not: the conveyor (Stage 3) and bag spawning/scoring (Stage 5) -- their commands
answer NOT_IMPLEMENTED, and the status lists them in not_implemented.
"""
from __future__ import annotations

import time

OK, NOT_IMPLEMENTED, REJECTED = 0, 1, 2
START, STOP, MARK_SUCCESS, MARK_FAILURE = 0, 1, 2, 3
LINE_COMMANDS = {0: "run", 1: "stop", 2: "release crate"}
LEFT_TABLE = {"success", "wrong_target", "drop", "regrasp", "overflow", "unverified"}
IN_CRATE = {"success", "regrasp", "overflow"}


class CellState:
    def __init__(self, crate_fill_target: int = 10, bags_on_table: int = 0, crate_at_pack_position: bool = True):
        self.episode_id = ""
        self.episode_active = False
        self.mark: str | None = None
        self.line_running = False
        self.crate_at_pack_position = crate_at_pack_position
        self.crate_fill_count = 0
        self.crate_fill_target = crate_fill_target
        self.bags_on_table = bags_on_table
        self.picks_done = 0

    @property
    def crate_full(self) -> bool:
        return self.crate_fill_count >= self.crate_fill_target

    def not_implemented(self) -> list[str]:
        return ["line_running/crate_at_pack_position: assumed (Stage 3 conveyor indexing not implemented)",
                "bags_on_table: counted from pick results, not sensed (Stage 5 spawning not implemented)",
                "crate_fill_count: only verified placements count (Stage 5 scoring not implemented)"]

    # TODO(Stage 3): drive the conveyor surface velocity and the (kinematic) stop gate, watch the
    # photo-eye, index the next crate in, and update line_running / crate_at_pack_position.
    def line_command(self, command: int):
        what = LINE_COMMANDS.get(command)
        if what is None:
            return REJECTED, f"unknown line command {command}"
        return NOT_IMPLEMENTED, f"line '{what}' not implemented: conveyor indexing is Stage 3"

    # TODO(Stage 5): drop `count` bags (variant/label randomised with `seed`) above the table,
    # publish their BagInfo, and feed them to pick_bag_server's catalog.
    def spawn_bags(self, count: int, seed: int):
        return NOT_IMPLEMENTED, f"spawning {count} bag(s) not implemented: bag spawning is Stage 5"

    def episode(self, command: int, episode_id: str = "", now: str | None = None):
        """-> (status, message, episode_id)."""
        if command == START:
            if self.episode_active:
                return REJECTED, f"episode {self.episode_id} is already running", self.episode_id
            self.episode_id = episode_id or f"ep_{now or time.strftime('%Y%m%d_%H%M%S')}"
            self.episode_active, self.mark = True, None
            self.crate_fill_count = self.picks_done = 0
            return OK, f"episode {self.episode_id} started", self.episode_id
        if command == STOP:
            if not self.episode_active:
                return REJECTED, "no episode running", self.episode_id
            self.episode_active = False
            return OK, f"episode {self.episode_id} stopped", self.episode_id
        if command in (MARK_SUCCESS, MARK_FAILURE):
            if not self.episode_id:
                return REJECTED, "no episode to mark", ""
            self.mark = "success" if command == MARK_SUCCESS else "failure"
            return OK, f"episode {self.episode_id} marked {self.mark}", self.episode_id
        return REJECTED, f"unknown episode command {command}", self.episode_id

    def on_pick_result(self, outcome: str, verified: bool, intended_target: str) -> None:
        self.picks_done += 1
        if outcome in LEFT_TABLE and self.bags_on_table > 0:
            self.bags_on_table -= 1
        if verified and outcome in IN_CRATE and intended_target == "crate":
            self.crate_fill_count += 1
