"""config/tasks.yaml -> typed task configuration (no ROS imports)."""
from __future__ import annotations

from dataclasses import dataclass

from g2cell_tasks.bags import BagCatalog
from g2cell_tasks.reachability import ReachTable
from g2cell_tasks.sorting import SortRules
from g2cell_tasks.waypoints import CellTargets


@dataclass
class TaskConfig:
    bags: BagCatalog
    sort: SortRules
    targets: CellTargets
    reach: ReachTable
    max_grasp_attempts: int = 2


def load_task_config(doc: dict) -> TaskConfig:
    targets = CellTargets.from_dict(doc["targets"])
    s = doc["sort_rules"]
    return TaskConfig(
        bags=BagCatalog.from_list(doc.get("bags", [])),
        sort=SortRules(s.get("lanes", {}), s["default_lane"], list(targets.lanes), s.get("good_label", "good")),
        targets=targets,
        reach=ReachTable.from_dict(doc.get("reachability", {})),
        max_grasp_attempts=int(doc.get("max_grasp_attempts", 2)),
    )
