# Make the sibling source packages importable for host tests (no colcon install needed).
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2]
for pkg in ("g2cell_datagen", "g2cell_tasks", "g2cell_motion", "g2cell_recorder", "g2cell_teleop",
            "g2cell_learning"):
    sys.path.insert(0, str(_SRC / pkg))
