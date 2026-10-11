# g2cell_tasks builds on g2cell_motion; make the sibling source packages importable for host tests.
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "g2cell_tasks"))
sys.path.insert(0, str(_SRC / "g2cell_motion"))
