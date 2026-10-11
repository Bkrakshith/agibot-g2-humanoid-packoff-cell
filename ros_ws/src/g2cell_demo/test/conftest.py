# g2cell_demo builds on g2cell_motion; make the sibling source package importable for host tests.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "g2cell_motion"))
