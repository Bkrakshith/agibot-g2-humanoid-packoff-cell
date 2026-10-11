# g2cell_teleop's waist auto-face uses g2cell_motion.waist_policy; make the sibling sources importable.
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "g2cell_teleop"))
sys.path.insert(0, str(_SRC / "g2cell_motion"))
