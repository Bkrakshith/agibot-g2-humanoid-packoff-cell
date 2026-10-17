import os
import pathlib
import sys

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "test"))

# The cloned training repo (Apache-2.0, commit 29e4dc3). Tests that need it are skipped without it.
# Set GDK_TRAINING_DIR and G2_URDF to your checkouts; the defaults assume they sit next to this repository.
_WORK = PKG.parents[3]  # the work folder that holds this repository
TRAINING = pathlib.Path(os.environ.get("GDK_TRAINING_DIR", _WORK / "agibot-gdk-training"))
URDF = pathlib.Path(os.environ.get("G2_URDF", _WORK / "g2_viz_ws" / "g2_crs_omnipicker.urdf"))
SCENE = PKG.parents[1] / "config" / "scenes" / "scene_g2cell.yaml"


@pytest.fixture
def training_dir():
    if not (TRAINING / "fake" / "agibot_gdk" / "__init__.py").exists():
        pytest.skip(f"training repo not found at {TRAINING}")
    return TRAINING


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    """Keep the persisted control mode of every test in its own folder."""
    monkeypatch.setenv("GDK_SIM_STATE_DIR", str(tmp_path / "state"))
