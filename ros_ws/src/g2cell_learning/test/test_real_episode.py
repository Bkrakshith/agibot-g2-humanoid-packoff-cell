"""Integration test on the real Stage 1 teleop episode; skipped when the file is absent."""
from pathlib import Path

import pytest
import torch

REAL = (Path(__file__).resolve().parents[4] / "output/episodes/stage1_teleop/stage1_teleop.hdf5")
pytestmark = pytest.mark.skipif(not REAL.exists(), reason=f"{REAL} not present")


def test_real_episode_dataset_and_policy_step():
    from g2cell_learning.dataset import DatasetConfig, EpisodeDataset
    from g2cell_learning.policy import build_policy, masked_loss

    ds = EpisodeDataset([REAL], DatasetConfig(image_size=(64, 80), chunk_size=10,
                                              extra_cameras=("free_rgb",)))
    assert len(ds.joint_names) == 16
    assert ds.joint_names[7] == "idx31_gripper_l_inner_joint1"
    assert len(ds) > 500
    batch = torch.utils.data.default_collate([ds[i] for i in (0, 100, len(ds) - 1)])
    assert batch["observation.images.head_front_rgb"].shape == (3, 3, 64, 80)
    assert batch["action_is_pad"][2].all()
    pol = build_policy("act", 16, 16, 10, cameras=ds.cfg.cameras, d_model=32)
    loss = masked_loss(pol(batch), batch["action"], batch["action_is_pad"])
    assert torch.isfinite(loss)
