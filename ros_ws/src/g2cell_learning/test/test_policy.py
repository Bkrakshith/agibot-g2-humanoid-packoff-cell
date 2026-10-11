import pytest
import torch

from g2cell_learning.policy import ACTLite, BCMLP, build_policy, masked_loss

B, K, J = 2, 5, 6


def _batch(cams=("head_front_rgb",), hw=(32, 40)):
    b = {"observation.state": torch.randn(B, J),
         "action": torch.randn(B, K, J),
         "action_is_pad": torch.zeros(B, K, dtype=torch.bool)}
    for c in cams:
        b[f"observation.images.{c}"] = torch.rand(B, 3, *hw)
    return b


@pytest.mark.parametrize("kind", ["act", "bc"])
@pytest.mark.parametrize("backbone", ["small", "resnet18"])
def test_forward_shape(kind, backbone):
    pol = build_policy(kind, state_dim=J, action_dim=J, chunk_size=K,
                       cameras=("head_front_rgb", "free_rgb"), backbone=backbone, d_model=32)
    out = pol(_batch(("head_front_rgb", "free_rgb")))
    assert out.shape == (B, K, J)


def test_build_policy_types():
    assert isinstance(build_policy("act", J, J, K, d_model=32), ACTLite)
    assert isinstance(build_policy("bc", J, J, K, d_model=32), BCMLP)
    with pytest.raises(ValueError):
        build_policy("diffusion", J, J, K)


def test_masked_loss_ignores_padding():
    pred = torch.zeros(1, 3, 2)
    tgt = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [100.0, 100.0]]])
    pad = torch.tensor([[False, False, True]])
    assert masked_loss(pred, tgt, pad, "l1").item() == pytest.approx(1.0)
    assert masked_loss(pred, tgt, pad, "mse").item() == pytest.approx(1.0)
    assert masked_loss(pred, tgt, torch.ones_like(pad), "l1").item() == 0.0


@pytest.mark.parametrize("kind", ["act", "bc"])
def test_policy_overfits_one_batch(kind):
    torch.manual_seed(0)
    pol = build_policy(kind, J, J, K, backbone="small", d_model=32)
    batch = _batch(hw=(24, 32))
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss = masked_loss(pol(batch), batch["action"], batch["action_is_pad"], "l1")
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3 * first
