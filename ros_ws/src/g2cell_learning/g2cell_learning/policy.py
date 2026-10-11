"""Policies: ACT-lite (CNN + transformer over an action chunk) and a plain BC-MLP baseline.

Both map a batch dict (as produced by ``EpisodeDataset``) to a normalised action chunk
``[B, K, action_dim]``.

ACT-lite vs. ACT (Zhao et al. 2023): same encoder/decoder layout (image feature tokens + proprio
token -> transformer encoder; K learned queries -> transformer decoder -> linear head; L1 loss on
the chunk) but **no CVAE latent** (deterministic), a small CNN or untrained ResNet-18, and no
temporal ensembling at inference (done by the caller if wanted).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SmallCNN(nn.Module):
    """4 stride-2 conv blocks: [B,3,H,W] -> [B,C,H/16,W/16]."""

    def __init__(self, out_channels: int = 128):
        super().__init__()
        chans = [3, 32, 64, 128, out_channels]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                       nn.GroupNorm(8, cout), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


def make_backbone(name: str) -> nn.Module:
    if name == "small":
        return SmallCNN()
    if name == "resnet18":
        import torchvision
        # weights=None: no download, trained from scratch (BatchNorm: use batch size > 1).
        r = torchvision.models.resnet18(weights=None)
        body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3,
                             r.layer4)
        body.out_channels = 512
        return body
    raise ValueError(f"unknown backbone {name!r} (small | resnet18)")


def sine_pos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Fixed 2-D sine position embedding, [h*w, dim]."""
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device) / max(d, 1)))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    ys, xs = ys.reshape(-1, 1) * omega, xs.reshape(-1, 1) * omega
    pe = torch.cat([ys.sin(), ys.cos(), xs.sin(), xs.cos()], dim=1)
    return F.pad(pe, (0, dim - pe.shape[1]))


class _Base(nn.Module):
    def __init__(self, state_dim, action_dim, chunk_size, cameras, backbone):
        super().__init__()
        self.state_dim, self.action_dim, self.chunk_size = state_dim, action_dim, chunk_size
        self.cameras = tuple(cameras)
        self.backbone = make_backbone(backbone)
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)

    def _features(self, batch) -> list[torch.Tensor]:
        """Per camera: [B, C, h, w] feature maps (shared backbone)."""
        imgs = [batch[f"observation.images.{c}"] for c in self.cameras]
        n = len(imgs)
        x = torch.cat(imgs, 0)  # one backbone pass for all cameras
        if x.dtype == torch.uint8:  # cached datasets ship uint8; convert on the device
            x = x.float().div_(255.0)
        x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
        f = self.backbone(x)
        return list(f.chunk(n, 0))


class ACTLite(_Base):
    def __init__(self, state_dim, action_dim, chunk_size, cameras=("head_front_rgb",),
                 backbone="small", d_model=256, nhead=8, enc_layers=4, dec_layers=4,
                 ff_dim=None, dropout=0.1):
        super().__init__(state_dim, action_dim, chunk_size, cameras, backbone)
        if d_model % nhead:
            raise ValueError(f"d_model={d_model} not divisible by nhead={nhead}")
        ff_dim = ff_dim or 4 * d_model
        self.d_model = d_model
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d_model, 1)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.cam_embed = nn.Embedding(len(self.cameras), d_model)
        self.state_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.queries = nn.Embedding(chunk_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)
        self.head = nn.Linear(d_model, action_dim)
        nn.init.normal_(self.queries.weight, std=0.02)

    def forward(self, batch) -> torch.Tensor:
        state = batch["observation.state"]
        tokens = [self.state_proj(state)[:, None] + self.state_pos]
        for ci, f in enumerate(self._features(batch)):
            f = self.input_proj(f)
            b, d, h, w = f.shape
            t = f.flatten(2).transpose(1, 2)  # [B, h*w, d]
            t = t + sine_pos_2d(h, w, d, f.device)[None] + self.cam_embed.weight[ci]
            tokens.append(t)
        src = torch.cat(tokens, 1)
        tgt = self.queries.weight[None].expand(state.shape[0], -1, -1)
        out = self.transformer(src, tgt)
        return self.head(out)


class BCMLP(_Base):
    """Global-pooled image features + state -> MLP -> flattened chunk."""

    def __init__(self, state_dim, action_dim, chunk_size, cameras=("head_front_rgb",),
                 backbone="small", d_model=256, hidden_layers=2, dropout=0.1):
        super().__init__(state_dim, action_dim, chunk_size, cameras, backbone)
        feat = self.backbone.out_channels * len(self.cameras) + state_dim
        layers, cin = [], feat
        for _ in range(hidden_layers):
            layers += [nn.Linear(cin, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            cin = d_model
        layers.append(nn.Linear(cin, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        feats = [f.mean(dim=(2, 3)) for f in self._features(batch)]
        x = torch.cat(feats + [batch["observation.state"]], 1)
        return self.mlp(x).view(-1, self.chunk_size, self.action_dim)


def build_policy(kind: str, state_dim: int, action_dim: int, chunk_size: int,
                 cameras: Sequence[str] = ("head_front_rgb",), backbone: str = "small",
                 d_model: int = 256, **kw) -> nn.Module:
    if kind == "act":
        return ACTLite(state_dim, action_dim, chunk_size, cameras, backbone, d_model=d_model,
                       **kw)
    if kind == "bc":
        return BCMLP(state_dim, action_dim, chunk_size, cameras, backbone, d_model=d_model, **kw)
    raise ValueError(f"unknown policy {kind!r} (act | bc)")


def masked_loss(pred, target, is_pad, kind: str = "l1") -> torch.Tensor:
    """Mean per-element loss over non-padded chunk steps (0 if everything is padded)."""
    err = (pred - target).abs() if kind == "l1" else (pred - target) ** 2
    valid = (~is_pad).unsqueeze(-1).to(err.dtype)
    denom = valid.sum() * err.shape[-1]
    return (err * valid).sum() / denom.clamp(min=1.0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


__all__ = ["ACTLite", "BCMLP", "build_policy", "masked_loss", "count_parameters"]
