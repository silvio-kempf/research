"""Shared neural building blocks used across the VLA policy and world models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageEncoder(nn.Module):
    """Small CNN mapping (B,3,H,W) float images in [0,1] to a feature vector.

    Plain global encoder (no language conditioning) — used by the world models.
    """

    def __init__(self, image_size: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),   # 64->32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 32->16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(), # 16->8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),# 8->4
        )
        feat = 128 * (image_size // 16) * (image_size // 16)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class SpatialSoftmax(nn.Module):
    """Expected 2D coordinate of each feature channel's activation (Levine et al. 2016).

    Preserves *where* things are — essential for a policy that must move toward a
    specific object rather than regressing a global scene embedding.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", gx.reshape(-1))
        self.register_buffer("grid_y", gy.reshape(-1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        attn = torch.softmax(feat.reshape(b, c, h * w), dim=-1)
        exp_x = (attn * self.grid_x).sum(-1)
        exp_y = (attn * self.grid_y).sum(-1)
        return torch.stack([exp_x, exp_y], dim=-1).reshape(b, c * 2)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: condition conv features on a language vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)
        self.channels = channels

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


class LanguageConditionedEncoder(nn.Module):
    """Vision encoder whose conv features are FiLM-modulated by a language token, then
    reduced with spatial-softmax keypoints. This lets the network localize *the object
    named by the instruction* — the core of grounded VLA perception.
    """

    def __init__(self, image_size: int = 64, cond_dim: int = 16, out_dim: int = 128):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)    # 64->32
        self.c2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)   # 32->16
        self.c3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)   # 16->16
        self.film1 = FiLM(cond_dim, 32)
        self.film2 = FiLM(cond_dim, 64)
        self.film3 = FiLM(cond_dim, 64)
        self.spatial = SpatialSoftmax(image_size // 4, image_size // 4)
        self.head = nn.Sequential(nn.Linear(64 * 2, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.film1(self.c1(x), cond))
        h = torch.relu(self.film2(self.c2(h), cond))
        h = torch.relu(self.film3(self.c3(h), cond))
        return self.head(self.spatial(h))


def images_to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float in [0,1] on device."""
    if images.ndim == 3:
        images = images[None]
    x = torch.from_numpy(images).float().to(device) / 255.0
    return x.permute(0, 3, 1, 2).contiguous()
