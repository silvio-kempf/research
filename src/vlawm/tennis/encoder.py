"""Frozen DINOv2 encoder: each frame -> 384-d latent (CLS token), cached to disk.

Vision is "solved for free" by a pretrained, frozen backbone. The world model then
only has to learn dynamics in this latent space, which is the part learnable from
our ~490 frames. Mirrors the latent-space philosophy of P1 (latent_wm.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vlawm.nn import get_device

EMBED_DIM = 384
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(device: torch.device | None = None) -> torch.nn.Module:
    device = device or get_device()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def embed_frames(model: torch.nn.Module, frames: np.ndarray,
                 device: torch.device | None = None, batch: int = 16) -> np.ndarray:
    """frames: (T,H,W,3) uint8 with H,W multiples of 14. Returns (T,384) float32."""
    device = device or get_device()
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    outs = []
    for i in range(0, x.shape[0], batch):
        chunk = x[i:i + batch].to(device)
        feat = model(chunk)  # (B, 384) CLS token
        outs.append(feat.cpu().numpy().astype("float32"))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def embed_frames_patches(model: torch.nn.Module, frames: np.ndarray,
                         device: torch.device | None = None, batch: int = 8) -> np.ndarray:
    """frames: (T,H,W,3) uint8. Returns (T, N_patches, 384) spatial patch tokens.

    Unlike the global CLS token, patch tokens keep *where* things are, so a ball on
    the right of the image shows up in the right-side patches.
    """
    device = device or get_device()
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    outs = []
    for i in range(0, x.shape[0], batch):
        feat = model.forward_features(x[i:i + batch].to(device))["x_norm_patchtokens"]
        outs.append(feat.cpu().numpy().astype("float32"))
    return np.concatenate(outs, axis=0)


def cache_path(emb_dir: Path, clip_name: str) -> Path:
    return emb_dir / f"{Path(clip_name).stem}.npy"


def embed_clip_cached(model: torch.nn.Module, frames: np.ndarray,
                      emb_dir: Path, clip_name: str) -> np.ndarray:
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(emb_dir, clip_name)
    if path.exists():
        return np.load(path)
    emb = embed_frames(model, frames)
    np.save(path, emb)
    return emb
