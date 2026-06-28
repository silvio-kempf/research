"""Latent preprocessing for P5: corrupt-frame removal and standardization.

Two fixes that turned a model losing to persistence into one that beats it:

1. clean_latents: drop frames whose latent jumps far from the previous kept frame
   (motion-blur / near-black transition frames whose DINOv2 latent is an outlier).
2. Standardizer: per-dimension z-score. DINOv2 latent dims have very different
   scales, so raw MSE is dominated by a few high-variance dims; standardizing lets
   the dynamics model learn all dims evenly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clean_mask(seq: np.ndarray, max_cos_dist: float = 0.3) -> list[int]:
    """Indices to keep: frame 0, then any frame <= max_cos_dist from last kept frame."""
    keep = [0]
    for i in range(1, len(seq)):
        a, b = seq[keep[-1]], seq[i]
        cos = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        if 1.0 - cos < max_cos_dist:
            keep.append(i)
    return keep


def clean_latents(seq: np.ndarray, max_cos_dist: float = 0.3) -> np.ndarray:
    """Drop corrupt frames whose latent jumps far from the previous kept frame."""
    return seq[clean_mask(seq, max_cos_dist)]


@dataclass
class Standardizer:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(mu=x.mean(0, keepdims=True), sd=x.std(0, keepdims=True) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.sd + self.mu
