"""Rollout metrics + baselines for P5.

Metric: per-horizon cosine distance between predicted and ground-truth latents,
averaged over clips. Baselines: persistence (copy last context latent),
unconditioned model, wrong-label model (built by callers via the model flags).
"""
from __future__ import annotations

import numpy as np


def persistence_rollout(context: np.ndarray, horizon: int) -> np.ndarray:
    """context: (B, C, D). Repeats the last context latent `horizon` times."""
    last = context[:, -1:]  # (B,1,D)
    return np.repeat(last, horizon, axis=1)


def rollout_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred,target: (B, H, D). Returns (H,) mean cosine distance per horizon step."""
    def norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    cos = (norm(pred) * norm(target)).sum(-1)  # (B,H)
    return (1.0 - cos).mean(axis=0)
