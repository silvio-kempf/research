"""Turn predicted latents back into watchable frames via nearest-neighbor lookup.

Avoids training a pixel decoder: each predicted latent is mapped to the real frame
whose latent is closest (cosine). Good enough to "watch" a latent rollout.
"""
from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class ReferenceBank:
    def __init__(self, latents: np.ndarray, frames: np.ndarray):
        assert latents.shape[0] == frames.shape[0]
        self.latents = _normalize(latents.astype("float32"))
        self.frames = frames

    def nearest_index(self, query: np.ndarray) -> int:
        q = _normalize(query.astype("float32").reshape(1, -1))
        sims = (self.latents @ q.T).ravel()
        return int(np.argmax(sims))

    def retrieve_frames(self, preds: np.ndarray) -> np.ndarray:
        idxs = [self.nearest_index(p) for p in preds]
        return self.frames[idxs]
