"""Latent world model for Imagine-then-Act planning (P1).

Components:
    - a language-conditioned image encoder  : obs image + instruction -> latent z0
    - a deterministic latent transition      : (z_t, a_t) -> z_{t+1}
    - a goal head                            : z_t -> predicted distance-to-target

The model is trained to roll out H steps in latent space and predict, at every step,
the agent's distance to the *instructed* target. The planner (mpc.py) then imagines the
outcome of candidate action sequences and selects the one minimizing predicted distance.

This is the core project idea in miniature: a world model placed *inside* the
VLA's control loop to verify and improve its actions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vlawm.nn import LanguageConditionedEncoder
from vlawm.vla.text import TextEncoder


class LatentWorldModel(nn.Module):
    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 lang_dim: int = 16, action_dim: int = 2):
        super().__init__()
        self.text = TextEncoder(out_dim=lang_dim)
        self.encoder = LanguageConditionedEncoder(image_size, lang_dim, latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.goal_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Softplus(),  # distance >= 0
        )
        self.latent_dim = latent_dim

    def encode(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        lang = self.text(token_ids)
        return self.encoder(images, lang)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Residual latent transition (stabler than predicting z' from scratch)."""
        return z + self.transition(torch.cat([z, a], dim=-1))

    def predict_distance(self, z: torch.Tensor) -> torch.Tensor:
        return self.goal_head(z).squeeze(-1)

    def rollout(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Imagine a rollout. action_seq: (B, H, A). Returns predicted distances (B, H)."""
        z = z0
        dists = []
        for t in range(action_seq.shape[1]):
            z = self.step(z, action_seq[:, t])
            dists.append(self.predict_distance(z))
        return torch.stack(dists, dim=1)
