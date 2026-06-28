"""Swing-conditioned latent dynamics (P5).

Predicts z_{t+1} from a GRU over past latents, FiLM-modulated by a swing-type
embedding (forehand/backhand = the "action"/intent). Residual transition keeps
the prediction near the current latent (stabler; same trick as latent_wm.py).
Setting conditioned=False ignores the label, giving the unconditioned baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VectorFiLM(nn.Module):
    """FiLM for 1-D feature vectors (cf. nn.FiLM which operates on conv maps)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return feat * (1 + gamma) + beta


class SwingConditionedDynamics(nn.Module):
    def __init__(self, latent_dim: int = 384, n_labels: int = 2,
                 hidden: int = 256, label_dim: int = 16, conditioned: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.conditioned = conditioned
        self.label_emb = nn.Embedding(n_labels, label_dim)
        self.gru = nn.GRUCell(latent_dim, hidden)
        self.film = VectorFiLM(label_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def init_hidden(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=self.label_emb.weight.device)

    def step(self, z: torch.Tensor, label: torch.Tensor,
             h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.gru(z, h)
        if self.conditioned:
            h = self.film(h, self.label_emb(label))
        z_next = z + self.head(h)  # residual transition
        return z_next, h

    def rollout(self, context: torch.Tensor, label: torch.Tensor,
                horizon: int) -> torch.Tensor:
        """context: (B, C, D) warmup latents. Returns (B, horizon, D) predictions."""
        b = context.shape[0]
        h = self.init_hidden(b)
        z = context[:, 0]
        for t in range(context.shape[1]):  # warm up GRU on the context (teacher forced)
            z = context[:, t]
            _, h = self.step(z, label, h)
        preds = []
        for _ in range(horizon):  # free-running from last context latent
            z, h = self.step(z, label, h)
            preds.append(z)
        return torch.stack(preds, dim=1)
