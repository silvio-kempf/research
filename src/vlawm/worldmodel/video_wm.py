"""Action-conditioned video world model: predict the next *frame* given action.

A convolutional encoder-decoder that maps (frame_t, action_t) -> frame_{t+1}. Rolling it
out autoregressively lets the model "dream" a video of an imagined trajectory - a pixel-
space neural simulator in the spirit of UniSim / Genie, shrunk to run on a laptop. Such a
model can render synthetic rollouts for training or evaluating a VLA without a simulator.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VideoWorldModel(nn.Module):
    def __init__(self, image_size: int = 64, action_dim: int = 2, latent: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),    # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # 16->8
        )
        self.enc_dim = 128 * (image_size // 8) * (image_size // 8)
        self.fc_in = nn.Linear(self.enc_dim + action_dim, latent)
        self.fc_out = nn.Linear(latent, self.enc_dim)
        self.spatial = image_size // 8
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 16->32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),  # 32->64
        )

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """frame: (B,3,H,W) in [0,1]; action: (B,2). Returns predicted next frame (residual)."""
        h = self.enc(frame).flatten(1)
        z = torch.relu(self.fc_in(torch.cat([h, action], dim=-1)))
        h2 = self.fc_out(z).view(-1, 128, self.spatial, self.spatial)
        delta = self.dec(h2)
        # Residual prediction around the input frame keeps static background crisp.
        return torch.clamp(frame + (delta - 0.5), 0.0, 1.0)

    @torch.no_grad()
    def dream(self, frame0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Autoregressively roll out. actions: (T,2). Returns frames (T,3,H,W)."""
        f = frame0
        out = []
        for t in range(actions.shape[0]):
            f = self.forward(f, actions[t:t + 1])
            out.append(f)
        return torch.cat(out, dim=0)
