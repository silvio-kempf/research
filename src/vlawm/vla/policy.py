"""Toy VLA policy: image + language(color token) -> action.

A behavior-cloned CNN policy that stands in for a real Vision-Language-Action model.
It consumes the same modalities a real VLA does (pixels + a language goal) and emits
continuous actions, which is all the downstream world-model machinery (P1) and the
failure analysis (P3) need. Swapping this for SmolVLA/Octo on PushT/LIBERO is a drop-in
change at the `act()` interface.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vlawm.nn import LanguageConditionedEncoder, images_to_tensor
from vlawm.vla.text import TextEncoder


class VLAPolicy(nn.Module):
    """Vision-Language-Action policy: RGB image + instruction text -> 2D action."""

    def __init__(self, image_size: int = 64, feat_dim: int = 128,
                 lang_dim: int = 16, action_dim: int = 2):
        super().__init__()
        self.text = TextEncoder(out_dim=lang_dim)
        self.encoder = LanguageConditionedEncoder(image_size, lang_dim, feat_dim)
        self.policy = nn.Sequential(
            nn.Linear(feat_dim + lang_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        lang = self.text(token_ids)
        feat = self.encoder(images, lang)
        return self.policy(torch.cat([feat, lang], dim=-1))

    @torch.no_grad()
    def act(self, obs: dict, device: torch.device) -> np.ndarray:
        self.eval()
        img = images_to_tensor(obs["image"], device)
        lang = self.text.encode_texts([obs["instruction"]], device)
        feat = self.encoder(img, lang)
        return self.policy(torch.cat([feat, lang], dim=-1))[0].cpu().numpy()
