"""Imagine-then-Act controllers: planning through a learned latent world model.

Random-shooting model-predictive control. At each step we sample K candidate action
*sequences*, imagine each H-step rollout in latent space, score it by the predicted
distance-to-goal, and execute the first action of the best sequence (receding horizon).

Two controllers:
    - WorldModelMPC : candidates sampled around zero (NO policy) -> shows the world model
                      alone is an accurate enough simulator to control the agent.
    - PolicyGuidedMPC: candidates sampled around the VLA's action -> the world model
                      verifies/refines the policy (VLA + world model in the loop).

A pure random policy is the natural lower-bound baseline.
"""
from __future__ import annotations

import numpy as np
import torch

from vlawm.nn import images_to_tensor
from vlawm.vla.text import encode_ids


class WorldModelMPC:
    def __init__(self, world_model, device, n_candidates: int = 32, horizon: int = 5,
                 sigma: float = 1.0, policy=None, policy_weight: float = 0.0, seed: int = 0):
        self.wm = world_model
        self.device = device
        self.K = n_candidates
        self.H = horizon
        self.sigma = sigma
        self.policy = policy            # optional proposal policy
        self.pw = policy_weight         # 0 = pure world-model planning
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, env) -> np.ndarray:
        mean = np.zeros(2, np.float32)
        if self.policy is not None and self.pw > 0:
            mean = self.pw * self.policy.act(obs, self.device).astype(np.float32)

        # Candidate action sequences: (K, H, 2)
        seqs = (mean[None, None, :] +
                self.rng.normal(0, self.sigma, size=(self.K, self.H, 2)).astype(np.float32))
        seqs = np.clip(seqs, -1.0, 1.0)

        img = images_to_tensor(obs["image"], self.device)
        tok = torch.tensor([encode_ids(obs["instruction"])], dtype=torch.long, device=self.device)
        z0 = self.wm.encode(img, tok).repeat(self.K, 1)
        seqs_t = torch.from_numpy(seqs).to(self.device)
        dists = self.wm.rollout(z0, seqs_t)              # (K, H) predicted distance per step
        # Score: reach the goal AND stay there -> weight final step, plus best-point bonus.
        score = dists[:, -1] + 0.5 * dists.min(dim=1).values
        best = int(torch.argmin(score).item())
        return seqs[best, 0]


def random_policy_act_fn(seed: int = 0):
    """Lower bound: uniform random actions."""
    rng = np.random.default_rng(seed)
    return lambda obs, env: rng.uniform(-1, 1, size=2).astype(np.float32)
