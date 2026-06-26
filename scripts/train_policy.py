"""Behavior-clone the toy VLA policy from scripted-expert demos.

Usage:
    uv run python scripts/train_policy.py --episodes 600 --epochs 12
Outputs:
    results/policy.pt           trained policy weights
    results/dataset.npz         cached demonstration dataset (reused by world models)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from vlawm.data import generate, Dataset
from vlawm.nn import get_device
from vlawm.vla import VLAPolicy
from vlawm.vla.text import encode_ids
from vlawm.eval import evaluate
from vlawm.envs import ReachEnv


def make_act_fn(policy: VLAPolicy, device):
    def act(obs, env: ReachEnv):
        return policy.act(obs, device)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"[device] {device}")

    ds_path = os.path.join(args.out, "dataset.npz")
    if os.path.exists(ds_path):
        print(f"[data] loading cached {ds_path}")
        ds = Dataset.load(ds_path)
    else:
        print(f"[data] generating {args.episodes} episodes...")
        ds = generate(n_episodes=args.episodes, seed=args.seed)
        ds.save(ds_path)
    print(f"[data] {len(ds)} transitions")

    policy = VLAPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    images = torch.from_numpy(ds.images).to(device)  # keep uint8 on device
    token_ids = torch.tensor([encode_ids(s) for s in ds.instructions],
                             dtype=torch.long, device=device)
    actions = torch.from_numpy(ds.actions).to(device)
    n = len(ds)

    for ep in range(args.epochs):
        policy.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        t0 = time.time()
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            img = images[idx].float().permute(0, 3, 1, 2) / 255.0
            pred = policy(img, token_ids[idx])
            loss = loss_fn(pred, actions[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"[train] epoch {ep+1}/{args.epochs} loss={total/n:.4f} ({time.time()-t0:.1f}s)")

    torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))
    print(f"[save] {os.path.join(args.out, 'policy.pt')}")

    metrics = evaluate(make_act_fn(policy, device), n_episodes=100)
    print(f"[eval-clean] {metrics.as_dict()}")


if __name__ == "__main__":
    main()
