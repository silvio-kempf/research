"""Train the latent world model for Imagine-then-Act planning (P1).

Learns to predict the agent's distance-to-target over an imagined H-step latent rollout,
from the start image + instruction + an action sequence. Reuses the demonstration dataset
cached by train_policy.py.

Usage: uv run python scripts/train_world_model.py --epochs 40 --horizon 5
Output: results/world_model.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from vlawm.data import Dataset, generate
from vlawm.nn import get_device
from vlawm.vla.text import encode_ids
from vlawm.worldmodel import LatentWorldModel

OUT = "results"


def build_windows(ds: Dataset, horizon: int):
    """Return arrays of (start_img_idx, action_window, dist_window, mask) per window."""
    dist_after = np.linalg.norm(ds.next_states[:, :2] - ds.next_states[:, 2:], axis=1)
    # group transition indices by episode, ordered by timestep
    order = np.lexsort((ds.timesteps, ds.episode_ids))
    start_idx, act_win, dist_win, masks = [], [], [], []
    ep_to_indices: dict[int, list[int]] = {}
    for i in order:
        ep_to_indices.setdefault(int(ds.episode_ids[i]), []).append(int(i))
    for idxs in ep_to_indices.values():
        for s in range(len(idxs)):
            window = idxs[s:s + horizon]
            a = np.zeros((horizon, ds.actions.shape[1]), np.float32)
            d = np.zeros(horizon, np.float32)
            m = np.zeros(horizon, np.float32)
            for k, j in enumerate(window):
                a[k] = ds.actions[j]
                d[k] = dist_after[j]
                m[k] = 1.0
            start_idx.append(idxs[s])
            act_win.append(a); dist_win.append(d); masks.append(m)
    return (np.asarray(start_idx), np.asarray(act_win),
            np.asarray(dist_win), np.asarray(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--explore_eps", type=float, default=0.4)
    args = ap.parse_args()

    device = get_device()
    print(f"[device] {device}")
    wm_path = os.path.join(OUT, "wm_dataset.npz")
    if os.path.exists(wm_path):
        print(f"[data] loading cached {wm_path}")
        ds = Dataset.load(wm_path)
    else:
        print(f"[data] generating {args.episodes} episodes with explore_eps={args.explore_eps} "
              "(coverage off the expert manifold for planning)...")
        ds = generate(n_episodes=args.episodes, seed=7, explore_eps=args.explore_eps)
        ds.save(wm_path)
    start_idx, act_win, dist_win, masks = build_windows(ds, args.horizon)
    print(f"[data] {len(start_idx)} rollout windows (H={args.horizon})")

    images = torch.from_numpy(ds.images).to(device)
    token_ids = torch.tensor([encode_ids(s) for s in ds.instructions],
                             dtype=torch.long, device=device)
    start_idx_t = torch.from_numpy(start_idx).long().to(device)
    act_win_t = torch.from_numpy(act_win).to(device)
    dist_win_t = torch.from_numpy(dist_win).to(device)
    masks_t = torch.from_numpy(masks).to(device)

    wm = LatentWorldModel().to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    n = len(start_idx)

    for ep in range(args.epochs):
        wm.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        t0 = time.time()
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            si = start_idx_t[b]
            img = images[si].float().permute(0, 3, 1, 2) / 255.0
            z0 = wm.encode(img, token_ids[si])
            pred = wm.rollout(z0, act_win_t[b])           # (B,H)
            m = masks_t[b]
            loss = (((pred - dist_win_t[b]) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        print(f"[train] epoch {ep+1}/{args.epochs} dist-MSE={total/n:.5f} ({time.time()-t0:.1f}s)")

    torch.save(wm.state_dict(), os.path.join(OUT, "world_model.pt"))
    print(f"[save] {os.path.join(OUT, 'world_model.pt')}")


if __name__ == "__main__":
    main()
