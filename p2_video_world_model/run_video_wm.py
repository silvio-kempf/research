"""P2 - Action-conditioned video world model ("dreaming").

Trains a conv encoder-decoder to predict the next frame from (frame, action), then rolls
it out autoregressively to dream an imagined trajectory from a single start frame + an
action sequence. Produces:
    results/p2_dream_vs_real.png   filmstrip: real rollout vs dreamed rollout
    results/p2_rollout_error.png   per-step pixel error of autoregressive dreaming
    results/p2_dream.gif           animated dreamed trajectory

Usage: uv run python p2_video_world_model/run_video_wm.py --epochs 25
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from vlawm.data import Dataset, generate
from vlawm.nn import get_device
from vlawm.worldmodel.video_wm import VideoWorldModel

OUT = "results"


def get_dataset():
    path = os.path.join(OUT, "dataset.npz")
    if os.path.exists(path):
        return Dataset.load(path)
    ds = generate(n_episodes=2500, seed=0)
    ds.save(path)
    return ds


def episode_slices(ds: Dataset):
    """Yield (indices) per episode, ordered by timestep."""
    order = np.lexsort((ds.timesteps, ds.episode_ids))
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(int(ds.episode_ids[i]), []).append(int(i))
    return list(groups.values())


def build_seq_windows(ds: Dataset, L: int):
    """(start_img_idx, action_window (L,2), target_frames idx window (L,))."""
    starts, awins, twins = [], [], []
    for ids in episode_slices(ds):
        for s in range(len(ids)):
            win = ids[s:s + L]
            if len(win) < 2:
                continue
            a = np.zeros((L, 2), np.float32)
            tgt = np.zeros(L, np.int64)
            for k, j in enumerate(win):
                a[k] = ds.actions[j]; tgt[k] = j
            # pad short windows by repeating last (masked out via valid length)
            for k in range(len(win), L):
                a[k] = 0.0; tgt[k] = win[-1]
            starts.append(ids[s]); awins.append(a); twins.append(tgt)
    return np.asarray(starts), np.asarray(awins), np.asarray(twins)


def train(wm, ds, device, epochs, batch, lr, horizon=4):
    """Free-running multi-step training with a change-weighted pixel loss.

    Supervising autoregressive rollouts (not just one-step) forces the model to actually
    *move* the agent; weighting pixels that change between frames stops the loss from
    being dominated by the static background.
    """
    images = torch.from_numpy(ds.images).to(device)
    next_images = torch.from_numpy(ds.next_images).to(device)
    starts, awins, twins = build_seq_windows(ds, horizon)
    starts_t = torch.from_numpy(starts).long().to(device)
    awins_t = torch.from_numpy(awins).to(device)
    twins_t = torch.from_numpy(twins).long().to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    n = len(starts)
    print(f"[data] {n} rollout windows (H={horizon})")
    for ep in range(epochs):
        wm.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            f = images[starts_t[b]].float().permute(0, 3, 1, 2) / 255.0
            acts = awins_t[b]                          # (B,H,2)
            tgt = next_images[twins_t[b]].float().permute(0, 1, 4, 2, 3) / 255.0  # (B,H,3,Hh,Ww)
            loss = 0.0
            cur = f
            prev_gt = f
            for t in range(horizon):
                cur = wm(cur, acts[:, t])
                gt = tgt[:, t]
                # weight pixels that changed from the previous GT frame (the moving agent)
                w = 1.0 + 8.0 * (gt - prev_gt).abs().amax(dim=1, keepdim=True)
                loss = loss + (w * (cur - gt) ** 2).mean()
                prev_gt = gt
            loss = loss / horizon
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        print(f"[train] epoch {ep+1}/{epochs} weighted-rollout-loss={total/n:.5f}")


def to_img(t):  # (3,H,W) float -> (H,W,3) uint8
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def evaluate_and_visualize(wm, ds, device):
    eps = episode_slices(ds)
    # pick a reasonably long episode for a nice dream
    eps = sorted(eps, key=len, reverse=True)
    idxs = eps[0][:8]
    start = torch.from_numpy(ds.images[idxs[0]]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    acts = torch.from_numpy(ds.actions[idxs]).to(device)
    real = [ds.images[i] for i in idxs] + [ds.next_images[idxs[-1]]]
    dreamed_t = wm.dream(start, acts)
    dreamed = [to_img(dreamed_t[t]) for t in range(dreamed_t.shape[0])]

    # Filmstrip: real (top) vs dreamed (bottom)
    T = len(dreamed)
    fig, axs = plt.subplots(2, T, figsize=(1.5 * T, 3.2))
    for t in range(T):
        axs[0, t].imshow(real[t + 1]); axs[0, t].axis("off")
        axs[1, t].imshow(dreamed[t]); axs[1, t].axis("off")
        axs[0, t].set_title(f"t+{t+1}", fontsize=8)
    axs[0, 0].set_ylabel("REAL", fontsize=10)
    axs[1, 0].set_ylabel("DREAM", fontsize=10)
    for r, lab in [(0, "REAL"), (1, "DREAM")]:
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
        axs[r, 0].set_ylabel(lab, fontsize=11)
    fig.suptitle("P2 - Action-conditioned dreaming: real vs imagined rollout", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/p2_dream_vs_real.png", dpi=130)
    print(f"[save] {OUT}/p2_dream_vs_real.png")

    # GIF of the dream
    imageio.mimsave(f"{OUT}/p2_dream.gif",
                    [np.array(d) for d in dreamed], duration=0.25)
    print(f"[save] {OUT}/p2_dream.gif")

    # Autoregressive rollout error vs horizon, averaged over several episodes.
    errs = []
    for ep in eps[:40]:
        ids = ep[:8]
        if len(ids) < 4:
            continue
        s = torch.from_numpy(ds.images[ids[0]]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        a = torch.from_numpy(ds.actions[ids]).to(device)
        d = wm.dream(s, a)
        gt = torch.from_numpy(np.stack([ds.next_images[i] for i in ids])).float().permute(0, 3, 1, 2).to(device) / 255.0
        per_step = ((d - gt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        errs.append(per_step)
    maxlen = max(len(e) for e in errs)
    padded = np.full((len(errs), maxlen), np.nan)
    for i, e in enumerate(errs):
        padded[i, :len(e)] = e
    mean_err = np.nanmean(padded, axis=0)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(mean_err) + 1), mean_err, "o-", color="#264653")
    plt.xlabel("autoregressive dream step"); plt.ylabel("pixel MSE")
    plt.title("P2 - Dreaming accuracy vs rollout horizon")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/p2_rollout_error.png", dpi=130)
    print(f"[save] {OUT}/p2_rollout_error.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0)
    args = ap.parse_args()
    lr = args.lr or 5e-4

    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    print(f"[device] {device}")
    ds = get_dataset()
    wm = VideoWorldModel().to(device)
    train(wm, ds, device, args.epochs, args.batch, lr)
    torch.save(wm.state_dict(), f"{OUT}/video_world_model.pt")
    print(f"[save] {OUT}/video_world_model.pt")
    wm.eval()
    evaluate_and_visualize(wm, ds, device)


if __name__ == "__main__":
    main()
