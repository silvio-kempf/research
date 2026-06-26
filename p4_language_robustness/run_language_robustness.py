"""P4 - Fixing the language bottleneck found in P3 with instruction augmentation.

P3 showed the VLA is language-brittle: trained on a single instruction phrasing, it
collapses (~6%) on unseen paraphrases. Here we test the hypothesis that this is a
*training-distribution coverage* problem, not a fundamental limit: we train a second,
identical policy on a POOL of phrasings (instruction augmentation) and evaluate it on
HELD-OUT phrasings it never saw during training.

Train phrasings : templates {0,1,2,4,5}
Held-out phrasings (test): templates {3,6,7}   (never seen in training)

Comparison:
    - baseline  : results/policy.pt        (trained on the canonical phrasing only, from P3)
    - augmented : results/policy_aug.pt     (trained on the phrasing pool, this script)

Usage: uv run python p4_language_robustness/run_language_robustness.py
Outputs: results/policy_aug.pt, results/p4_language_robustness.png, results/p4_metrics.json
"""
from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from vlawm.data import generate, Dataset
from vlawm.eval import evaluate
from vlawm.envs import Perturbation
from vlawm.nn import get_device
from vlawm.vla import VLAPolicy
from vlawm.vla.text import encode_ids

OUT = "results"
TRAIN_TEMPLATES = (0, 1, 2, 4, 5)
HELDOUT_TEMPLATES = (3, 6, 7)
N_EP = 100


def train_augmented(device, episodes=2500, epochs=60, batch=256, lr=3e-4):
    path = os.path.join(OUT, "dataset_aug.npz")
    if os.path.exists(path):
        ds = Dataset.load(path)
    else:
        print(f"[data] generating {episodes} augmented episodes (phrasings {TRAIN_TEMPLATES})")
        ds = generate(n_episodes=episodes, seed=0, template_pool=TRAIN_TEMPLATES)
        ds.save(path)
    policy = VLAPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    images = torch.from_numpy(ds.images).to(device)
    tokens = torch.tensor([encode_ids(s) for s in ds.instructions], dtype=torch.long, device=device)
    actions = torch.from_numpy(ds.actions).to(device)
    n = len(ds)
    for ep in range(epochs):
        policy.train()
        perm = torch.randperm(n, device=device)
        total = 0.0; t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            img = images[idx].float().permute(0, 3, 1, 2) / 255.0
            loss = loss_fn(policy(img, tokens[idx]), actions[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"[train-aug] epoch {ep+1}/{epochs} loss={total/n:.4f} ({time.time()-t0:.1f}s)")
    torch.save(policy.state_dict(), os.path.join(OUT, "policy_aug.pt"))
    print(f"[save] {OUT}/policy_aug.pt")
    return policy


def load_baseline(device):
    p = VLAPolicy().to(device)
    p.load_state_dict(torch.load(f"{OUT}/policy.pt", map_location=device, weights_only=True))
    p.eval()
    return p


def eval_on_templates(policy, device, templates):
    act = lambda obs, env: policy.act(obs, device)
    return [evaluate(act, n_episodes=N_EP,
                     perturbation=Perturbation(paraphrase_idx=t)).success_rate
            for t in templates]


def main():
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    print(f"[device] {device}")
    baseline = load_baseline(device)
    augmented = (VLAPolicy().to(device))
    aug_path = f"{OUT}/policy_aug.pt"
    if os.path.exists(aug_path):
        augmented.load_state_dict(torch.load(aug_path, map_location=device, weights_only=True))
        augmented.eval()
        print("[load] cached policy_aug.pt")
    else:
        augmented = train_augmented(device)
        augmented.eval()

    # In-distribution canonical phrasing (template 0) and held-out paraphrases.
    res = {}
    res["baseline_canonical"] = eval_on_templates(baseline, device, (0,))[0]
    res["augmented_canonical"] = eval_on_templates(augmented, device, (0,))[0]
    res["baseline_heldout"] = eval_on_templates(baseline, device, HELDOUT_TEMPLATES)
    res["augmented_heldout"] = eval_on_templates(augmented, device, HELDOUT_TEMPLATES)

    print(f"[canonical]  baseline={res['baseline_canonical']:.2f}  augmented={res['augmented_canonical']:.2f}")
    for t, b, a in zip(HELDOUT_TEMPLATES, res["baseline_heldout"], res["augmented_heldout"]):
        print(f"[heldout #{t}] baseline={b:.2f}  augmented={a:.2f}")

    # Plot: grouped bars over canonical + each held-out paraphrase.
    labels = ["canonical\n(#0)"] + [f"held-out\n#{t}" for t in HELDOUT_TEMPLATES]
    base_vals = [res["baseline_canonical"]] + res["baseline_heldout"]
    aug_vals = [res["augmented_canonical"]] + res["augmented_heldout"]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.2, base_vals, 0.4, label="baseline (single phrasing)", color="#e76f51")
    plt.bar(x + 0.2, aug_vals, 0.4, label="augmented (phrasing pool)", color="#2a9d8f")
    plt.xticks(x, labels)
    plt.ylabel("success rate"); plt.ylim(0, 1.05)
    plt.title("P4 - Instruction augmentation fixes the P3 language bottleneck\n"
              "(held-out phrasings were never seen in training)")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(base_vals): plt.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(aug_vals): plt.text(i + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/p4_language_robustness.png", dpi=130)
    print(f"[save] {OUT}/p4_language_robustness.png")

    with open(f"{OUT}/p4_metrics.json", "w") as f:
        json.dump({"train_templates": TRAIN_TEMPLATES, "heldout_templates": HELDOUT_TEMPLATES,
                   **res}, f, indent=2)
    print(f"[save] {OUT}/p4_metrics.json")


if __name__ == "__main__":
    main()
