"""P3 - When do VLAs fail? A controlled distribution-shift study.

Takes the trained VLA policy and evaluates it across one-axis-at-a-time perturbations:
    - distractor overload   (more objects than seen in training)
    - novel target colors   (held-out colors -> <unk> language token)
    - visual noise          (sensor corruption)
    - instruction paraphrase(unseen phrasings of the same goal)

Produces results/p3_failure_analysis.png and a JSON of metrics. Each axis is a
different *kind* of generalization gap, and (P1) a world model can both flag and
repair some of them.

Usage: uv run python p3_failure_analysis/run_failure_analysis.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlawm.eval import evaluate
from vlawm.envs import Perturbation
from vlawm.nn import get_device
from vlawm.vla import VLAPolicy

OUT = "results"


def load_policy(device):
    policy = VLAPolicy().to(device)
    policy.load_state_dict(torch.load(os.path.join(OUT, "policy.pt"),
                                      map_location=device, weights_only=True))
    policy.eval()
    return policy


def make_act_fn(policy, device):
    return lambda obs, env: policy.act(obs, device)


def sweep(act_fn):
    """Return {axis_name: [(label, success_rate, wrong_object_rate), ...]}."""
    n = 80
    axes: dict[str, list] = {}

    # Baseline reference (training distribution: 2 distractors, canonical phrasing).
    base = evaluate(act_fn, n_episodes=n, perturbation=Perturbation(n_distractors=2))
    base_pt = ("in-dist", base.success_rate, base.wrong_object_rate)

    axes["distractor overload"] = [base_pt] + [
        (f"+{k}", *_metrics(evaluate(act_fn, n_episodes=n,
            perturbation=Perturbation(n_distractors=2, distractor_overload=k))))
        for k in (2, 4, 6)
    ]
    axes["visual noise (sigma)"] = [base_pt] + [
        (f"{s:.2f}", *_metrics(evaluate(act_fn, n_episodes=n,
            perturbation=Perturbation(n_distractors=2, visual_noise=s))))
        for s in (0.15, 0.30, 0.50)
    ]
    axes["instruction paraphrase"] = [base_pt] + [
        (f"para#{p}", *_metrics(evaluate(act_fn, n_episodes=n,
            perturbation=Perturbation(n_distractors=2, paraphrase_idx=p))))
        for p in (1, 2, 3)
    ]
    axes["novel target color"] = [
        base_pt,
        ("novel", *_metrics(evaluate(act_fn, n_episodes=n,
            perturbation=Perturbation(n_distractors=2, novel_target_color=True)))),
    ]
    return axes, base


def _metrics(m):
    return m.success_rate, m.wrong_object_rate


def plot(axes: dict, path: str):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (name, rows) in zip(axs, axes.items()):
        labels = [r[0] for r in rows]
        succ = [r[1] for r in rows]
        wrong = [r[2] for r in rows]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, succ, 0.4, label="success", color="#2a9d8f")
        ax.bar(x + 0.2, wrong, 0.4, label="wrong-object", color="#e76f51")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
        ax.set_title(name); ax.set_ylim(0, 1.0)
        ax.axhline(rows[0][1], ls="--", c="gray", lw=1, alpha=0.7)
    axs[0].set_ylabel("rate")
    axs[0].legend(loc="lower left")
    fig.suptitle("P3 - VLA robustness under controlled distribution shift "
                 "(dashed = in-distribution success)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[save] {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    print(f"[device] {device}")
    policy = load_policy(device)
    act_fn = make_act_fn(policy, device)

    axes, base = sweep(act_fn)
    print(f"[baseline] success={base.success_rate:.2f} wrong-obj={base.wrong_object_rate:.2f}")
    for name, rows in axes.items():
        print(f"  {name}:")
        for label, s, w in rows:
            print(f"     {label:>8}  success={s:.2f}  wrong-obj={w:.2f}")

    plot(axes, os.path.join(OUT, "p3_failure_analysis.png"))
    with open(os.path.join(OUT, "p3_metrics.json"), "w") as f:
        json.dump({k: [list(r) for r in v] for k, v in axes.items()}, f, indent=2)
    print(f"[save] {os.path.join(OUT, 'p3_metrics.json')}")


if __name__ == "__main__":
    main()
