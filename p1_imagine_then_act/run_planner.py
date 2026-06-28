"""P1 - Imagine-then-Act: planning through a learned latent world model.

Headline result: a world model trained only to predict distance-to-goal is an accurate
enough neural simulator that *planning through it* (random-shooting MPC, no policy) drives
the agent to the instructed object - while a random policy fails. Adding the VLA as a
proposal (policy-guided MPC) is the "VLA + world model in the loop" configuration.

Experiments:
  (A) Controller comparison: random policy / world-model MPC / policy-guided MPC / VLA.
  (B) Success vs. number of imagined candidates K (more imagination -> better control).

Usage: uv run python p1_imagine_then_act/run_planner.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlawm.eval import evaluate
from vlawm.nn import get_device
from vlawm.vla import VLAPolicy
from vlawm.worldmodel import LatentWorldModel
from vlawm.worldmodel.mpc import WorldModelMPC, random_policy_act_fn

OUT = "results"
N_EP = 100


def load(device):
    policy = VLAPolicy().to(device)
    policy.load_state_dict(torch.load(f"{OUT}/policy.pt", map_location=device, weights_only=True))
    policy.eval()
    wm = LatentWorldModel().to(device)
    wm.load_state_dict(torch.load(f"{OUT}/world_model.pt", map_location=device, weights_only=True))
    wm.eval()
    return policy, wm


def controller_comparison(policy, wm, device):
    results = {}
    results["random policy"] = evaluate(random_policy_act_fn(seed=1), n_episodes=N_EP).success_rate
    results["world-model MPC\n(no policy)"] = evaluate(
        WorldModelMPC(wm, device, n_candidates=32, seed=1).act, n_episodes=N_EP).success_rate
    results["policy-guided MPC\n(VLA + world model)"] = evaluate(
        WorldModelMPC(wm, device, n_candidates=32, policy=policy, policy_weight=1.0,
                      sigma=0.6, seed=1).act, n_episodes=N_EP).success_rate
    results["VLA only"] = evaluate(
        lambda obs, env: policy.act(obs, device), n_episodes=N_EP).success_rate
    for k, v in results.items():
        print(f"  {k.replace(chr(10),' '):<40} {v:.2f}")
    return results


def candidate_sweep(wm, device):
    Ks = [1, 2, 4, 8, 16, 32, 64]
    succ = []
    for K in Ks:
        m = evaluate(WorldModelMPC(wm, device, n_candidates=K, seed=2).act, n_episodes=N_EP)
        succ.append(m.success_rate)
        print(f"  K={K:>2}  success={m.success_rate:.2f}")
    return Ks, succ


def plot(results, Ks, succ, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    names = list(results.keys()); vals = list(results.values())
    colors = ["#e76f51", "#e9c46a", "#2a9d8f", "#264653"]
    ax1.bar(range(len(names)), vals, color=colors)
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("success rate"); ax1.set_ylim(0, 1.02)
    ax1.set_title("(A) Planning through the world model enables control")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    ax2.plot(Ks, succ, "s-", color="#2a9d8f")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("# imagined candidates K"); ax2.set_ylabel("success rate")
    ax2.set_title("(B) More imagination -> better control (no policy)")
    ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.3)
    fig.suptitle("P1 - Imagine-then-Act: a latent world model inside the control loop", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"[save] {path}")


def main():
    device = get_device()
    print(f"[device] {device}")
    policy, wm = load(device)
    print("[exp A] controller comparison")
    results = controller_comparison(policy, wm, device)
    print("[exp B] success vs number of candidates (pure world-model MPC)")
    Ks, succ = candidate_sweep(wm, device)
    plot(results, Ks, succ, f"{OUT}/p1_imagine_then_act.png")
    with open(f"{OUT}/p1_metrics.json", "w") as f:
        json.dump({"comparison": results, "K": Ks, "success_vs_K": succ}, f, indent=2)
    print(f"[save] {OUT}/p1_metrics.json")


if __name__ == "__main__":
    main()
