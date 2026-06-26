"""Evaluation harness: roll out any act-function in ReachEnv and compute metrics.

`ActFn` is the single interface every controller implements — the plain VLA policy,
and the Imagine-then-Act planner (P1) — so they are compared on identical episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

from vlawm.envs import ReachEnv, EnvConfig, Perturbation

# An ActFn maps (observation, env) -> action. Most controllers ignore `env`, but the
# world-model planner uses it read-only (e.g., to know action bounds).
ActFn = Callable[[dict, ReachEnv], np.ndarray]


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_dist: float
    wrong_object: bool  # ended closer to a distractor than to the target


@dataclass
class EvalMetrics:
    success_rate: float
    wrong_object_rate: float
    mean_final_dist: float
    mean_steps_on_success: float
    n_episodes: int

    def as_dict(self) -> dict:
        return asdict(self)


def rollout(env: ReachEnv, act_fn: ActFn, seed: int) -> EpisodeResult:
    obs = env.reset(seed=seed)
    done = False
    info = {"success": False, "dist": 1.0, "is_terminal": False}
    steps = 0
    while not done:
        action = act_fn(obs, env)
        obs, _, done, info = env.step(action)
        steps += 1

    # Did we end up nearest a distractor instead of the target?
    agent = env.agent
    dists = [float(np.linalg.norm(agent - o["pos"])) for o in env.objects]
    nearest = int(np.argmin(dists))
    wrong = (nearest != env.target_idx) and not info["success"]
    return EpisodeResult(bool(info["success"]), steps, float(info["dist"]), wrong)


def evaluate(act_fn: ActFn, n_episodes: int = 100, base_seed: int = 10_000,
             perturbation: Optional[Perturbation] = None,
             image_size: int = 64) -> EvalMetrics:
    cfg = EnvConfig(image_size=image_size,
                    perturbation=perturbation or Perturbation())
    env = ReachEnv(cfg)
    results = [rollout(env, act_fn, seed=base_seed + i) for i in range(n_episodes)]
    succ = [r for r in results if r.success]
    return EvalMetrics(
        success_rate=float(np.mean([r.success for r in results])),
        wrong_object_rate=float(np.mean([r.wrong_object for r in results])),
        mean_final_dist=float(np.mean([r.final_dist for r in results])),
        mean_steps_on_success=float(np.mean([r.steps for r in succ])) if succ else float("nan"),
        n_episodes=n_episodes,
    )
