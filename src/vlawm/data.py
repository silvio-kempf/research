"""Generate demonstration datasets from the scripted expert in ReachEnv.

Stores flat transition tuples used for (a) behavior cloning the VLA policy and
(b) training the latent / video world models. Trajectories are also recoverable via
the `episode_id` and `t` fields for sequence models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlawm.envs import ReachEnv, EnvConfig, Perturbation


@dataclass
class Dataset:
    images: np.ndarray       # (N,H,W,3) uint8
    next_images: np.ndarray  # (N,H,W,3) uint8
    color_ids: np.ndarray    # (N,) int
    actions: np.ndarray      # (N,2) float32
    states: np.ndarray       # (N,4) float32  [agent_xy, target_xy]
    next_states: np.ndarray  # (N,4) float32
    rewards: np.ndarray      # (N,) float32
    dones: np.ndarray        # (N,) bool
    episode_ids: np.ndarray  # (N,) int
    timesteps: np.ndarray    # (N,) int
    instructions: np.ndarray # (N,) str  (the language goal)

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.__dict__)

    @staticmethod
    def load(path: str) -> "Dataset":
        d = np.load(path)
        return Dataset(**{k: d[k] for k in Dataset.__dataclass_fields__})


def generate(n_episodes: int = 600, seed: int = 0, expert_noise: float = 0.05,
             n_distractors: int = 2, image_size: int = 64,
             explore_eps: float = 0.0, template_pool=None) -> Dataset:
    """Roll out the scripted expert to collect demonstrations.

    `explore_eps`: per-step probability of taking a uniform-random action instead of
    the expert action. Pure-expert data (eps=0) is ideal for behavior cloning, but a
    world model used for planning needs coverage *off* the expert manifold, so the
    world-model dataset is generated with eps>0 to avoid model-exploitation at plan time.

    `template_pool`: tuple of instruction-template indices to sample phrasings from each
    episode (instruction augmentation for P4). None = canonical phrasing only.
    """
    cfg = EnvConfig(seed=seed, image_size=image_size,
                    perturbation=Perturbation(n_distractors=n_distractors,
                                              template_pool=template_pool))
    env = ReachEnv(cfg)
    explore_rng = np.random.default_rng(seed + 777)
    buf: dict[str, list] = {k: [] for k in
                            ["images", "next_images", "color_ids", "actions",
                             "states", "next_states", "rewards", "dones",
                             "episode_ids", "timesteps", "instructions"]}
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 100003 + ep)
        done = False
        t = 0
        while not done:
            if explore_eps > 0 and explore_rng.random() < explore_eps:
                action = explore_rng.uniform(-1, 1, size=2).astype(np.float32)
            else:
                action = env.expert_action(noise=expert_noise)
            state = obs["state"].copy()
            img = obs["image"].copy()
            cid = obs["color_id"]
            nobs, reward, done, info = env.step(action)
            buf["images"].append(img)
            buf["next_images"].append(nobs["image"].copy())
            buf["color_ids"].append(cid)
            buf["actions"].append(action.astype(np.float32))
            buf["states"].append(state)
            buf["next_states"].append(nobs["state"].copy())
            buf["rewards"].append(np.float32(reward))
            buf["dones"].append(bool(info["is_terminal"]))
            buf["episode_ids"].append(ep)
            buf["timesteps"].append(t)
            buf["instructions"].append(obs["instruction"])
            obs = nobs
            t += 1
    return Dataset(
        images=np.asarray(buf["images"], dtype=np.uint8),
        next_images=np.asarray(buf["next_images"], dtype=np.uint8),
        color_ids=np.asarray(buf["color_ids"], dtype=np.int64),
        actions=np.asarray(buf["actions"], dtype=np.float32),
        states=np.asarray(buf["states"], dtype=np.float32),
        next_states=np.asarray(buf["next_states"], dtype=np.float32),
        rewards=np.asarray(buf["rewards"], dtype=np.float32),
        dones=np.asarray(buf["dones"], dtype=bool),
        episode_ids=np.asarray(buf["episode_ids"], dtype=np.int64),
        timesteps=np.asarray(buf["timesteps"], dtype=np.int64),
        instructions=np.asarray(buf["instructions"], dtype="U48"),
    )
