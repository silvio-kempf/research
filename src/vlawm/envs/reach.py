"""ReachEnv: a fast, image-observation Vision-Language-Action testbed.

A point-mass agent lives on a 2D canvas with several colored objects. A natural
language instruction ("reach the red object") selects the goal. This is a minimal
but genuine VLA task:
    - Vision   : a rendered RGB image (HxWx3)
    - Language : an instruction selecting a target by color
    - Action   : a continuous 2D velocity command

It is deliberately self-contained (pure NumPy rendering, no MuJoCo/sim deps) so the
whole research portfolio runs in seconds on a laptop, while exposing the levers that
matter for the experiments: distractors, novel colors, visual noise and instruction
paraphrases (for failure analysis), and clean deterministic dynamics (for world-model
learning). The methods built on top transfer directly to PushT / LIBERO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Canonical colors the policy is trained on. RGB in [0, 1].
BASE_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.75, 0.20),
    "blue": (0.20, 0.35, 0.90),
    "yellow": (0.95, 0.85, 0.10),
}
# Held-out colors used only to probe generalization (never seen in training).
NOVEL_COLORS: dict[str, tuple[float, float, float]] = {
    "purple": (0.60, 0.15, 0.75),
    "orange": (0.95, 0.55, 0.10),
    "cyan": (0.10, 0.80, 0.85),
}

# Instruction templates (paraphrases of the same goal). Index 0 is the canonical
# training phrasing; the rest probe / train language robustness (P3 diagnoses brittleness
# to unseen phrasings, P4 fixes it with instruction augmentation over a pool of these).
INSTRUCTION_TEMPLATES: list[str] = [
    "reach the {color} object",        # 0  canonical
    "go to the {color} one",           # 1
    "navigate towards the {color} target",  # 2
    "move to {color}",                 # 3
    "head for the {color} object",     # 4
    "approach the {color} thing",      # 5
    "drive to the {color} circle",     # 6
    "find the {color} object and touch it",  # 7
]


@dataclass
class Perturbation:
    """Controlled distribution shift for failure analysis (P3).

    Each field is off by default; turning one on isolates a single failure axis.
    """

    n_distractors: int = 2            # baseline number of non-target objects
    novel_target_color: bool = False  # target uses a held-out (unseen) color
    visual_noise: float = 0.0         # std of additive gaussian pixel noise
    paraphrase_idx: int = 0           # which INSTRUCTION_TEMPLATE to use (0 = canonical)
    distractor_overload: int = 0      # extra distractors beyond n_distractors
    template_pool: Optional[tuple] = None  # if set, sample phrasing from this pool of
    #                                        template indices each reset (instruction
    #                                        augmentation for P4); overrides paraphrase_idx


@dataclass
class EnvConfig:
    image_size: int = 64
    dt: float = 0.08
    max_steps: int = 40
    agent_radius: float = 0.045
    object_radius: float = 0.095
    success_radius: float = 0.08      # reach if dist(agent, target) < this
    action_scale: float = 0.11        # max velocity per step (world units)
    seed: int = 0
    perturbation: Perturbation = field(default_factory=Perturbation)


class ReachEnv:
    """Continuous 2D reach task with RGB observations and a language goal.

    World coordinates live in [0, 1]^2. Observations are uint8 images of shape
    (image_size, image_size, 3). Actions are 2D in [-1, 1] (scaled by action_scale).
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build_color_palette()
        self.agent = np.zeros(2, dtype=np.float32)
        self.objects: list[dict] = []
        self.target_idx = 0
        self.instruction = ""
        self.t = 0

    # ----------------------------------------------------------------- palette
    def _build_color_palette(self) -> None:
        self.base_names = list(BASE_COLORS.keys())
        self.base_rgb = {k: np.array(v, dtype=np.float32) for k, v in BASE_COLORS.items()}
        self.novel_rgb = {k: np.array(v, dtype=np.float32) for k, v in NOVEL_COLORS.items()}

    # -------------------------------------------------------------------- api
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self.cfg.perturbation
        n_obj = max(1, p.n_distractors + 1 + max(0, p.distractor_overload))

        # Sample non-overlapping object positions.
        positions = self._sample_positions(n_obj)

        # Target color: canonical base color, or a held-out novel color under shift.
        if p.novel_target_color:
            tname = self.rng.choice(list(self.novel_rgb.keys()))
            target_rgb = self.novel_rgb[tname]
        else:
            tname = self.rng.choice(self.base_names)
            target_rgb = self.base_rgb[tname]

        # Distractors get distinct base colors (excluding the target's name).
        distractor_pool = [c for c in self.base_names if c != tname]
        self.rng.shuffle(distractor_pool)

        self.objects = []
        self.target_idx = 0
        self.objects.append({"pos": positions[0], "rgb": target_rgb, "name": tname})
        for i in range(1, n_obj):
            cname = distractor_pool[(i - 1) % len(distractor_pool)]
            self.objects.append({"pos": positions[i], "rgb": self.base_rgb[cname], "name": cname})

        # Agent starts away from all objects.
        self.agent = self._sample_agent_start(positions)
        self.t = 0

        if p.template_pool is not None:
            idx = int(self.rng.choice(p.template_pool))
        else:
            idx = p.paraphrase_idx
        template = INSTRUCTION_TEMPLATES[idx % len(INSTRUCTION_TEMPLATES)]
        self.instruction = template.format(color=tname)
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.agent = self.agent + action * self.cfg.action_scale
        self.agent = np.clip(self.agent, 0.0, 1.0)
        self.t += 1

        dist = float(np.linalg.norm(self.agent - self.objects[self.target_idx]["pos"]))
        success = dist < self.cfg.success_radius
        reward = -dist + (5.0 if success else 0.0)
        done = success or self.t >= self.cfg.max_steps
        info = {"success": success, "dist": dist, "is_terminal": done}
        return self._obs(), reward, done, info

    # ------------------------------------------------------------- properties
    @property
    def target_color_id(self) -> int:
        """Integer id of the target's *name* in the canonical color vocabulary.

        Novel colors map to id = len(base) (a single 'unknown' token), which is what
        makes them genuinely out-of-distribution for the language conditioning.
        """
        name = self.objects[self.target_idx]["name"]
        if name in self.base_names:
            return self.base_names.index(name)
        return len(self.base_names)

    @property
    def n_color_tokens(self) -> int:
        return len(self.base_names) + 1  # +1 for the 'unknown'/novel token

    def state_vector(self) -> np.ndarray:
        """Low-dim privileged state (for world-model sanity baselines)."""
        target = self.objects[self.target_idx]["pos"]
        return np.concatenate([self.agent, target]).astype(np.float32)

    # ----------------------------------------------------------------- render
    def _obs(self) -> dict:
        return {
            "image": self.render(),
            "instruction": self.instruction,
            "color_id": self.target_color_id,
            "state": self.state_vector(),
        }

    def render(self) -> np.ndarray:
        s = self.cfg.image_size
        img = np.ones((s, s, 3), dtype=np.float32) * 0.12  # dark background
        yy, xx = np.mgrid[0:s, 0:s]
        grid = np.stack([xx, yy], axis=-1).astype(np.float32) / (s - 1)

        for obj in self.objects:
            self._paint_disc(img, grid, obj["pos"], self.cfg.object_radius, obj["rgb"])
        # Agent rendered as a white disc.
        self._paint_disc(img, grid, self.agent, self.cfg.agent_radius,
                         np.array([0.95, 0.95, 0.95], dtype=np.float32))

        noise = self.cfg.perturbation.visual_noise
        if noise > 0:
            img = img + self.rng.normal(0, noise, img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _paint_disc(img, grid, center, radius, rgb) -> None:
        d = np.linalg.norm(grid - center.reshape(1, 1, 2), axis=-1)
        mask = d < radius
        img[mask] = rgb

    # ----------------------------------------------------------------- helpers
    def _sample_positions(self, n: int) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        margin = self.cfg.object_radius + 0.05
        attempts = 0
        while len(positions) < n and attempts < 1000:
            attempts += 1
            cand = self.rng.uniform(margin, 1 - margin, size=2).astype(np.float32)
            if all(np.linalg.norm(cand - p) > 2.2 * self.cfg.object_radius for p in positions):
                positions.append(cand)
        while len(positions) < n:  # fallback if packing failed
            positions.append(self.rng.uniform(margin, 1 - margin, size=2).astype(np.float32))
        return positions

    def _sample_agent_start(self, positions: list[np.ndarray]) -> np.ndarray:
        margin = self.cfg.agent_radius + 0.02
        for _ in range(1000):
            cand = self.rng.uniform(margin, 1 - margin, size=2).astype(np.float32)
            if all(np.linalg.norm(cand - p) > 0.25 for p in positions):
                return cand
        return np.array([0.5, 0.5], dtype=np.float32)

    # ------------------------------------------------- scripted expert (data)
    def expert_action(self, noise: float = 0.0) -> np.ndarray:
        """Proportional controller toward the target — used to generate demos."""
        target = self.objects[self.target_idx]["pos"]
        delta = target - self.agent
        norm = np.linalg.norm(delta) + 1e-8
        act = delta / norm
        if noise > 0:
            act = act + self.rng.normal(0, noise, size=2).astype(np.float32)
        return np.clip(act, -1.0, 1.0)
