# P5 Tennis Latent World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn intent-conditioned dynamics from 11 real handheld tennis clips by predicting the next frame in a frozen-DINOv2 latent space, conditioned on swing type (forehand/backhand), and visualize predictions via nearest-neighbor retrieval.

**Architecture:** A frozen DINOv2 encoder maps every frame to a 384-d latent (cached once). A small FiLM-conditioned GRU learns the latent transition `z_t -> z_{t+1}` given a swing-type embedding. Evaluation compares conditioned rollout error against persistence, unconditioned, and wrong-label baselines; the payoff is a counterfactual label-swap. A nearest-neighbor reference bank turns predicted latents back into watchable frames without training a decoder. Primary artifact is an annotated teaching notebook; a `run_tennis_wm.py` provides the non-interactive path.

**Tech Stack:** Python 3.12, PyTorch + MPS, torch.hub DINOv2 ViT-S/14, imageio-ffmpeg (video decode), scikit-learn (PCA), matplotlib/imageio (figures+gif), pytest.

---

## File Structure

- Create: `src/vlawm/tennis/__init__.py` — package marker.
- Create: `src/vlawm/tennis/data.py` — clip discovery, label parsing, video decode, clip-level split.
- Create: `src/vlawm/tennis/encoder.py` — frozen DINOv2 wrapper, frame embedding, on-disk cache.
- Create: `src/vlawm/worldmodel/tennis_wm.py` — `SwingConditionedDynamics` (vector FiLM + GRU).
- Create: `src/vlawm/tennis/retrieval.py` — reference bank + nearest-neighbor frame lookup.
- Create: `src/vlawm/tennis/evaluate.py` — rollout, rollout-error metric, baselines.
- Create: `p5_tennis_world_model/run_tennis_wm.py` — end-to-end non-interactive driver producing the figures.
- Create: `p5_tennis_world_model/p5_walkthrough.ipynb` — primary teaching artifact.
- Create: `p5_tennis_world_model/README.md` — write-up (no em dashes).
- Create: `tests/test_tennis_data.py`, `tests/test_tennis_model.py`, `tests/test_tennis_retrieval.py`.
- Modify: `pyproject.toml` — add `scikit-learn`, `pytest` deps.

Conventions to follow (from existing repo): residual transitions (`z + delta`, see `latent_wm.py:step`), FiLM from `src/vlawm/nn.py:61`, `get_device()` from `src/vlawm/nn.py:9`, module docstrings explaining the thesis link (as in `video_wm.py`).

---

## Task 1: Dependencies and test scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Add deps**

In `pyproject.toml`, add to `dependencies` list (after `"pyyaml>=6.0",`):

```toml
    "scikit-learn>=1.4",
    "pytest>=8.0",
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: resolves and installs scikit-learn + pytest.

- [ ] **Step 3: Create empty test package marker**

Create `tests/__init__.py` with no content.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "P5: add scikit-learn + pytest deps and test scaffold"
```

---

## Task 2: Data pipeline (label parsing + split)

**Files:**
- Create: `src/vlawm/tennis/__init__.py`
- Create: `src/vlawm/tennis/data.py`
- Test: `tests/test_tennis_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tennis_data.py`:

```python
from pathlib import Path

from vlawm.tennis.data import label_from_name, train_test_split_clips, Clip


def test_label_from_name_forehand():
    assert label_from_name("forehand_03.MOV") == 0


def test_label_from_name_backhand():
    assert label_from_name("backhand_12.mov") == 1


def test_label_from_name_rejects_unknown():
    try:
        label_from_name("serve_01.MOV")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown label")


def _clip(name, label):
    return Clip(path=Path(name), label=label, name=name)


def test_split_holds_out_one_per_class_deterministically():
    clips = [_clip(f"forehand_{i}.MOV", 0) for i in range(5)]
    clips += [_clip(f"backhand_{i}.MOV", 1) for i in range(6)]
    train, test = train_test_split_clips(clips, seed=0)
    assert len(test) == 2
    assert {c.label for c in test} == {0, 1}
    assert len(train) == 9
    # determinism
    train2, test2 = train_test_split_clips(clips, seed=0)
    assert [c.name for c in test] == [c.name for c in test2]
    # no leakage
    test_names = {c.name for c in test}
    assert all(c.name not in test_names for c in train)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tennis_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlawm.tennis'`.

- [ ] **Step 3: Implement**

Create `src/vlawm/tennis/__init__.py` (empty).

Create `src/vlawm/tennis/data.py`:

```python
"""P5 data pipeline: discover tennis clips, parse swing labels, decode frames.

Labels are intent ("action") for the world model: forehand=0, backhand=1. We split
by clip (never by frame) so frames from one swing never leak across train/test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

LABELS = {"forehand": 0, "backhand": 1}
LABEL_NAMES = ["forehand", "backhand"]


@dataclass
class Clip:
    path: Path
    label: int
    name: str


def label_from_name(name: str) -> int:
    stem = name.lower()
    for key, idx in LABELS.items():
        if stem.startswith(key):
            return idx
    raise ValueError(f"cannot parse forehand/backhand from {name!r}")


def discover_clips(raw_dir: Path) -> list[Clip]:
    clips = []
    for p in sorted(raw_dir.iterdir()):
        if p.suffix.lower() not in {".mov", ".mp4"}:
            continue
        clips.append(Clip(path=p, label=label_from_name(p.name), name=p.name))
    return clips


def train_test_split_clips(clips: list[Clip], seed: int = 0) -> tuple[list[Clip], list[Clip]]:
    """Hold out exactly one clip per class as test; rest is train. Deterministic."""
    rng = random.Random(seed)
    test = []
    for label in (0, 1):
        members = [c for c in clips if c.label == label]
        test.append(rng.choice(members))
    test_names = {c.name for c in test}
    train = [c for c in clips if c.name not in test_names]
    return train, test


def decode_clip(clip: Clip, image_size: int = 224) -> np.ndarray:
    """Decode a clip to (T, H, W, 3) uint8, center-cropped square then resized.

    Returns frames at native fps (clips are ~1-1.5s, so no downsampling).
    """
    frames = iio.imread(clip.path, plugin="pyav")  # (T, H, W, 3) uint8
    out = np.stack([_center_crop_resize(f, image_size) for f in frames])
    return out


def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame[top:top + side, left:left + side]
    img = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
    return np.asarray(img)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tennis_data.py -v`
Expected: 4 passed. (`decode_clip` is exercised later against real files; the unit tests cover the deterministic logic.)

- [ ] **Step 5: Commit**

```bash
git add src/vlawm/tennis/__init__.py src/vlawm/tennis/data.py tests/test_tennis_data.py
git commit -m "P5: clip discovery, label parsing, clip-level split"
```

---

## Task 3: Frozen DINOv2 encoder + embedding cache

**Files:**
- Create: `src/vlawm/tennis/encoder.py`
- Test: covered by a smoke step (network + model download), not a unit test.

- [ ] **Step 1: Implement encoder**

Create `src/vlawm/tennis/encoder.py`:

```python
"""Frozen DINOv2 encoder: each frame -> 384-d latent (CLS token), cached to disk.

Vision is "solved for free" by a pretrained, frozen backbone. The world model then
only has to learn dynamics in this latent space, which is the part learnable from
our ~490 frames. Mirrors the latent-space philosophy of P1 (latent_wm.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vlawm.nn import get_device

EMBED_DIM = 384
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(device: torch.device | None = None) -> torch.nn.Module:
    device = device or get_device()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def embed_frames(model: torch.nn.Module, frames: np.ndarray,
                 device: torch.device | None = None, batch: int = 16) -> np.ndarray:
    """frames: (T,H,W,3) uint8 with H,W multiples of 14. Returns (T,384) float32."""
    device = device or get_device()
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    outs = []
    for i in range(0, x.shape[0], batch):
        chunk = x[i:i + batch].to(device)
        feat = model(chunk)  # (B, 384) CLS token
        outs.append(feat.cpu().numpy().astype("float32"))
    return np.concatenate(outs, axis=0)


def cache_path(emb_dir: Path, clip_name: str) -> Path:
    return emb_dir / f"{Path(clip_name).stem}.npy"


def embed_clip_cached(model: torch.nn.Module, frames: np.ndarray,
                      emb_dir: Path, clip_name: str) -> np.ndarray:
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(emb_dir, clip_name)
    if path.exists():
        return np.load(path)
    emb = embed_frames(model, frames)
    np.save(path, emb)
    return emb
```

- [ ] **Step 2: Smoke test against real data**

Run (downloads DINOv2 weights on first call; needs network):

```bash
uv run python -c "
from pathlib import Path
from vlawm.tennis.data import discover_clips, decode_clip
from vlawm.tennis.encoder import load_encoder, embed_clip_cached
raw = Path('p5_tennis_world_model/data/raw')
clips = discover_clips(raw)
print('clips:', len(clips))
m = load_encoder()
c = clips[0]
emb = embed_clip_cached(m, decode_clip(c), raw.parent/'embeddings', c.name)
print('embedding shape:', emb.shape)
assert emb.shape[1] == 384
"
```
Expected: prints `clips: 11`, `embedding shape: (T, 384)` with T in ~30-60.

- [ ] **Step 3: Commit**

```bash
git add src/vlawm/tennis/encoder.py
git commit -m "P5: frozen DINOv2 encoder with on-disk embedding cache"
```

---

## Task 4: Swing-conditioned latent dynamics model

**Files:**
- Create: `src/vlawm/worldmodel/tennis_wm.py`
- Test: `tests/test_tennis_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tennis_model.py`:

```python
import torch

from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics


def test_step_shapes():
    model = SwingConditionedDynamics(latent_dim=384, n_labels=2)
    z = torch.randn(4, 384)
    h = model.init_hidden(4)
    label = torch.tensor([0, 1, 0, 1])
    z_next, h_next = model.step(z, label, h)
    assert z_next.shape == (4, 384)
    assert h_next.shape == h.shape


def test_rollout_length():
    model = SwingConditionedDynamics(latent_dim=384, n_labels=2)
    context = torch.randn(2, 3, 384)  # (B, C, D)
    label = torch.tensor([0, 1])
    preds = model.rollout(context, label, horizon=5)
    assert preds.shape == (2, 5, 384)


def test_unconditioned_flag_runs():
    model = SwingConditionedDynamics(latent_dim=384, n_labels=2, conditioned=False)
    context = torch.randn(2, 3, 384)
    label = torch.tensor([0, 1])
    preds = model.rollout(context, label, horizon=4)
    assert preds.shape == (2, 4, 384)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tennis_model.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `tennis_wm`.

- [ ] **Step 3: Implement**

Create `src/vlawm/worldmodel/tennis_wm.py`:

```python
"""Swing-conditioned latent dynamics (P5).

Predicts z_{t+1} from a GRU over past latents, FiLM-modulated by a swing-type
embedding (forehand/backhand = the "action"/intent). Residual transition keeps
the prediction near the current latent (stabler; same trick as latent_wm.py).
Setting conditioned=False ignores the label, giving the unconditioned baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VectorFiLM(nn.Module):
    """FiLM for 1-D feature vectors (cf. nn.FiLM which operates on conv maps)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return feat * (1 + gamma) + beta


class SwingConditionedDynamics(nn.Module):
    def __init__(self, latent_dim: int = 384, n_labels: int = 2,
                 hidden: int = 256, label_dim: int = 16, conditioned: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.conditioned = conditioned
        self.label_emb = nn.Embedding(n_labels, label_dim)
        self.gru = nn.GRUCell(latent_dim, hidden)
        self.film = VectorFiLM(label_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def init_hidden(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=self.label_emb.weight.device)

    def step(self, z: torch.Tensor, label: torch.Tensor,
             h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.gru(z, h)
        if self.conditioned:
            h = self.film(h, self.label_emb(label))
        z_next = z + self.head(h)  # residual transition
        return z_next, h

    def rollout(self, context: torch.Tensor, label: torch.Tensor,
                horizon: int) -> torch.Tensor:
        """context: (B, C, D) warmup latents. Returns (B, horizon, D) predictions."""
        b = context.shape[0]
        h = self.init_hidden(b)
        z = context[:, 0]
        for t in range(context.shape[1]):  # warm up GRU on the context (teacher forced)
            z = context[:, t]
            _, h = self.step(z, label, h)
        preds = []
        for _ in range(horizon):  # free-running from last context latent
            z, h = self.step(z, label, h)
            preds.append(z)
        return torch.stack(preds, dim=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tennis_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vlawm/worldmodel/tennis_wm.py tests/test_tennis_model.py
git commit -m "P5: swing-conditioned latent dynamics (FiLM GRU, residual)"
```

---

## Task 5: Nearest-neighbor retrieval (visualization, no decoder)

**Files:**
- Create: `src/vlawm/tennis/retrieval.py`
- Test: `tests/test_tennis_retrieval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tennis_retrieval.py`:

```python
import numpy as np

from vlawm.tennis.retrieval import ReferenceBank


def test_retrieves_exact_match_index():
    latents = np.eye(5, dtype="float32")  # 5 orthogonal latents
    frames = np.arange(5 * 2 * 2 * 3, dtype="uint8").reshape(5, 2, 2, 3)
    bank = ReferenceBank(latents, frames)
    query = np.array([0, 0, 1, 0, 0], dtype="float32")  # closest to row 2
    idx = bank.nearest_index(query)
    assert idx == 2


def test_retrieve_frames_shape():
    latents = np.random.randn(7, 4).astype("float32")
    frames = np.zeros((7, 8, 8, 3), dtype="uint8")
    bank = ReferenceBank(latents, frames)
    preds = np.random.randn(3, 4).astype("float32")
    out = bank.retrieve_frames(preds)
    assert out.shape == (3, 8, 8, 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tennis_retrieval.py -v`
Expected: FAIL with import error for `retrieval`.

- [ ] **Step 3: Implement**

Create `src/vlawm/tennis/retrieval.py`:

```python
"""Turn predicted latents back into watchable frames via nearest-neighbor lookup.

Avoids training a pixel decoder: each predicted latent is mapped to the real frame
whose latent is closest (cosine). Good enough to "watch" a latent rollout.
"""
from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class ReferenceBank:
    def __init__(self, latents: np.ndarray, frames: np.ndarray):
        assert latents.shape[0] == frames.shape[0]
        self.latents = _normalize(latents.astype("float32"))
        self.frames = frames

    def nearest_index(self, query: np.ndarray) -> int:
        q = _normalize(query.astype("float32").reshape(1, -1))
        sims = (self.latents @ q.T).ravel()
        return int(np.argmax(sims))

    def retrieve_frames(self, preds: np.ndarray) -> np.ndarray:
        idxs = [self.nearest_index(p) for p in preds]
        return self.frames[idxs]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tennis_retrieval.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vlawm/tennis/retrieval.py tests/test_tennis_retrieval.py
git commit -m "P5: nearest-neighbor reference bank for latent visualization"
```

---

## Task 6: Evaluation and baselines

**Files:**
- Create: `src/vlawm/tennis/evaluate.py`
- Test: `tests/test_tennis_model.py` (add baseline tests here)

- [ ] **Step 1: Write failing test**

Append to `tests/test_tennis_model.py`:

```python
import numpy as np
from vlawm.tennis.evaluate import persistence_rollout, rollout_error


def test_persistence_repeats_last_context_latent():
    context = np.random.randn(2, 3, 8).astype("float32")
    preds = persistence_rollout(context, horizon=4)
    assert preds.shape == (2, 4, 8)
    # every predicted step equals the last context latent
    for t in range(4):
        assert np.allclose(preds[:, t], context[:, -1])


def test_rollout_error_zero_for_identical():
    a = np.random.randn(2, 5, 8).astype("float32")
    err = rollout_error(a, a)
    assert err.shape == (5,)
    assert np.allclose(err, 0.0, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tennis_model.py -v`
Expected: FAIL with import error for `evaluate`.

- [ ] **Step 3: Implement**

Create `src/vlawm/tennis/evaluate.py`:

```python
"""Rollout metrics + baselines for P5.

Metric: per-horizon cosine distance between predicted and ground-truth latents,
averaged over clips. Baselines: persistence (copy last context latent),
unconditioned model, wrong-label model (built by callers via the model flags).
"""
from __future__ import annotations

import numpy as np


def persistence_rollout(context: np.ndarray, horizon: int) -> np.ndarray:
    """context: (B, C, D). Repeats the last context latent `horizon` times."""
    last = context[:, -1:]  # (B,1,D)
    return np.repeat(last, horizon, axis=1)


def rollout_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred,target: (B, H, D). Returns (H,) mean cosine distance per horizon step."""
    def norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    cos = (norm(pred) * norm(target)).sum(-1)  # (B,H)
    return (1.0 - cos).mean(axis=0)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (data + model + retrieval + evaluate).

- [ ] **Step 5: Commit**

```bash
git add src/vlawm/tennis/evaluate.py tests/test_tennis_model.py
git commit -m "P5: rollout error metric + persistence baseline"
```

---

## Task 7: End-to-end driver script (figures + gif)

**Files:**
- Create: `p5_tennis_world_model/run_tennis_wm.py`

- [ ] **Step 1: Implement the driver**

Create `p5_tennis_world_model/run_tennis_wm.py`:

```python
"""P5 end-to-end: embed clips, train swing-conditioned dynamics, evaluate, visualize.

Produces results/p5_rollout_error.png, results/p5_counterfactual.png, results/p5_dream.gif.
Non-interactive twin of p5_walkthrough.ipynb. Runs on Mac MPS in a few minutes.
"""
from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from vlawm.nn import get_device
from vlawm.tennis.data import discover_clips, decode_clip, train_test_split_clips, LABEL_NAMES
from vlawm.tennis.encoder import load_encoder, embed_clip_cached
from vlawm.tennis.evaluate import persistence_rollout, rollout_error
from vlawm.tennis.retrieval import ReferenceBank
from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
EMB = ROOT / "data" / "embeddings"
RESULTS = ROOT.parent / "results"
CONTEXT = 4
HORIZON = 10
EPOCHS = 300
SEED = 0


def build_windows(emb_by_clip, labels_by_clip):
    """Yield (context (C,D), target (H,D), label) windows from each clip."""
    ctx, tgt, lab = [], [], []
    for name, emb in emb_by_clip.items():
        if emb.shape[0] < CONTEXT + HORIZON:
            continue
        for s in range(emb.shape[0] - CONTEXT - HORIZON + 1):
            ctx.append(emb[s:s + CONTEXT])
            tgt.append(emb[s + CONTEXT:s + CONTEXT + HORIZON])
            lab.append(labels_by_clip[name])
    return (np.stack(ctx).astype("float32"),
            np.stack(tgt).astype("float32"),
            np.array(lab))


def train(model, ctx, tgt, lab, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx_t = torch.from_numpy(ctx).to(device)
    tgt_t = torch.from_numpy(tgt).to(device)
    lab_t = torch.from_numpy(lab).to(device)
    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = model.rollout(ctx_t, lab_t, HORIZON)
        mse = torch.nn.functional.mse_loss(pred, tgt_t)
        cos = 1 - torch.nn.functional.cosine_similarity(pred, tgt_t, dim=-1).mean()
        loss = mse + cos
        loss.backward()
        opt.step()
    return model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    RESULTS.mkdir(exist_ok=True)

    clips = discover_clips(RAW)
    encoder = load_encoder(device)
    frames_by_clip, emb_by_clip, labels_by_clip = {}, {}, {}
    for c in clips:
        fr = decode_clip(c)
        frames_by_clip[c.name] = fr
        emb_by_clip[c.name] = embed_clip_cached(encoder, fr, EMB, c.name)
        labels_by_clip[c.name] = c.label

    train_clips, test_clips = train_test_split_clips(clips, seed=SEED)
    tr_emb = {c.name: emb_by_clip[c.name] for c in train_clips}
    te_emb = {c.name: emb_by_clip[c.name] for c in test_clips}
    tr_lab = {c.name: labels_by_clip[c.name] for c in train_clips}
    te_lab = {c.name: labels_by_clip[c.name] for c in test_clips}

    ctx, tgt, lab = build_windows(tr_emb, tr_lab)

    cond = train(SwingConditionedDynamics().to(device), ctx, tgt, lab, device)
    uncond = train(SwingConditionedDynamics(conditioned=False).to(device), ctx, tgt, lab, device)

    # ---- evaluate on held-out clips ----
    te_ctx, te_tgt, te_lab_arr = build_windows(te_emb, te_lab)
    te_ctx_t = torch.from_numpy(te_ctx).to(device)
    te_lab_t = torch.from_numpy(te_lab_arr).to(device)
    wrong_lab_t = 1 - te_lab_t

    with torch.no_grad():
        p_cond = cond.rollout(te_ctx_t, te_lab_t, HORIZON).cpu().numpy()
        p_uncond = uncond.rollout(te_ctx_t, te_lab_t, HORIZON).cpu().numpy()
        p_wrong = cond.rollout(te_ctx_t, wrong_lab_t, HORIZON).cpu().numpy()
    p_persist = persistence_rollout(te_ctx, HORIZON)

    e_cond = rollout_error(p_cond, te_tgt)
    e_uncond = rollout_error(p_uncond, te_tgt)
    e_wrong = rollout_error(p_wrong, te_tgt)
    e_persist = rollout_error(p_persist, te_tgt)

    plt.figure(figsize=(6, 4))
    h = np.arange(1, HORIZON + 1)
    plt.plot(h, e_cond, "-o", label="conditioned")
    plt.plot(h, e_uncond, "-s", label="unconditioned")
    plt.plot(h, e_wrong, "-^", label="wrong label")
    plt.plot(h, e_persist, "--", label="persistence")
    plt.xlabel("rollout horizon (frames)")
    plt.ylabel("cosine error")
    plt.title("P5: latent rollout error vs horizon (held-out swings)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "p5_rollout_error.png", dpi=120)
    plt.close()

    # ---- counterfactual + dream gif on one held-out forehand clip ----
    fh = next(c for c in test_clips if c.label == 0)
    all_emb = np.concatenate([emb_by_clip[c.name] for c in clips])
    all_frames = np.concatenate([frames_by_clip[c.name] for c in clips])
    bank = ReferenceBank(all_emb, all_frames)

    ctx0 = torch.from_numpy(emb_by_clip[fh.name][None, :CONTEXT]).to(device)
    with torch.no_grad():
        roll_fh = cond.rollout(ctx0, torch.tensor([0], device=device), HORIZON).cpu().numpy()[0]
        roll_bh = cond.rollout(ctx0, torch.tensor([1], device=device), HORIZON).cpu().numpy()[0]
    frames_fh = bank.retrieve_frames(roll_fh)
    frames_bh = bank.retrieve_frames(roll_bh)

    n = min(5, HORIZON)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(frames_fh[i]); axes[0, i].axis("off")
        axes[1, i].imshow(frames_bh[i]); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("label=forehand", fontsize=9)
    axes[1, 0].set_ylabel("label=backhand", fontsize=9)
    fig.suptitle("P5 counterfactual: same forehand context, swapped intent label")
    plt.tight_layout()
    plt.savefig(RESULTS / "p5_counterfactual.png", dpi=120)
    plt.close()

    imageio.mimsave(RESULTS / "p5_dream.gif", list(frames_fh), fps=6, loop=0)

    print("cond     :", np.round(e_cond, 4))
    print("uncond   :", np.round(e_uncond, 4))
    print("wrong    :", np.round(e_wrong, 4))
    print("persist  :", np.round(e_persist, 4))
    print("wrote figures to", RESULTS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end to end**

Run: `uv run python p5_tennis_world_model/run_tennis_wm.py`
Expected: prints four error arrays and writes 3 files in `results/`. Sanity: `e_cond` mean <= `e_uncond` mean and clearly < `e_persist` at longer horizons (relative ordering may be noisy given tiny data; record whatever it is honestly).

- [ ] **Step 3: Verify outputs exist**

Run: `ls -la results/p5_rollout_error.png results/p5_counterfactual.png results/p5_dream.gif`
Expected: three files present, non-zero size.

- [ ] **Step 4: Commit**

```bash
git add p5_tennis_world_model/run_tennis_wm.py results/p5_rollout_error.png results/p5_counterfactual.png results/p5_dream.gif
git commit -m "P5: end-to-end driver producing rollout, counterfactual, dream figures"
```

---

## Task 8: Teaching notebook

**Files:**
- Create: `p5_tennis_world_model/p5_walkthrough.ipynb`

- [ ] **Step 1: Build the notebook**

Create the notebook with `nbformat` so cells are exact. Run this builder once, then delete it:

Create `p5_tennis_world_model/_build_nb.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# P5: A Latent World Model of My Tennis Swings

This notebook teaches world models by building one on real video: 11 handheld clips
of my forehands and backhands. We predict the *next frame* not in pixels but in a
**latent space**, conditioned on the swing type (the "action"/intent).

**Concepts as we go:** what a world model is, why latent beats pixels on tiny data,
FiLM conditioning, teacher forcing, rollout error, and a counterfactual test.
Connects to P1 (latent control) and P2 (action-conditioned dreaming).""")

md("""## 1. What is a world model?

A world model learns dynamics: given the current state and an action, predict the
next state, `s_t, a_t -> s_{t+1}`. Roll it forward and you can *imagine* futures
without the real environment. Here:
- **state** = a latent embedding of a video frame,
- **action/intent** = the swing label (forehand / backhand),
- **dynamics** = a small GRU we train.

Why latent, not pixels? With only ~490 frames a pixel model cannot learn what the
world looks like, so it predicts blur. A frozen pretrained encoder gives us good
visual features for free, so the model only learns the part that is learnable: how
the latent evolves.""")

code("""from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from vlawm.nn import get_device
from vlawm.tennis.data import discover_clips, decode_clip, train_test_split_clips, LABEL_NAMES
from vlawm.tennis.encoder import load_encoder, embed_clip_cached
from vlawm.tennis.retrieval import ReferenceBank
from vlawm.tennis.evaluate import persistence_rollout, rollout_error
from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics

ROOT = Path.cwd() if Path.cwd().name == 'p5_tennis_world_model' else Path.cwd()/'p5_tennis_world_model'
RAW, EMB = ROOT/'data'/'raw', ROOT/'data'/'embeddings'
device = get_device(); print('device:', device)""")

md("## 2. Load and peek at the data\\nDecode clips and look at a forehand vs a backhand frame.")
code("""clips = discover_clips(RAW)
print(len(clips), 'clips:', [(c.name, LABEL_NAMES[c.label]) for c in clips])
frames = {c.name: decode_clip(c) for c in clips}
fh = next(c for c in clips if c.label==0); bh = next(c for c in clips if c.label==1)
fig, ax = plt.subplots(1,2, figsize=(6,3))
ax[0].imshow(frames[fh.name][len(frames[fh.name])//2]); ax[0].set_title('forehand'); ax[0].axis('off')
ax[1].imshow(frames[bh.name][len(frames[bh.name])//2]); ax[1].set_title('backhand'); ax[1].axis('off')
plt.show()""")

md("""## 3. The frozen encoder and the latent space

We embed every frame with DINOv2 (frozen). Then we visualize all frame latents with
PCA, colored by swing type. **Question to notice:** do forehand and backhand frames
already separate before we train any dynamics? If so, vision alone carries a lot of
signal.""")
code("""encoder = load_encoder(device)
emb = {c.name: embed_clip_cached(encoder, frames[c.name], EMB, c.name) for c in clips}
from sklearn.decomposition import PCA
X = np.concatenate([emb[c.name] for c in clips])
y = np.concatenate([[c.label]*emb[c.name].shape[0] for c in clips])
P = PCA(n_components=2).fit_transform(X)
for lbl in (0,1):
    plt.scatter(P[y==lbl,0], P[y==lbl,1], s=8, label=LABEL_NAMES[lbl], alpha=.6)
plt.legend(); plt.title('frame latents (PCA)'); plt.show()""")

md("""## 4. Build the dynamics model

`SwingConditionedDynamics`: a GRU over latents whose hidden state is FiLM-modulated
by a swing-type embedding, predicting a **residual** `z_{t+1} = z_t + f(...)`.
Residual prediction keeps the output near the current latent, which is easier to
learn. Set `conditioned=False` to ignore the label (our baseline).

**Try changing X:** later, set `CONTEXT` to 2 or 6 and see how rollout error moves.""")
code("""CONTEXT, HORIZON, EPOCHS = 4, 10, 300
def build_windows(emb_d, lab_d):
    ctx,tgt,lab=[],[],[]
    for n,e in emb_d.items():
        if e.shape[0] < CONTEXT+HORIZON: continue
        for s in range(e.shape[0]-CONTEXT-HORIZON+1):
            ctx.append(e[s:s+CONTEXT]); tgt.append(e[s+CONTEXT:s+CONTEXT+HORIZON]); lab.append(lab_d[n])
    return np.stack(ctx).astype('float32'), np.stack(tgt).astype('float32'), np.array(lab)

tr,te = train_test_split_clips(clips, seed=0)
print('test (held out):', [c.name for c in te])
tr_emb={c.name:emb[c.name] for c in tr}; tr_lab={c.name:c.label for c in tr}
ctx,tgt,lab = build_windows(tr_emb, tr_lab)
print('train windows:', ctx.shape)""")

md("""## 5. Train it (teacher forcing + free-running rollout)

We warm the GRU on the context latents, then let it run free for `HORIZON` steps and
match the true future latents (MSE + cosine). Watch the loss curve drop.""")
code("""def train(model):
    opt=torch.optim.Adam(model.parameters(), lr=1e-3); losses=[]
    C=torch.from_numpy(ctx).to(device); T=torch.from_numpy(tgt).to(device); L=torch.from_numpy(lab).to(device)
    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad(); pred=model.rollout(C,L,HORIZON)
        loss=torch.nn.functional.mse_loss(pred,T)+(1-torch.nn.functional.cosine_similarity(pred,T,dim=-1).mean())
        loss.backward(); opt.step(); losses.append(loss.item())
    return model, losses
cond,lc = train(SwingConditionedDynamics().to(device))
uncond,lu = train(SwingConditionedDynamics(conditioned=False).to(device))
plt.plot(lc,label='conditioned'); plt.plot(lu,label='unconditioned'); plt.legend(); plt.title('training loss'); plt.show()""")

md("## 6. Roll out and evaluate on held-out swings\\nCompare conditioned vs unconditioned vs wrong-label vs persistence.")
code("""te_emb={c.name:emb[c.name] for c in te}; te_lab={c.name:c.label for c in te}
tc,tt,tl = build_windows(te_emb, te_lab)
TC=torch.from_numpy(tc).to(device); TL=torch.from_numpy(tl).to(device)
with torch.no_grad():
    pc=cond.rollout(TC,TL,HORIZON).cpu().numpy()
    pu=uncond.rollout(TC,TL,HORIZON).cpu().numpy()
    pw=cond.rollout(TC,1-TL,HORIZON).cpu().numpy()
pp=persistence_rollout(tc,HORIZON)
hh=np.arange(1,HORIZON+1)
for name,p in [('conditioned',pc),('unconditioned',pu),('wrong label',pw),('persistence',pp)]:
    plt.plot(hh, rollout_error(p,tt), '-o', label=name)
plt.xlabel('horizon'); plt.ylabel('cosine error'); plt.legend(); plt.title('rollout error'); plt.show()""")

md("""## 7. The counterfactual (the payoff)

Take a forehand context, then roll out twice: once told `forehand`, once told
`backhand`. If the model learned intent-conditioned dynamics, the two imagined
futures should diverge. We watch them via nearest-neighbor retrieval (predicted
latent -> closest real frame).""")
code("""bank = ReferenceBank(np.concatenate([emb[c.name] for c in clips]),
                     np.concatenate([frames[c.name] for c in clips]))
fhc = next(c for c in te if c.label==0)
c0 = torch.from_numpy(emb[fhc.name][None,:CONTEXT]).to(device)
with torch.no_grad():
    rf=cond.rollout(c0,torch.tensor([0],device=device),HORIZON).cpu().numpy()[0]
    rb=cond.rollout(c0,torch.tensor([1],device=device),HORIZON).cpu().numpy()[0]
ff,fb = bank.retrieve_frames(rf), bank.retrieve_frames(rb)
n=5; fig,ax=plt.subplots(2,n,figsize=(2*n,4))
for i in range(n):
    ax[0,i].imshow(ff[i]); ax[0,i].axis('off'); ax[1,i].imshow(fb[i]); ax[1,i].axis('off')
ax[0,0].set_title('told: forehand'); ax[1,0].set_title('told: backhand'); plt.show()""")

md("""## 8. Reflection

- **What worked:** dynamics in a frozen latent space are learnable from tiny data;
  conditioning on intent changes the imagined rollout (counterfactual).
- **Limits:** ~490 handheld frames means heavy overfitting risk and camera motion
  is entangled with swing motion. Results are relative + qualitative, not photoreal.
- **Thesis directions:** scale data, add a real action signal (pose), decode latents
  to pixels, and place this world model inside a VLA control loop (P1 idea on real
  video).""")

nb.cells = c
nbf.write(nb, str(Path(__file__).parent/'p5_walkthrough.ipynb'))
print('wrote p5_walkthrough.ipynb')
```

- [ ] **Step 2: Generate the notebook, then remove the builder**

Run:
```bash
uv run python p5_tennis_world_model/_build_nb.py
rm p5_tennis_world_model/_build_nb.py
```
Expected: prints `wrote p5_walkthrough.ipynb`.

- [ ] **Step 3: Execute the notebook end to end to confirm it runs**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace p5_tennis_world_model/p5_walkthrough.ipynb`
(If `jupyter` is not installed: `uv pip install jupyter` first.)
Expected: completes without error; all cells get outputs.

- [ ] **Step 4: Commit**

```bash
git add p5_tennis_world_model/p5_walkthrough.ipynb
git commit -m "P5: annotated teaching notebook walkthrough"
```

---

## Task 9: README and run_all wiring

**Files:**
- Create: `p5_tennis_world_model/README.md`
- Modify: `scripts/run_all.sh`

- [ ] **Step 1: Write the README**

Create `p5_tennis_world_model/README.md` (no em dashes). Structure: Intro, Related Work, Method, Results, Limitations, Thesis Directions. Use the actual numbers printed by `run_tennis_wm.py`. Template body:

```markdown
# P5: A Latent World Model of Tennis Swings

## Intro
World models learn dynamics: predict the next state from the current state and an
action. This project learns swing dynamics from 11 real handheld phone clips (5
forehand, 6 backhand) by predicting the next *frame latent* conditioned on swing
type. It moves the latent-world-model idea from simulation (P1) to real video.

## Related Work
DINOv2 self-supervised features (frozen encoder). FiLM conditioning. Latent-space
world models and action-conditioned video prediction (P2). Citations from memory,
verify before presenting.

## Method
Frozen DINOv2 ViT-S/14 maps each frame to a 384-d latent (cached once). A small
FiLM-conditioned GRU predicts residual latent transitions, warmed on a context
window then rolled free for H frames. Trained on 9 clips, tested on 2 held-out
clips (one per class). Predicted latents are visualized by nearest-neighbor
retrieval against a bank of real frames, so no pixel decoder is trained.

## Results
Rollout cosine error vs horizon on held-out swings (figure
`results/p5_rollout_error.png`): conditioned <NUM>, unconditioned <NUM>,
wrong-label <NUM>, persistence <NUM>. Counterfactual
(`results/p5_counterfactual.png`, `results/p5_dream.gif`): the same forehand
context produces divergent imagined rollouts when told forehand vs backhand.

## Limitations
About 490 frames total is very small; overfitting is expected. Handheld camera
couples camera motion with swing motion. Results are relative to baselines and
qualitative, not photoreal video generation.

## Thesis Directions
Scale data, add a real action signal (body/racket pose), decode latents to pixels,
and embed this world model in a VLA control loop on real video.
```

Fill `<NUM>` placeholders with the printed error values (use the mean over horizon, or the value at H=10).

- [ ] **Step 2: Wire into run_all.sh**

Read `scripts/run_all.sh`, then add a P5 line following the existing per-project pattern (match how P1-P4 are invoked). It should run:

```bash
uv run python p5_tennis_world_model/run_tennis_wm.py
```
Place it after the P4 invocation. Match surrounding echo/section style exactly.

- [ ] **Step 3: Verify run_all still parses**

Run: `bash -n scripts/run_all.sh`
Expected: no output (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add p5_tennis_world_model/README.md scripts/run_all.sh
git commit -m "P5: README write-up and run_all wiring"
```

---

## Task 10: Top-level integration and push

**Files:**
- Modify: `README.md` (repo root)

- [ ] **Step 1: Add P5 to the root README**

Read root `README.md`, find where P1-P4 are summarized, and add a P5 entry in the
same format. One sentence: latent world model of real tennis swings, intent-
conditioned, sim (P1) to real (P5). Update any "four projects" / "P1-P4" counts to
include P5.

- [ ] **Step 2: Full test sweep**

Run: `uv run pytest tests/ -v`
Expected: all pass.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "P5: add to portfolio README (P1-P5)"
git push
```
Expected: pushes to `origin/main`. (Author remains Silvio Kempf; do not add any co-author trailer.)

---

## Notes for the implementer

- Author all commits as the repo user; do NOT add a `Co-Authored-By` trailer.
- Raw `.MOV` clips and `embeddings/` are gitignored on purpose. Do not force-add them.
- `results/*.png` and `*.gif` ARE tracked (so READMEs render on GitHub) but `*.pt`/`*.npz` are not.
- DINOv2 downloads weights from torch.hub on first run; needs network once.
- If `imread(..., plugin="pyav")` fails, ensure `av` is available (`uv pip install av`) or fall back to `plugin="FFMPEG"`.
- Everything targets Mac MPS; keep batch sizes small (24GB).
```
