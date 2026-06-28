# P5: Latent World Model of Tennis Swings (design)

Date: 2026-06-28
Status: approved design, pre-implementation

## One-line

Learn intent-conditioned dynamics from real, handheld tennis clips by predicting
the next frame in a frozen-encoder latent space, conditioned on swing type
(forehand / backhand), and visualize predictions via nearest-neighbor retrieval.

## Goals

- Produce a real result that slots into the portfolio as P5: "latent world models,
  sim (P1) to real (P5)."
- Be a deep, hands-on learning vehicle for world models / VLAs. The primary
  artifact is an annotated, runnable notebook the user follows cell by cell.
- Leave behind a small reusable pipeline (package code + run script) consistent
  with P1-P4.

## Non-goals

- Photoreal video generation. The data is tiny and handheld; sharp pixel
  synthesis is out of scope.
- Beating any published video-prediction benchmark. The contribution is the
  method and the conditioning (counterfactual) result, evaluated relative to
  baselines, plus qualitative retrieval demos.

## Data

- Source: 11 user-recorded clips in `p5_tennis_world_model/data/raw/`.
  5 forehand, 6 backhand. All 720x1280 portrait, 30 fps, ~1.0-1.5 s each.
  Handheld camera (camera motion is entangled with scene motion).
- Label source: filename prefix (`forehand_XX.MOV`, `backhand_XX.MOV`). No CSV
  needed. A `labels.csv.example` template exists for the alternate path.
- Decode: keep native 30 fps to maximize transitions (~490 frames total; at 10
  fps there would be too few). Resize / center-crop to the encoder input size.
- Split: by clip, never by frame (frames from one swing must not leak across
  train/test). Hold out 1 forehand + 1 backhand as the test set; train on the
  other 9. Optionally rotate held-out clips later for a more robust number.

## Architecture

Three isolated units, each independently understandable and testable.

### 1. Data pipeline (`src/vlawm/data` additions or `p5_*/data.py`)
- Input: clips in `data/raw/`. Output: per-clip ordered frame tensors + label.
- Responsibilities: decode video, resize/crop, parse label from filename,
  produce clip-level train/test split.

### 2. Frozen encoder + embedding cache
- DINOv2 ViT-S/14 via `torch.hub`, frozen, on MPS. Each frame -> 384-d latent
  (CLS token).
- Embed every frame once, cache to `embeddings/*.npy`. Training never touches
  pixels or the encoder again. Vision is "solved for free"; the world model only
  learns dynamics.

### 3. Latent dynamics model (`src/vlawm/worldmodel/` addition)
- Input: a context of C past latents + a swing-type embedding. Output: predicted
  next latent. Rolled autoregressively for multi-step prediction.
- Small GRU (tiny transformer as alternative). This is the only trained
  component.
- Conditioning via FiLM, reusing the exact mechanism from the existing VLA, for a
  consistent repo story.
- Loss in latent space (cosine + MSE), teacher-forced training, evaluated on
  free-running rollout.

### Visualization (no decoder)
- Reference bank: all real frames + their latents. Map each predicted latent to
  its nearest real frame -> stitch into a predicted "video" GIF.
- Avoids training a pixel decoder entirely.

## Experiments

- Baselines: (a) persistence (copy last latent), (b) unconditioned dynamics (no
  swing label), (c) wrong-label model.
- Primary metric: rollout latent error (cosine / MSE) vs horizon, on held-out
  clips.
- Money result, the counterfactual: feed forehand context but condition on
  backhand; show the rollout diverges toward backhand-like dynamics. Evidence the
  model learned intent-conditioned dynamics, not the average swing. Mirrors P2's
  action-conditioning story on real video.
- Expected honest finding: conditioned < unconditioned < persistence in error;
  label-swap visibly changes the predicted trajectory.

## Deliverables

- Primary artifact: `p5_tennis_world_model/p5_walkthrough.ipynb`, an annotated,
  cell-by-cell teaching notebook. Notebook imports package code; stable logic does
  not live in cells.
- Notebook sections (each teaches a concept while doing it):
  1. What is a world model? (`s_t, a_t -> s_{t+1}`, why latent not pixel, swing
     label = action; connect to P1/P2)
  2. Load and peek at data (play a clip inline; FH vs BH frames)
  3. Frozen encoder + visualize latent space (PCA/t-SNE colored by FH/BH; see if
     vision already separates swings)
  4. Build the dynamics model (FiLM-conditioned GRU, explained line by line)
  5. Train (live loss curve; teacher forcing explained)
  6. Roll out and evaluate (error vs horizon vs baselines, inline)
  7. Counterfactual (FH context, swap label, watch retrieval change)
  8. Reflection (limits of tiny data, thesis directions)
  - Include occasional "try changing X" prompts (encoder layer, context length)
    for active experimentation.
- Non-interactive path: `p5_tennis_world_model/run_tennis_wm.py` so P5 fits
  `scripts/run_all.sh`.
- Figures: `results/p5_rollout_error.png`, `results/p5_counterfactual.png`,
  `results/p5_dream.gif`.
- `p5_tennis_world_model/README.md` write-up (Intro/Related Work/Method/Results),
  no em dashes, stating data-size and handheld caveats plainly.

## Honest caveats (to state in README)

- ~490 frames is very small; overfitting / memorization is expected.
- Handheld camera couples camera motion with scene motion.
- Results are relative to baselines + qualitative, not SOTA video generation.

## Thesis hook

Latent world models learn controllable dynamics from real, uncurated video:
sim (P1) -> real (P5). Conditioning on intent (swing type) echoes the
action-conditioned dreamer (P2).
