"""P5: a world model that anticipates a tennis swing from the ball's approach.

Story (all from 11 real handheld clips, 5 forehand + 6 backhand):
  1. Anticipation: from the OPENING frames (ball approaching + preparation pose)
     a linear probe predicts the swing side at ~0.95 on held-out clips, even though
     it is only ~0.6 over the full clip (mid-swing forehand and backhand look alike).
     Horizontal mirror-flip augmentation doubles and balances the data and helps.
  2. Dynamics: a small GRU predicts the next frame latent. Rolled out it beats a
     persistence ("predict no change") baseline, so it has learned the motion.
  3. Dreaming: seeded with a swing's opening frames the model rolls forward, and
     nearest-neighbor retrieval turns the predicted latents into a watchable
     continuation that matches the seeded swing.

Frames are encoded once with a frozen DINOv2 (global CLS token). Patch tokens were
tried and did not help at this data scale.

Produces: results/p5_anticipation.png, results/p5_rollout_error.png,
          results/p5_dream.png, results/p5_dream_forehand.gif,
          results/p5_dream_backhand.gif. Runs on CPU in a few minutes.
"""
from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from vlawm.tennis.data import discover_clips, decode_clip
from vlawm.tennis.encoder import load_encoder, embed_frames
from vlawm.tennis.prep import clean_mask, Standardizer
from vlawm.tennis.evaluate import persistence_rollout, rollout_error
from vlawm.tennis.retrieval import ReferenceBank
from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
EMB = ROOT / "data" / "embeddings"
RESULTS = ROOT.parent / "results"
CONTEXT, HORIZON, EPOCHS, WD, SEED = 4, 12, 200, 1e-3, 0
OPENING = 8                      # frames at the start that carry the "which swing" signal
DEVICE = torch.device("cpu")
# side: forehand -> ball on the right (1), backhand -> ball on the left (0).
# A horizontal flip mirrors the image, so it swaps the side.


def cached(encoder, frames, name):
    EMB.mkdir(parents=True, exist_ok=True)
    p = EMB / f"{name}.npy"
    if p.exists():
        return np.load(p)
    e = embed_frames(encoder, frames, device=DEVICE)
    np.save(p, e)
    return e


def load_all():
    """Return per clip: original + flipped frames and latents (corrupt frames dropped)."""
    clips = discover_clips(RAW)
    encoder = load_encoder(DEVICE)
    data = {}
    for c in clips:
        fr = decode_clip(c)
        frf = fr[:, :, ::-1, :].copy()               # horizontal mirror
        eo = cached(encoder, fr, Path(c.name).stem)
        ef = cached(encoder, frf, Path(c.name).stem + "_flip")
        ko, kf = clean_mask(eo), clean_mask(ef)
        data[c.name] = {
            "side": 1 - c.label,                     # forehand(0)->1, backhand(1)->0
            "fr": fr[ko], "emb": eo[ko],
            "fr_f": frf[kf], "emb_f": ef[kf],
        }
    return data


def build_windows(seqs_labels):
    """seqs_labels: list of (latents (T,D), side). Returns context, target, side arrays."""
    ctx, tgt, sd = [], [], []
    for e, side in seqs_labels:
        if e.shape[0] < CONTEXT + HORIZON:
            continue
        for s in range(e.shape[0] - CONTEXT - HORIZON + 1):
            ctx.append(e[s:s + CONTEXT])
            tgt.append(e[s + CONTEXT:s + CONTEXT + HORIZON])
            sd.append(side)
    return np.stack(ctx).astype("float32"), np.stack(tgt).astype("float32"), np.array(sd)


def train_dynamics(seqs):
    torch.manual_seed(SEED)
    ctx, tgt, _ = build_windows(seqs)
    m = SwingConditionedDynamics(conditioned=False).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=WD)
    C, T = torch.from_numpy(ctx).to(DEVICE), torch.from_numpy(tgt).to(DEVICE)
    L = torch.zeros(len(ctx), dtype=torch.long, device=DEVICE)
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m.rollout(C, L, HORIZON), T).backward()
        opt.step()
    return m


# --------------------------------------------------------------------------- #
def anticipation(data):
    """Held-out side accuracy: opening vs full clip, with and without flip aug."""
    names = list(data)

    def probe(opening_only, use_flip):
        accs = []
        for held in names:
            X, y = [], []
            for n in names:
                if n == held:
                    continue
                eo = data[n]["emb"][:OPENING] if opening_only else data[n]["emb"]
                X.append(eo); y += [data[n]["side"]] * len(eo)
                if use_flip:
                    ef = data[n]["emb_f"][:OPENING] if opening_only else data[n]["emb_f"]
                    X.append(ef); y += [1 - data[n]["side"]] * len(ef)
            clf = LogisticRegression(max_iter=2000).fit(np.concatenate(X), np.array(y))
            eh = data[held]["emb"][:OPENING] if opening_only else data[held]["emb"]
            accs.append(clf.score(eh, [data[held]["side"]] * len(eh)))
        return float(np.mean(accs))

    res = {
        "full / no flip": probe(False, False),
        "full / + flip": probe(False, True),
        "opening / no flip": probe(True, False),
        "opening / + flip": probe(True, True),
    }
    plt.figure(figsize=(6.5, 4))
    colors = ["#bdbdbd", "#8c8c8c", "#4c72b0", "#2f4b7c"]
    plt.bar(res.keys(), res.values(), color=colors)
    plt.axhline(0.5, ls="--", color="k", label="chance")
    plt.ylim(0, 1); plt.ylabel("held-out swing-side accuracy")
    plt.title("P5: the swing is anticipated from the OPENING frames")
    plt.xticks(rotation=15); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "p5_anticipation.png", dpi=120)
    plt.close()
    return res


def rollout_vs_persistence(data):
    """Leave-one-clip-out: flip-augmented world model vs persistence baseline."""
    names = list(data)
    e_wm, e_pe = [], []
    for held in names:
        train_seqs = []
        for n in names:
            if n == held:
                continue
            train_seqs.append((data[n]["emb"], data[n]["side"]))
            train_seqs.append((data[n]["emb_f"], 1 - data[n]["side"]))
        std = Standardizer.fit(np.concatenate([s for s, _ in train_seqs]))
        m = train_dynamics([(std.transform(s), lab) for s, lab in train_seqs])
        if data[held]["emb"].shape[0] < CONTEXT + HORIZON:
            continue
        tc, tt, _ = build_windows([(std.transform(data[held]["emb"]), data[held]["side"])])
        with torch.no_grad():
            L = torch.zeros(len(tc), dtype=torch.long)
            pw = m.rollout(torch.from_numpy(tc), L, HORIZON).numpy()
        pp = persistence_rollout(tc, HORIZON)
        e_wm.append(rollout_error(std.inverse(pw), std.inverse(tt)))
        e_pe.append(rollout_error(std.inverse(pp), std.inverse(tt)))
    e_wm, e_pe = np.mean(e_wm, 0), np.mean(e_pe, 0)
    h = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(h, e_wm, "-o", label="world model (flip-augmented)")
    plt.plot(h, e_pe, "--", label="persistence (do nothing)")
    plt.xlabel("rollout horizon (frames)"); plt.ylabel("cosine error")
    plt.title("P5: world model beats persistence (leave-one-clip-out)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "p5_rollout_error.png", dpi=120)
    plt.close()
    return e_wm, e_pe


def dream(data):
    """Seed the model with a swing's opening frames and dream the continuation."""
    names = list(data)
    fh = next(n for n in names if data[n]["side"] == 1)
    bh = next(n for n in names if data[n]["side"] == 0)
    train_seqs = []
    for n in names:
        if n in (fh, bh):
            continue
        train_seqs.append((data[n]["emb"], data[n]["side"]))
        train_seqs.append((data[n]["emb_f"], 1 - data[n]["side"]))
    std = Standardizer.fit(np.concatenate([s for s, _ in train_seqs]))
    m = train_dynamics([(std.transform(s), lab) for s, lab in train_seqs])

    # retrieval bank: all real frames (originals + flips) and their latents
    bank = ReferenceBank(
        np.concatenate([data[n]["emb"] for n in names] + [data[n]["emb_f"] for n in names]),
        np.concatenate([data[n]["fr"] for n in names] + [data[n]["fr_f"] for n in names]),
    )

    strips = {}
    for tag, clip in (("forehand", fh), ("backhand", bh)):
        seed = std.transform(data[clip]["emb"])[None, :CONTEXT]
        with torch.no_grad():
            roll = m.rollout(torch.from_numpy(seed).float(),
                             torch.zeros(1, dtype=torch.long), HORIZON).numpy()[0]
        frames = bank.retrieve_frames(std.inverse(roll))
        strips[tag] = frames
        imageio.mimsave(RESULTS / f"p5_dream_{tag}.gif", list(frames), fps=6, loop=0)

    n = min(6, HORIZON)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.2))
    for r, tag in enumerate(("forehand", "backhand")):
        for i in range(n):
            axes[r, i].imshow(strips[tag][i]); axes[r, i].axis("off")
        axes[r, 0].set_title(f"dreamed from {tag} opening", loc="left", fontsize=9)
    fig.suptitle("P5: dreamed swing continuation (seeded with the opening frames)")
    plt.tight_layout()
    plt.savefig(RESULTS / "p5_dream.png", dpi=120)
    plt.close()


def main():
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    data = load_all()
    ant = anticipation(data)
    e_wm, e_pe = rollout_vs_persistence(data)
    dream(data)
    print("anticipation (held-out side accuracy):")
    for k, v in ant.items():
        print(f"  {k:18s}: {v:.3f}")
    print("rollout cosine error @ h=12: world model=%.3f persistence=%.3f"
          % (e_wm[-1], e_pe[-1]))
    print("wrote figures to", RESULTS)


if __name__ == "__main__":
    main()
