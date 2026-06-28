"""P5 end-to-end: a latent world model of tennis swings.

Pipeline: embed clips with frozen DINOv2 -> clean corrupt frames -> standardize
latents -> train a GRU dynamics model -> evaluate with leave-one-clip-out CV
against persistence -> diagnose why swing-type conditioning fails -> visualize an
unconditioned "dream" via nearest-neighbor retrieval.

Key findings reproduced here:
  - Unconditioned latent world model BEATS persistence at every horizon.
  - Conditioning on swing type HURTS (label not generalizable at 11 clips); a
    linear probe decodes swing type at ~chance on held-out clips.

Produces: results/p5_rollout_error.png, results/p5_representation.png,
          results/p5_dream.gif. Runs on CPU in a few minutes.
"""
from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from vlawm.tennis.data import discover_clips, decode_clip, LABEL_NAMES
from vlawm.tennis.encoder import load_encoder, embed_clip_cached
from vlawm.tennis.prep import clean_mask, Standardizer
from vlawm.tennis.evaluate import persistence_rollout, rollout_error
from vlawm.tennis.retrieval import ReferenceBank
from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
EMB = ROOT / "data" / "embeddings"
RESULTS = ROOT.parent / "results"
CONTEXT, HORIZON, EPOCHS, WD, SEED = 4, 10, 150, 1e-3, 0
DEVICE = torch.device("cpu")  # many tiny models; CPU avoids MPS transfer overhead


def build_windows(emb_by_clip, label_by_clip):
    ctx, tgt, lab = [], [], []
    for name, e in emb_by_clip.items():
        if e.shape[0] < CONTEXT + HORIZON:
            continue
        for s in range(e.shape[0] - CONTEXT - HORIZON + 1):
            ctx.append(e[s:s + CONTEXT])
            tgt.append(e[s + CONTEXT:s + CONTEXT + HORIZON])
            lab.append(label_by_clip[name])
    return (np.stack(ctx).astype("float32"),
            np.stack(tgt).astype("float32"),
            np.array(lab))


def train_model(ctx, tgt, lab, conditioned):
    torch.manual_seed(SEED)
    m = SwingConditionedDynamics(conditioned=conditioned).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=WD)
    C, T, L = (torch.from_numpy(ctx).to(DEVICE),
               torch.from_numpy(tgt).to(DEVICE),
               torch.from_numpy(lab).to(DEVICE))
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m.rollout(C, L, HORIZON), T)
        loss.backward()
        opt.step()
    return m


def main():
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)

    clips = discover_clips(RAW)
    encoder = load_encoder(DEVICE)
    frames_raw, emb_raw = {}, {}
    for c in clips:
        fr = decode_clip(c)
        raw_emb = embed_clip_cached(encoder, fr, EMB, c.name)
        keep = clean_mask(raw_emb)  # drop corrupt frames; keep latents+frames aligned
        emb_raw[c.name] = raw_emb[keep]
        frames_raw[c.name] = fr[keep]
    label = {c.name: c.label for c in clips}
    names = [c.name for c in clips]

    # ---- leave-one-clip-out CV: unconditioned vs conditioned vs persistence ----
    e_uncond, e_cond, e_persist = [], [], []
    for held in names:
        tr = [n for n in names if n != held]
        if emb_raw[held].shape[0] < CONTEXT + HORIZON:
            continue
        std = Standardizer.fit(np.concatenate([emb_raw[n] for n in tr]))
        tr_emb = {n: std.transform(emb_raw[n]) for n in tr}
        te_emb = {held: std.transform(emb_raw[held])}
        ctx, tgt, lab = build_windows(tr_emb, label)
        tc, tt, tl = build_windows(te_emb, label)

        mu = train_model(ctx, tgt, lab, conditioned=False)
        mc = train_model(ctx, tgt, lab, conditioned=True)
        with torch.no_grad():
            TC, TL = torch.from_numpy(tc).to(DEVICE), torch.from_numpy(tl).to(DEVICE)
            pu = mu.rollout(TC, TL, HORIZON).cpu().numpy()
            pc = mc.rollout(TC, TL, HORIZON).cpu().numpy()
        pp = persistence_rollout(tc, HORIZON)
        # error measured in original latent space (de-standardize predictions)
        e_uncond.append(rollout_error(std.inverse(pu), std.inverse(tt)))
        e_cond.append(rollout_error(std.inverse(pc), std.inverse(tt)))
        e_persist.append(rollout_error(std.inverse(pp), std.inverse(tt)))

    e_uncond = np.mean(e_uncond, 0)
    e_cond = np.mean(e_cond, 0)
    e_persist = np.mean(e_persist, 0)

    h = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(h, e_uncond, "-o", label="world model (unconditioned)")
    plt.plot(h, e_persist, "--", label="persistence (do nothing)")
    plt.plot(h, e_cond, "-^", label="world model (swing-conditioned)")
    plt.xlabel("rollout horizon (frames)")
    plt.ylabel("cosine error")
    plt.title("P5: latent rollout error (leave-one-clip-out CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "p5_rollout_error.png", dpi=120)
    plt.close()

    # ---- representation diagnosis: PCA + held-out swing-type decodability ----
    all_emb = np.concatenate([emb_raw[n] for n in names])
    all_lab = np.concatenate([[label[n]] * emb_raw[n].shape[0] for n in names])
    proj = PCA(n_components=2).fit_transform(all_emb)

    probe_tr, probe_te = [], []
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([emb_raw[n] for n in tr])
        ytr = np.concatenate([[label[n]] * emb_raw[n].shape[0] for n in tr])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        probe_tr.append(clf.score(Xtr, ytr))
        probe_te.append(clf.score(emb_raw[held], [label[held]] * emb_raw[held].shape[0]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lbl in (0, 1):
        ax[0].scatter(proj[all_lab == lbl, 0], proj[all_lab == lbl, 1],
                      s=10, alpha=.6, label=LABEL_NAMES[lbl])
    ax[0].legend(); ax[0].set_title("Frame latents (DINOv2 + PCA)")
    ax[1].bar(["train", "held-out"], [np.mean(probe_tr), np.mean(probe_te)],
              color=["#4c72b0", "#c44e52"])
    ax[1].axhline(0.5, ls="--", color="k", label="chance")
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("swing-type linear-probe accuracy")
    ax[1].set_title("Swing type only weakly decodable on held-out clips")
    ax[1].legend()
    fig.suptitle("P5 diagnosis: weak/unreliable swing-type signal in the representation")
    plt.tight_layout()
    plt.savefig(RESULTS / "p5_representation.png", dpi=120)
    plt.close()

    # ---- dream: unconditioned rollout on one clip, visualized by retrieval ----
    bank = ReferenceBank(all_emb, np.concatenate([frames_raw[n] for n in names]))
    demo = names[0]
    std = Standardizer.fit(np.concatenate([emb_raw[n] for n in names if n != demo]))
    ctx0 = torch.from_numpy(std.transform(emb_raw[demo])[None, :CONTEXT]).float().to(DEVICE)
    mu = train_model(*build_windows({n: std.transform(emb_raw[n]) for n in names if n != demo}, label),
                     conditioned=False)
    with torch.no_grad():
        roll = std.inverse(mu.rollout(ctx0, torch.tensor([label[demo]]).to(DEVICE), HORIZON).cpu().numpy()[0])
    imageio.mimsave(RESULTS / "p5_dream.gif", list(bank.retrieve_frames(roll)), fps=6, loop=0)

    print("LOO-CV cosine error vs horizon", list(h))
    print("world model (uncond):", np.round(e_uncond, 3))
    print("persistence         :", np.round(e_persist, 3))
    print("world model (cond)  :", np.round(e_cond, 3))
    print("swing probe acc: train=%.2f held-out=%.2f (chance 0.5)"
          % (np.mean(probe_tr), np.mean(probe_te)))
    print("wrote figures to", RESULTS)


if __name__ == "__main__":
    main()
