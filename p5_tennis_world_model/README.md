# P5: A Latent World Model of Tennis Swings

A world model that learns swing dynamics from 11 real handheld phone clips (5
forehand, 6 backhand) by predicting the next *frame latent* rather than pixels.
It moves the latent-world-model idea from simulation (P1) to real video, and it
turned into a small, honest study of when conditioning works and when it does not.

The full build, including the failure and the fix, is in
`p5_walkthrough.ipynb`. Reproduce the figures with
`python run_tennis_wm.py` (CPU, a few minutes).

## Intro
A world model learns dynamics: predict the next state from the current state, and
optionally an action. Here the state is a frozen DINOv2 latent of a video frame,
the dynamics is a small GRU, and the candidate action is the swing type
(forehand or backhand).

## Method
1. Frozen DINOv2 ViT-S/14 maps each frame to a 384-d latent (cached once).
2. Drop corrupt frames (motion-blur or near-black transition frames whose latent
   is an outlier), then standardize the latents per dimension.
3. A FiLM-conditioned GRU predicts residual latent transitions, warmed on a short
   context window then rolled free for H frames.
4. Evaluate with leave-one-clip-out cross validation against a persistence
   baseline (predict no change). Visualize a rollout by nearest-neighbor retrieval
   against a bank of real frames, so no pixel decoder is trained.

## Results
Leave-one-clip-out cosine error vs horizon (`results/p5_rollout_error.png`):

| horizon | 1 | 5 | 10 |
| --- | --- | --- | --- |
| world model (unconditioned) | 0.025 | 0.057 | 0.070 |
| persistence | 0.028 | 0.076 | 0.089 |
| world model (swing-conditioned) | 0.084 | 0.458 | 0.723 |

The unconditioned world model beats persistence at every horizon: it learned to
extrapolate the swing trajectory instead of freezing the frame. Standardizing the
latents was the key fix, since DINOv2 dimensions have very different scales and raw
MSE was dominated by a few of them.

Swing-type conditioning makes the model about ten times worse. The diagnosis
(`results/p5_representation.png`): a linear probe decodes swing type at ~1.0 on
training frames but only ~0.62 on held-out clips (chance is 0.5), so the label
does not generalize. Given the label, the conditioned model predicts the average
training swing of that class and overrides the actual context.

## Limitations
About 490 frames total is very small, so overfitting is expected and the swing-type
signal is weak. The camera is handheld, so camera motion is entangled with swing
motion. Results are relative to baselines and qualitative, not photoreal video.

## Thesis Directions
Scale the data, use a motion-aware state (DINOv2 patch tokens, frame stacks, or
pose) so swing type becomes decodable, decode latents to pixels, and place this
world model inside a VLA control loop on real video, extending P1 from simulation
to the real world.

## Related Work
DINOv2 self-supervised features (frozen encoder), FiLM conditioning, latent-space
world models, and action-conditioned video prediction (P2). Citations from memory,
verify before presenting.
