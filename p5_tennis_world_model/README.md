# P5: Anticipating a Tennis Swing from the Ball Approach

A tiny real-video world model built from 11 handheld phone clips (5 forehand, 6
backhand). The key result is that the swing side is strongly decodable from the
opening frames, when the ball is approaching and the player is preparing, even
though the signal is much weaker when all frames are averaged together.

The generative side then trains a latent GRU world model and "dreams" a swing
continuation from the opening frames by rolling forward in DINOv2 latent space and
retrieving the nearest real frames.

The walkthrough is in `p5_walkthrough.ipynb`. Reproduce the figures with
`python run_tennis_wm.py` (CPU, a few minutes).

## Intro
A world model learns dynamics: predict the next state from the current state, and
optionally an action. Here the state is a frozen DINOv2 latent of a video frame,
the dynamics is a small GRU, and the practical question is whether the future
swing can be anticipated from the ball approach and preparation pose.

## Method
1. Frozen DINOv2 ViT-S/14 maps each frame to a 384-d latent (cached once).
2. Drop corrupt frames (motion-blur or near-black transition frames whose latent
   is an outlier), then standardize the latents per dimension.
3. Mirror-flip each clip to double the data and swap the left/right side label.
4. Run a leave-one-clip-out linear probe on the side label: full clip versus only
   the first 8 opening frames.
5. Train an unconditioned GRU world model on standardized latents and evaluate
   leave-one-clip-out against persistence (predict no change).
6. Visualize a rollout by nearest-neighbor retrieval against a bank of real
   frames, so no pixel decoder is trained.

## Results
Held-out side accuracy, chance = 0.5 (`results/p5_anticipation.png`):

| setup | accuracy |
| --- | ---: |
| full clip, no flip | 0.616 |
| full clip, + flip | 0.582 |
| opening frames, no flip | 0.943 |
| opening frames, + flip | 0.989 |

This flips the earlier conclusion: the swing is predictable, but the signal lives
at the start of the clip. The ball approach and preparation pose carry the
forehand/right versus backhand/left information; mid-swing frames partially wash
that signal out.

Leave-one-clip-out rollout error (`results/p5_rollout_error.png`):

| horizon | 1 | 5 | 12 |
| --- | --- | --- | --- |
| world model, flip-augmented | 0.025 | 0.056 | 0.069 |
| persistence | 0.028 | 0.074 | 0.096 |

The unconditioned world model beats persistence at every horizon: it learned to
extrapolate the swing trajectory instead of freezing the frame. Standardizing the
latents was the key fix, since DINOv2 dimensions have very different scales and raw
MSE was dominated by a few of them.

The dream demo (`results/p5_dream.png`, `results/p5_dream_forehand.gif`,
`results/p5_dream_backhand.gif`) seeds the model with opening frames and rolls out
a predicted continuation. The frames are retrieved nearest neighbors from the real
frame bank, so the demo shows the model's latent prediction without pretending to
have a photoreal pixel decoder.

## Limitations
About 490 frames total is very small. The clips were handheld in one session, so
part of the opening-frame side signal could come from framing or camera motion,
not only the ball and body pose. Results are relative to baselines and qualitative,
not photoreal video.

## Future Directions
Collect multi-session clips with controlled framing, add explicit ball/pose state,
decode latents to pixels, and place this anticipation model inside a VLA control
loop on real video, extending P1 from simulation to the real world.

## Related Work
DINOv2 self-supervised features (frozen encoder), FiLM conditioning, latent-space
world models, and action-conditioned video prediction (P2). Citations from memory,
verify before presenting.
