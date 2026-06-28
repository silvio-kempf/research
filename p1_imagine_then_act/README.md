# P1: Imagine-then-Act

A latent **world model inside the control loop** of a Vision-Language-Action policy.

## The one-sentence idea

Train a world model that, given the current camera image + instruction, can predict *how
far the agent will be from its goal* after any sequence of actions. Then, instead of
trusting the policy blindly, **imagine many candidate action sequences, roll each one
forward in the world model, and execute the one predicted to get closest to the goal.**
That is "Imagine-then-Act" = model-predictive control (MPC) through a learned world model.

## The three pieces

### 1. The latent world model, in `src/vlawm/worldmodel/latent_wm.py`
Three sub-networks:
- `encode(image, instruction) → z₀`: a **language-conditioned CNN** compresses the current
  frame into a 128-d latent `z₀`. Language-conditioned so the latent "knows" *which* object
  is the target.
- `step(z, a) → z'`: a small MLP predicting the next latent from the current latent and an
  action. This is the **dynamics**, learned entirely in latent space (no pixels).
- `predict_distance(z) → d`: a head reading off the agent's distance-to-target from a latent.

Chaining `step` over an action sequence (`rollout`) lets the model imagine a whole
trajectory and predict the distance-to-goal at every step, **without touching the real
environment.** It is trained to match the true distance over H-step rollouts
(`scripts/train_world_model.py`), reaching distance-MSE ≈ 5e-5.

### 2. The planner (MPC), in `src/vlawm/worldmodel/mpc.py` (`WorldModelMPC`)
At each real timestep:
1. Sample **K candidate action sequences**, each H steps long (e.g. K=32).
2. Encode the current real frame → `z₀`.
3. **Imagine** each candidate's rollout in the world model → predicted distances over H steps.
4. **Score** each candidate (lower predicted distance = better).
5. Execute only the **first action** of the winning candidate in the real env.
6. Re-plan next timestep (receding horizon).

Key knobs (constructor args):
- `n_candidates` (K): how much it imagines.
- `policy=` / `policy_weight=`: sample candidates around the VLA's action instead of around
  zero → the "VLA + world model in the loop" configuration.
- `sigma`: exploration spread of the candidate sampler.
- `horizon` (H): how many steps ahead each candidate looks.

### 3. The experiment, `run_planner.py`
Runs two comparisons and writes the figure:
- **(A)** random policy vs. world-model MPC (no policy) vs. policy-guided MPC vs. VLA-only.
- **(B)** success rate as K (number of imagined candidates) grows.

## How to run

The planner needs two trained checkpoints: the VLA policy (`results/policy.pt`) and the
latent world model (`results/world_model.pt`).

**If the checkpoints already exist**, run the planner directly (≈1 min):
```bash
uv run python p1_imagine_then_act/run_planner.py     # → results/p1_imagine_then_act.png
```

**To rebuild from scratch** (after changing the env or models):
```bash
uv run python scripts/train_policy.py --episodes 2500 --epochs 60    # → results/policy.pt + dataset.npz
uv run python scripts/train_world_model.py --epochs 40 --horizon 5   # → results/world_model.pt (+ exploratory wm_dataset.npz)
uv run python p1_imagine_then_act/run_planner.py                     # → the figure
```

> The world-model dataset is generated with `explore_eps` > 0 (random actions mixed into the
> expert demos). This is essential, see the model-exploitation note below.

## Result

![P1](../results/p1_imagine_then_act.png)

**Planning *through* the learned world model controls the agent.** With **no policy at all**,
random-shooting MPC reaches **0.99** success (vs **0.14** random), and success rises
monotonically with the number of imagined candidates K (0.19 → 1.00). Combined with the VLA
(policy-guided MPC) it matches the 100% ceiling. The headline is that there is *no trained
controller* in the world-model-MPC condition, the agent succeeds purely by imagining action
sequences and selecting good ones with the learned model, proving the model captured the
dynamics well enough to control through.

## The model-exploitation lesson (worth telling)

The first attempt failed: a world model trained **only on expert demonstrations** is
inaccurate *off* the expert manifold, and the planner **exploited those errors**, selecting
candidates the model wrongly believed reached the goal, so *more* candidates made it *worse*.
Fixing it by training the world model on **exploratory data** (random actions mixed in) is a
core, non-obvious model-based-RL lesson: a model used for planning must be accurate where the
planner will probe it, not just on the demonstration distribution.

Code: `src/vlawm/worldmodel/{latent_wm,mpc}.py` · Training: `scripts/train_world_model.py`

---

# Write-up

A structured account of this study in the format of a short research note, mapping the same
content onto Introduction → Related Work → Method → Experiments → Results → Discussion →
Limitations → Future Work.

## 1. Introduction

Vision-Language-Action (VLA) policies map an image and a language instruction directly to
actions. Trained by behavior cloning, they are **reactive**: each action is a feed-forward
response to the current observation, with no explicit reasoning about *consequences*. When
the policy is uncertain or out-of-distribution it still acts, often confidently and wrongly.

A **world model**, a learned predictor of future states given actions, offers a
complementary capability: it can *imagine* the outcome of a candidate action before it is
taken. This study asks a focused question:

> **Can a learned world model, placed inside the control loop, select actions that solve an
> instruction-conditioned reaching task, even with no trained policy at all?**

We answer affirmatively on a controlled testbed: model-predictive control through a learned
latent world model reaches 0.99 success from random action proposals, and performance scales
with the amount of imagination (number of candidate rollouts).

## 2. Related work

- **Learned world models for control.** Ha & Schmidhuber's *World Models* (2018) and the
  *Dreamer* line (Hafner et al., 2019–2023) learn latent dynamics and optimise behaviour by
  imagined rollouts. *PETS* (Chua et al., 2018) showed sampling-based MPC with learned
  dynamics is strong and sample-efficient.
- **Sampling-based planning.** Random shooting, the Cross-Entropy Method (CEM), and MPPI
  (Williams et al., 2017) select actions by scoring imagined trajectories under a model, the
  scheme used here in its simplest (random-shooting) form.
- **Model exploitation.** A known failure of model-based RL: optimisation against an
  imperfect model finds adversarial actions that exploit model error (e.g. Janner et al.,
  *MBPO*, 2019). Coverage of the data distribution used to train the model is the standard
  mitigation, central to our findings in §5.
- **Vision-Language-Action policies.** RT-2 (Brohan et al., 2023), OpenVLA (Kim et al., 2024)
  and related models establish the image+language→action setting our policy abstracts.

This work is intentionally a **minimal, controlled reproduction** of the world-model-MPC idea
in the VLA observation setting (pixels + language), prioritising clean single-variable
ablations over scale.

## 3. Method

**Environment.** `ReachEnv`: a point-mass agent on a 2-D canvas with several coloured
objects; a language instruction selects the target by colour. Observations are 64×64 RGB
images; actions are continuous 2-D velocities. Episodes terminate on reaching the target.

**Latent world model.** Three components (`latent_wm.py`):
1. a language-conditioned image encoder `E(image, instruction) → z ∈ ℝ¹²⁸` (FiLM-modulated
   CNN + spatial-softmax keypoints);
2. a residual latent transition `T(z, a) → z′`;
3. a goal head `g(z) → \hat d`, the predicted distance from agent to the instructed target.

Given a start image and an action sequence `a₁..a_H`, the model imagines
`z_t = T(z_{t-1}, a_t)` and predicts `\hat d_t = g(z_t)` for each step. It is trained to
regress the true distance over H-step rollouts (Huber/MSE) on demonstrations, reaching
distance-MSE ≈ 5·10⁻⁵.

**Planner (Imagine-then-Act).** Receding-horizon random-shooting MPC (`mpc.py`). At each
real step: sample K candidate action sequences of length H; encode the current frame to
`z₀`; imagine all K rollouts; score each by predicted distance (final step + best step);
execute the first action of the lowest-scoring candidate; re-plan. Setting `policy_weight>0`
samples candidates around the VLA's proposed action (policy-guided MPC); K=1 around a single
sample reduces to executing the raw proposal.

## 4. Experiments

We evaluate over 100 held-out episodes (fixed seeds) and report success rate. Conditions:
(i) **random policy** (lower bound); (ii) **world-model MPC** with zero-mean candidates and
no policy; (iii) **policy-guided MPC** (VLA proposes, world model selects); (iv) **VLA only**
(upper reference). We additionally sweep the number of imagined candidates K ∈ {1,…,64} for
pure world-model MPC.

## 5. Results

| Controller | Success |
|---|---|
| Random policy | 0.14 |
| **World-model MPC (no policy)** | **0.99** |
| Policy-guided MPC (VLA + world model) | 1.00 |
| VLA only | 1.00 |

Success rises monotonically with imagination: K=1 → 0.19, K=2 → 0.76, K=4 → 0.97,
K≥16 → ≈1.00. See `results/p1_imagine_then_act.png`.

**Model exploitation (ablation).** A world model trained only on expert demonstrations made
the planner *worse* as K grew (more candidates → more chances to exploit model error off the
expert manifold). Training the world model on exploratory data (random actions mixed into the
demos) removed the pathology and produced the results above. This isolates *data coverage*,
not model capacity, as the operative factor.

## 6. Discussion

The central observation is that a model trained purely to predict a scalar progress signal
(distance-to-goal) is, in this setting, an accurate enough simulator to **control through**
with no learned controller. Imagination quantity (K) trades compute for competence, exactly
the lever sampling-based planners expose. Policy-guided MPC recovers the policy's ceiling
while retaining the model's ability to veto bad proposals.

## 7. Limitations

- The testbed is low-dimensional and deterministic; results establish a mechanism, not
  scalability. Stochastic dynamics, contact, and long horizons are untested.
- The goal head predicts a privileged distance signal; a fully image-only reward/goal model
  is a stricter test.
- The language module is deliberately small, so language-side generalisation is studied
  separately (see P3), not here.

## 8. Future work

- Replace the scalar goal head with an image- or goal-conditioned value to remove privileged
  information.
- Stochastic / ensemble dynamics with disagreement-based scoring, both for robustness and as
  an uncertainty signal for when to plan vs. act.
- Port the planner around a pretrained VLA on a standard benchmark (e.g. PushT / LIBERO) to
  test whether imagined selection improves a real policy under distribution shift.

## Reproducibility

`bash scripts/run_all.sh` regenerates all checkpoints and figures from scratch
(~10–15 min, Apple-silicon MPS). Seeds are fixed in the evaluation harness.
