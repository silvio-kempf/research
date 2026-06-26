#!/usr/bin/env bash
# Reproduce the full portfolio end-to-end. ~10-15 min on an Apple-silicon laptop.
set -e
cd "$(dirname "$0")/.."

echo "== Shared core: train the VLA policy =="
uv run python scripts/train_policy.py --episodes 2500 --epochs 60

echo "== Train the latent world model (P1) =="
uv run python scripts/train_world_model.py --epochs 40 --horizon 5

echo "== P3: VLA failure analysis =="
uv run python p3_failure_analysis/run_failure_analysis.py

echo "== P1: Imagine-then-Act planning =="
uv run python p1_imagine_then_act/run_planner.py

echo "== P2: Action-conditioned video world model =="
uv run python p2_video_world_model/run_video_wm.py --epochs 30

echo "== P4: Language robustness fix (closes the P3 loop) =="
uv run python p4_language_robustness/run_language_robustness.py

echo "All done. Figures are in results/."
