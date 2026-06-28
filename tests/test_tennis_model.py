import numpy as np
import torch

from vlawm.worldmodel.tennis_wm import SwingConditionedDynamics
from vlawm.tennis.evaluate import persistence_rollout, rollout_error


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


def test_persistence_repeats_last_context_latent():
    context = np.random.randn(2, 3, 8).astype("float32")
    preds = persistence_rollout(context, horizon=4)
    assert preds.shape == (2, 4, 8)
    for t in range(4):
        assert np.allclose(preds[:, t], context[:, -1])


def test_rollout_error_zero_for_identical():
    a = np.random.randn(2, 5, 8).astype("float32")
    err = rollout_error(a, a)
    assert err.shape == (5,)
    assert np.allclose(err, 0.0, atol=1e-5)
