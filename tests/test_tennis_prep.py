import numpy as np

from vlawm.tennis.prep import clean_latents, Standardizer


def test_clean_drops_corrupt_jump():
    # 4 smooth frames, then one wildly different (corrupt), then smooth again
    base = np.ones((1, 8), dtype="float32")
    seq = np.concatenate([base * 1.0, base * 1.01, -base * 5.0, base * 1.02])
    cleaned = clean_latents(seq, max_cos_dist=0.3)
    assert cleaned.shape[0] == 3  # the anti-correlated frame is dropped


def test_clean_keeps_smooth_sequence():
    seq = np.cumsum(np.ones((10, 4), dtype="float32") * 0.001, axis=0) + 1.0
    cleaned = clean_latents(seq, max_cos_dist=0.3)
    assert cleaned.shape[0] == 10


def test_standardizer_roundtrip():
    x = np.random.randn(20, 5).astype("float32") * 3 + 7
    s = Standardizer.fit(x)
    z = s.transform(x)
    assert np.allclose(z.mean(0), 0, atol=1e-4)
    assert np.allclose(z.std(0), 1, atol=1e-3)
    assert np.allclose(s.inverse(z), x, atol=1e-3)
