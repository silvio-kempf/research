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
