"""Minimal text encoder so the policy is conditioned on real language, not an id.

A bag-of-embeddings over a fixed word vocabulary. Tiny on purpose — the research point
is *grounding* (binding the instruction's color word to a region in the image), not NLP.
Words unseen at train time (held-out colors, paraphrase vocabulary) map to <unk>, which
is what makes novel-color and paraphrase evaluation genuinely out-of-distribution.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

# Vocabulary the policy is trained on. Only the canonical template + base colors.
# Paraphrase words ("one", "navigate", "towards", "go", "move") and novel colors
# ("purple", "orange", "cyan") are deliberately absent -> they become <unk> at test time.
TRAIN_VOCAB = ["<pad>", "<unk>", "reach", "the", "object",
               "red", "green", "blue", "yellow"]
WORD2ID = {w: i for i, w in enumerate(TRAIN_VOCAB)}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def encode_ids(text: str, max_len: int = 6) -> list[int]:
    ids = [WORD2ID.get(w, WORD2ID["<unk>"]) for w in tokenize(text)][:max_len]
    ids += [WORD2ID["<pad>"]] * (max_len - len(ids))
    return ids


class TextEncoder(nn.Module):
    """Bag-of-embeddings (mean over non-pad tokens) -> language vector."""

    def __init__(self, out_dim: int = 16, max_len: int = 6):
        super().__init__()
        self.embed = nn.Embedding(len(TRAIN_VOCAB), out_dim, padding_idx=0)
        self.max_len = max_len
        self.out_dim = out_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, max_len) long
        emb = self.embed(token_ids)                      # (B, L, D)
        mask = (token_ids != 0).float().unsqueeze(-1)    # ignore <pad>
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1.0)
        return summed / denom

    def encode_texts(self, texts: list[str], device) -> torch.Tensor:
        ids = torch.tensor([encode_ids(t, self.max_len) for t in texts],
                           dtype=torch.long, device=device)
        return self.forward(ids)
