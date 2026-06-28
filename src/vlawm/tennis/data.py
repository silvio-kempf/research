"""P5 data pipeline: discover tennis clips, parse swing labels, decode frames.

Labels are intent ("action") for the world model: forehand=0, backhand=1. We split
by clip (never by frame) so frames from one swing never leak across train/test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

LABELS = {"forehand": 0, "backhand": 1}
LABEL_NAMES = ["forehand", "backhand"]


@dataclass
class Clip:
    path: Path
    label: int
    name: str


def label_from_name(name: str) -> int:
    stem = name.lower()
    for key, idx in LABELS.items():
        if stem.startswith(key):
            return idx
    raise ValueError(f"cannot parse forehand/backhand from {name!r}")


def discover_clips(raw_dir: Path) -> list[Clip]:
    clips = []
    for p in sorted(raw_dir.iterdir()):
        if p.suffix.lower() not in {".mov", ".mp4"}:
            continue
        clips.append(Clip(path=p, label=label_from_name(p.name), name=p.name))
    return clips


def train_test_split_clips(clips: list[Clip], seed: int = 0) -> tuple[list[Clip], list[Clip]]:
    """Hold out exactly one clip per class as test; rest is train. Deterministic."""
    rng = random.Random(seed)
    test = []
    for label in (0, 1):
        members = [c for c in clips if c.label == label]
        test.append(rng.choice(members))
    test_names = {c.name for c in test}
    train = [c for c in clips if c.name not in test_names]
    return train, test


def decode_clip(clip: Clip, image_size: int = 224) -> np.ndarray:
    """Decode a clip to (T, H, W, 3) uint8, center-cropped square then resized.

    Returns frames at native fps (clips are ~1-1.5s, so no downsampling).
    """
    frames = iio.imread(clip.path, plugin="pyav")  # (T, H, W, 3) uint8
    out = np.stack([_center_crop_resize(f, image_size) for f in frames])
    return out


def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame[top:top + side, left:left + side]
    img = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
    return np.asarray(img)
