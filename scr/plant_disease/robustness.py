"""Image corruptions for robustness evaluation (uint8 RGB in, uint8 RGB out)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def brightness(images: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(images.astype("float32") * factor, 0, 255).astype("uint8")


def gaussian_noise(images: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Additive noise; sigma is the std on the [0, 1] intensity scale."""
    rng = np.random.default_rng(seed)
    x = images.astype("float32") / 255.0
    x = x + rng.normal(0.0, sigma, size=x.shape)
    return (np.clip(x, 0, 1) * 255).astype("uint8")


def blur(images: np.ndarray, sigma: float) -> np.ndarray:
    out = gaussian_filter(images.astype("float32"), sigma=(0, sigma, sigma, 0))
    return np.clip(out, 0, 255).astype("uint8")


CORRUPTIONS = {"brightness": brightness, "gaussian_noise": gaussian_noise, "blur": blur}


def apply(name: str, images: np.ndarray, level: float, seed: int = 42) -> np.ndarray:
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption: {name}")
    if name == "gaussian_noise":
        return gaussian_noise(images, level, seed)
    return CORRUPTIONS[name](images, level)
