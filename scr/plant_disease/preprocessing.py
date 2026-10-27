"""Feature extraction and image preprocessing."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog


def extract_hog(images: np.ndarray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)) -> np.ndarray:
    """HOG descriptors for a batch of RGB uint8 images (N, H, W, 3)."""
    feats = []
    for img in images:
        gray = (rgb2gray(img) * 255).astype(np.uint8)
        feats.append(
            hog(
                gray,
                orientations=orientations,
                pixels_per_cell=tuple(pixels_per_cell),
                cells_per_block=tuple(cells_per_block),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.array(feats)


def normalise(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images to float32 in [0, 1]."""
    return images.astype("float32") / 255.0
