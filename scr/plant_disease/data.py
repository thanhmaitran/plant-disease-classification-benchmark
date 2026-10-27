"""Dataset discovery, validation, de-duplication and splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_class_root(root: str | Path) -> Path:
    """Return the directory that directly contains the class folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    candidates = [p for p in [root, *root.rglob("*")] if p.is_dir()]
    best, best_n = None, 0
    for cand in candidates:
        subdirs = [d for d in cand.iterdir() if d.is_dir()]
        n = sum(
            1
            for d in subdirs
            if any(f.suffix.lower() in VALID_EXTENSIONS for f in d.iterdir())
        )
        if n > best_n:
            best, best_n = cand, n
    if best is None or best_n < 2:
        raise RuntimeError("No class directories found. Check the dataset path.")
    return best


def list_images(root: str | Path) -> pd.DataFrame:
    """Build a metadata table with columns image_path, label."""
    class_root = find_class_root(root)
    rows = []
    for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
        label = class_dir.name.replace("Potato___", "")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                rows.append({"image_path": str(path), "label": label})
    if not rows:
        raise RuntimeError("No images found.")
    return pd.DataFrame(rows)


def is_readable(path: str | Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def file_hash(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `readable` and `md5` columns used for the integrity audit."""
    df = df.copy()
    df["readable"] = df["image_path"].map(is_readable)
    df["md5"] = df["image_path"].map(file_hash)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each exact-duplicate image."""
    if "md5" not in df:
        df = add_quality_columns(df)
    return df.drop_duplicates(subset="md5").reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified train/validation/test split; returns df with a `split` column."""
    if df["label"].nunique() < 2:
        raise ValueError("At least two classes are required.")
    idx = np.arange(len(df))
    tmp_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=df["label"], random_state=seed
    )
    rel_val = validation_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        tmp_idx, test_size=rel_val, stratify=df["label"].iloc[tmp_idx], random_state=seed
    )
    out = df.copy()
    out["split"] = ""
    out.iloc[train_idx, out.columns.get_loc("split")] = "train"
    out.iloc[val_idx, out.columns.get_loc("split")] = "validation"
    out.iloc[test_idx, out.columns.get_loc("split")] = "test"
    return out


def split_overlap(df: pd.DataFrame) -> int:
    """Number of distinct images (by md5) that appear in more than one split."""
    g = df.groupby("md5")["split"].nunique()
    return int((g > 1).sum())


def load_array(paths, size: int) -> np.ndarray:
    return np.stack(
        [np.array(Image.open(p).convert("RGB").resize((size, size))) for p in paths]
    )
