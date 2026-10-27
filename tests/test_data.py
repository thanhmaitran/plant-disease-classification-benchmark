import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plant_disease.data import (
    add_quality_columns, drop_duplicates, find_class_root, list_images,
    load_array, make_splits, split_overlap,
)


def _make_dataset(root, counts, size=32):
    rng = np.random.default_rng(0)
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


@pytest.fixture
def dataset(tmp_path):
    _make_dataset(tmp_path / "ds", {"Potato___A": 20, "Potato___B": 20, "Potato___healthy": 10})
    return tmp_path / "ds"


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_class_root(tmp_path / "nope")


def test_nested_folder_is_found(tmp_path):
    _make_dataset(tmp_path / "outer" / "inner", {"A": 3, "B": 3})
    assert find_class_root(tmp_path / "outer").name == "inner"


def test_list_images_labels_and_counts(dataset):
    df = list_images(dataset)
    assert set(df["label"]) == {"A", "B", "healthy"}
    assert len(df) == 50


def test_corrupt_image_flagged(dataset):
    bad = dataset / "Potato___A" / "bad.png"
    bad.write_bytes(b"not an image")
    df = add_quality_columns(list_images(dataset))
    assert int((~df["readable"]).sum()) == 1


def test_exact_duplicates_removed(dataset):
    first = dataset / "Potato___A" / "0.png"
    (dataset / "Potato___B" / "copy.png").write_bytes(first.read_bytes())
    df = add_quality_columns(list_images(dataset))
    assert len(drop_duplicates(df)) == len(df) - 1


def test_split_sizes_stratified_and_disjoint(dataset):
    df = make_splits(add_quality_columns(list_images(dataset)), 0.2, 0.2, seed=1)
    assert set(df["split"]) == {"train", "validation", "test"}
    assert split_overlap(df) == 0
    assert set(df[df.split == "test"]["label"]) == {"A", "B", "healthy"}


def test_duplicated_dataset_leaks_across_splits(dataset):
    df = add_quality_columns(list_images(dataset))
    doubled = make_splits(pd.concat([df, df], ignore_index=True), 0.2, 0.2, seed=1)
    assert split_overlap(doubled) > 0


def test_split_is_reproducible(dataset):
    df = add_quality_columns(list_images(dataset))
    a = make_splits(df, 0.2, 0.2, seed=7)["split"].tolist()
    b = make_splits(df, 0.2, 0.2, seed=7)["split"].tolist()
    assert a == b


def test_single_class_rejected():
    df = pd.DataFrame({"image_path": ["a", "b"], "label": ["x", "x"], "md5": ["1", "2"]})
    with pytest.raises(ValueError):
        make_splits(df)


def test_load_array_shape(dataset):
    df = list_images(dataset)
    assert load_array(df["image_path"][:4], 24).shape == (4, 24, 24, 3)
