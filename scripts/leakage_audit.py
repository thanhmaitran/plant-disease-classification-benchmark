"""Quantify how duplicated images inflate test metrics (HOG + SVM).

Scenario A: clean dataset (each image once), stratified split.
Scenario B: every image present twice (4,304 = 2 x 2,152, the size reported in the
original coursework), then split at random -> copies land in train AND test.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease.data import (  # noqa: E402
    add_quality_columns, list_images, load_array, make_splits, split_overlap,
)
from plant_disease.evaluate import summary_metrics  # noqa: E402
from plant_disease.models import build_svm  # noqa: E402
from plant_disease.preprocessing import extract_hog  # noqa: E402


def run(df, feats, classes, seed, c):
    split = df["split"].values
    y = df["label"].map({k: i for i, k in enumerate(classes)}).values
    tr, te = split == "train", split == "test"
    model = build_svm(c=c, seed=seed).fit(feats[tr], y[tr])
    m = summary_metrics(y[te], model.predict(feats[te]))
    m.update(n_train=int(tr.sum()), n_test=int(te.sum()),
             shared_images=split_overlap(df))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="outputs/baseline/leakage_audit.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed, size = cfg["seed"], cfg["data"]["image_size"]
    h = cfg["hog_svm"]

    df = add_quality_columns(list_images(args.data_dir or cfg["data"]["root"]))
    classes = sorted(df["label"].unique())
    imgs = load_array(df["image_path"], size)
    feats = extract_hog(imgs, h["orientations"], h["pixels_per_cell"], h["cells_per_block"])

    clean = make_splits(df, cfg["data"]["test_size"], cfg["data"]["validation_size"], seed)
    res_a = run(clean, feats, classes, seed, h["c"])

    dup = pd.concat([df, df], ignore_index=True)
    dup_feats = np.concatenate([feats, feats])
    dup = make_splits(dup, cfg["data"]["test_size"], cfg["data"]["validation_size"], seed)
    res_b = run(dup, dup_feats, classes, seed, h["c"])

    out = pd.DataFrame(
        [{"scenario": "clean (2,152 unique images)", **res_a},
         {"scenario": "duplicated (4,304 = 2 x 2,152)", **res_b}]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
