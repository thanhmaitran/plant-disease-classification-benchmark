"""Run baseline + robustness experiments on one shared split.

Examples
--------
python scripts/run_experiments.py --models svm
python scripts/run_experiments.py --models svm cnn --data-dir /path/to/task1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease import robustness  # noqa: E402
from plant_disease.data import (  # noqa: E402
    add_quality_columns, drop_duplicates, list_images, load_array,
    make_splits, split_overlap,
)
from plant_disease.evaluate import (  # noqa: E402
    confusion_table, per_class_table, save_csv, summary_metrics,
)
from plant_disease.preprocessing import extract_hog, normalise  # noqa: E402
from plant_disease.train import train_cnn, train_hog_svm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--robustness-config", default="configs/robustness.yaml")
    ap.add_argument("--data-dir", default=None, help="Overrides data.root in config")
    ap.add_argument("--models", nargs="+", default=["svm", "cnn"], choices=["svm", "cnn"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rcfg = yaml.safe_load(open(args.robustness_config))
    seed, size = cfg["seed"], cfg["data"]["image_size"]
    out = Path(cfg["output"]["directory"])
    rout = Path(rcfg["output"]["directory"])

    # 1. metadata, integrity checks, de-duplication, ONE shared split
    df = add_quality_columns(list_images(args.data_dir or cfg["data"]["root"]))
    df = df[df["readable"]].reset_index(drop=True)
    n_raw = len(df)
    if cfg["data"]["deduplicate"]:
        df = drop_duplicates(df)
    print(f"images: raw={n_raw} after-dedup={len(df)}")
    df = make_splits(df, cfg["data"]["test_size"], cfg["data"]["validation_size"], seed)
    assert split_overlap(df) == 0, "Leakage: identical image in several splits"
    save_csv(df, out / "split.csv")

    classes = sorted(df["label"].unique())
    cid = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(cid).values
    imgs = load_array(df["image_path"], size)
    tr, va, te = (df["split"] == s for s in ("train", "validation", "test"))
    x_tr, x_va, x_te = imgs[tr], imgs[va], imgs[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    print("split:", {"train": len(x_tr), "val": len(x_va), "test": len(x_te)})

    h = cfg["hog_svm"]

    def hog_of(x):
        return extract_hog(x, h["orientations"], h["pixels_per_cell"], h["cells_per_block"])

    predictors, metrics_rows, pc_rows, cm_rows, pred_rows = {}, [], [], [], []

    def record(name, y_pred, train_s, infer_s, extra=None):
        m = {"model": name, **summary_metrics(y_te, y_pred),
             "train_seconds": round(train_s, 1), "inference_seconds": round(infer_s, 2),
             "n_train": len(x_tr), "n_val": len(x_va), "n_test": len(x_te)}
        m.update(extra or {})
        metrics_rows.append(m)
        pc_rows.append(per_class_table(y_te, y_pred, classes, name))
        cm_rows.append(confusion_table(y_te, y_pred, classes, name))
        pred_rows.append(pd.DataFrame({
            "model": name, "image_path": df.loc[te, "image_path"].values,
            "true_class": [classes[i] for i in y_te],
            "predicted_class": [classes[i] for i in y_pred],
            "correct": y_te == y_pred}))

    # 2. models
    if "svm" in args.models:
        svm, t_train = train_hog_svm(x_tr, y_tr, h, seed)
        t0 = time.time()
        pred = svm.predict(hog_of(x_te))
        record("HOG + SVM", pred, t_train, time.time() - t0)
        predictors["HOG + SVM"] = lambda x, m=svm: m.predict(hog_of(x))

    if "cnn" in args.models:
        cnn, hist, t_train = train_cnn(x_tr, y_tr, x_va, y_va, cfg["cnn"], size, seed)
        t0 = time.time()
        pred = np.argmax(cnn.predict(normalise(x_te), verbose=0), axis=1)
        record("Custom CNN", pred, t_train, time.time() - t0,
               {"epochs_ran": len(hist["loss"])})
        pd.DataFrame(hist).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(
            out / "cnn_history.csv", index=False)
        predictors["Custom CNN"] = lambda x, m=cnn: np.argmax(
            m.predict(normalise(x), verbose=0), axis=1)

    save_csv(pd.DataFrame(metrics_rows), out / "model_metrics.csv")
    save_csv(pd.concat(pc_rows), out / "per_class_metrics.csv")
    save_csv(pd.concat(cm_rows), out / "confusion_matrix.csv")
    save_csv(pd.concat(pred_rows), out / "predictions.csv")

    # 3. robustness on the SAME test images
    rrows = []
    for name, predict in predictors.items():
        base = summary_metrics(y_te, predict(x_te))
        rrows.append({"model": name, "corruption": "none", "severity": 0, **base})
        for spec in rcfg["corruptions"]:
            for level in spec["levels"]:
                xc = robustness.apply(spec["name"], x_te, level, seed)
                rrows.append({"model": name, "corruption": spec["name"],
                              "severity": level,
                              **summary_metrics(y_te, predict(xc))})
                print(name, spec["name"], level, round(rrows[-1]["accuracy"], 4))
    save_csv(pd.DataFrame(rrows), rout / "robustness_metrics.csv")
    print("done ->", out, rout)


if __name__ == "__main__":
    main()
