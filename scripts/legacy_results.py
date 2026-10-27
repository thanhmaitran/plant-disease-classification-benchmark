"""Convert the original coursework JSON into labelled CSV tables.

These numbers come from a run on a DUPLICATED dataset (4,304 images), so they are
stored under legacy/ and must not be presented as clean results.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="legacy")
    args = ap.parse_args()
    d = json.load(open(args.json))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = d["classes"]

    rows, pc, cm = [], [], []
    for key, name in (("svm", "HOG + SVM"), ("cnn", "Custom CNN")):
        rows.append({"model": name, "accuracy": d[key]["acc"],
                     "weighted_precision": d[key]["p"], "weighted_recall": d[key]["r"],
                     "weighted_f1": d[key]["f1"]})
        for c in classes:
            v = d[f"per_class_{key}"][c]
            pc.append({"model": name, "class": c, "precision": v["p"],
                       "recall": v["r"], "f1": v["f1"], "support": v["sup"]})
        m = confusion_matrix(d["y_test"], d[f"y_pred_{key}"])
        for i, t in enumerate(classes):
            for j, p in enumerate(classes):
                cm.append({"model": name, "true_class": t,
                           "predicted_class": p, "count": int(m[i, j])})
    pd.DataFrame(rows).to_csv(out / "legacy_model_metrics.csv", index=False)
    pd.DataFrame(pc).to_csv(out / "legacy_per_class_metrics.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "legacy_confusion_matrix.csv", index=False)
    pd.DataFrame(d["history"]).assign(epoch=lambda x: range(1, len(x) + 1)).to_csv(
        out / "legacy_cnn_history.csv", index=False)
    print("split:", d["split"], "epochs:", d["epochs_ran"], "img:", d["img_size"])


if __name__ == "__main__":
    main()
