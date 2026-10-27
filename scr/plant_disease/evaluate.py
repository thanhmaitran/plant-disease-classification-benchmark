"""Metrics, confusion matrices and CSV export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def summary_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def per_class_table(y_true, y_pred, class_names, model_name: str) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    return pd.DataFrame(
        {"model": model_name, "class": class_names, "precision": p,
         "recall": r, "f1": f, "support": s}
    )


def confusion_table(y_true, y_pred, class_names, model_name: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    rows = [
        {"model": model_name, "true_class": class_names[i],
         "predicted_class": class_names[j], "count": int(cm[i, j])}
        for i in range(len(class_names)) for j in range(len(class_names))
    ]
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
