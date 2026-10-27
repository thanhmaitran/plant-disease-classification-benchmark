import pandas as pd

from plant_disease.evaluate import (
    confusion_table, per_class_table, save_csv, summary_metrics,
)


def test_perfect_predictions():
    m = summary_metrics([0, 1, 2, 1], [0, 1, 2, 1])
    assert m == {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}


def test_macro_f1_penalises_missed_minority_class():
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 10
    m = summary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.8 and m["macro_f1"] < m["weighted_f1"]


def test_per_class_recall_and_support():
    t = per_class_table([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"], "m")
    assert t.loc[t["class"] == "a", "recall"].iloc[0] == 0.5
    assert t["support"].tolist() == [2, 2]


def test_confusion_table_sums_to_samples():
    t = confusion_table([0, 1, 1], [0, 1, 0], ["a", "b"], "m")
    assert t["count"].sum() == 3 and list(t["true_class"].unique()) == ["a", "b"]


def test_class_with_no_predictions_does_not_crash():
    t = per_class_table([0, 1], [0, 0], ["a", "b"], "m")
    assert t.loc[t["class"] == "b", "precision"].iloc[0] == 0.0


def test_save_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"model": ["x"], "accuracy": [0.5]})
    save_csv(df, tmp_path / "sub" / "m.csv")
    assert pd.read_csv(tmp_path / "sub" / "m.csv").equals(df)
