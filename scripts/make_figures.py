"""Generate README figures from the CSV outputs and the dataset."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402

FIG = Path("reports/figures")
COL = {"Early_blight": "#e67e22", "Late_blight": "#8e44ad", "healthy": "#27ae60"}


def class_distribution(split_csv):
    df = pd.read_csv(split_csv)
    tab = df.groupby(["label", "split"]).size().unstack()[["train", "validation", "test"]]
    ax = tab.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#34495e", "#7f8c8d", "#bdc3c7"])
    ax.set_title("Class distribution by split (de-duplicated dataset)", fontweight="bold")
    ax.set_ylabel("Images"); ax.set_xlabel("")
    plt.xticks(rotation=0)
    for i, t in enumerate(tab.sum(axis=1)):
        ax.text(i, t + 10, str(int(t)), ha="center", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=150); plt.close()


def sample_images(split_csv):
    df = pd.read_csv(split_csv)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (label, g) in zip(axs, df.groupby("label")):
        ax.imshow(Image.open(g.sort_values("image_path").iloc[5]["image_path"]).convert("RGB"))
        ax.set_title(label, fontweight="bold"); ax.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "sample_images.png", dpi=150); plt.close()


def confusion(cm_csv, legacy_csv=None):
    cm = pd.read_csv(cm_csv)
    models = list(cm["model"].unique())
    fig, axs = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
    for ax, m in zip(axs[0], models):
        t = cm[cm.model == m].pivot(index="true_class", columns="predicted_class", values="count")
        sns.heatmap(t, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(m, fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=150); plt.close()


def model_comparison(metrics_csv):
    m = pd.read_csv(metrics_csv).set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]
    ax = m.T.plot(kind="bar", figsize=(7, 4), colormap="Set2")
    ax.set_ylim(0, 1.05); ax.set_title("Test-set metrics (clean split)", fontweight="bold")
    plt.xticks(rotation=0); ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=150); plt.close()


def robustness(rob_csv):
    r = pd.read_csv(rob_csv)
    base = r[r.corruption == "none"].set_index("model")["accuracy"]
    names = [c for c in r.corruption.unique() if c != "none"]
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), sharey=True)
    for ax, c in zip(axs, names):
        for model, g in r[r.corruption == c].groupby("model"):
            g = g.sort_values("severity")
            ax.plot(g.severity, g.accuracy, marker="o", label=model)
            ax.axhline(base[model], ls=":", lw=1, color="gray")
        ax.set_title(c.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("severity"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Accuracy"); axs[0].legend()
    plt.tight_layout(); plt.savefig(FIG / "robustness_curves.png", dpi=150); plt.close()


def leakage(audit_csv):
    a = pd.read_csv(audit_csv)
    a["short"] = ["Clean\n(2,152 unique)", "Duplicated\n(4,304 = 2 x 2,152)"]
    ax = a.set_index("short")[["accuracy", "macro_f1"]].plot(
        kind="bar", figsize=(6.5, 4), color=["#2980b9", "#c0392b"])
    ax.set_ylim(0, 1.05); plt.xticks(rotation=0); ax.grid(axis="y", alpha=0.3)
    ax.set_title("HOG + SVM: effect of duplicated images in the split", fontweight="bold")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f")
    plt.tight_layout(); plt.savefig(FIG / "leakage_impact.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="outputs")
    b = Path(ap.parse_args().base)
    FIG.mkdir(parents=True, exist_ok=True)
    class_distribution(b / "baseline/split.csv"); sample_images(b / "baseline/split.csv")
    confusion(b / "baseline/confusion_matrix.csv")
    model_comparison(b / "baseline/model_metrics.csv")
    robustness(b / "robustness/robustness_metrics.csv")
    leakage(b / "baseline/leakage_audit.csv")


if __name__ == "__main__":
    main()
