"""Audit the dataset: counts, unreadable files, exact duplicates."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_disease.data import add_quality_columns, list_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/task1")
    ap.add_argument("--out", default="outputs/baseline/dataset_audit.csv")
    args = ap.parse_args()

    df = add_quality_columns(list_images(args.data_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("Dataset:", Path(args.data_dir).resolve())
    print("Total files:", len(df))
    print("Unreadable:", int((~df["readable"]).sum()))
    print("Unique (md5):", df["md5"].nunique())
    print("Exact-duplicate files:", len(df) - df["md5"].nunique())
    print(df.groupby("label").agg(files=("md5", "size"), unique=("md5", "nunique")))


if __name__ == "__main__":
    main()
