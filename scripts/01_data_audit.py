"""Step 1: reproducible exploratory audit of the raw SECOM manufacturing data.

Run from the project root:
    python scripts/01_data_audit.py

This is exploratory analysis only: it does NOT fit a model or a train/test scaler.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures even when there is no GUI
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.secom_io import load_secom  # noqa: E402

ARCHIVE = ROOT / "data" / "raw" / "secom.zip"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    X, y, timestamps = load_secom(ARCHIVE)

    missing_fraction = X.isna().mean()
    distinct_nonmissing = X.nunique(dropna=True)
    feature_stats = pd.DataFrame({
        "missing_fraction": missing_fraction,
        "nonmissing_unique_values": distinct_nonmissing,
        "std_dev": X.std(skipna=True),
        "median": X.median(skipna=True),
    })
    feature_stats.index.name = "feature"
    feature_stats.to_csv(REPORTS / "feature_statistics.csv")

    pass_count = int((y == 0).sum())
    fail_count = int((y == 1).sum())
    summary = {
        "samples": int(len(X)),
        "sensor_features": int(X.shape[1]),
        "passing_samples": pass_count,
        "failing_samples": fail_count,
        "failure_rate_percent": round(100 * fail_count / len(y), 3),
        "pass_to_fail_ratio": round(pass_count / fail_count, 3) if fail_count else None,
        "missing_cell_percent": round(100 * X.isna().to_numpy().mean(), 3),
        "features_completely_missing": int(X.isna().all().sum()),
        "features_at_least_50_percent_missing": int((missing_fraction >= 0.5).sum()),
        "features_constant_or_all_missing": int((distinct_nonmissing <= 1).sum()),
        "unique_timestamps": int(timestamps.nunique()),
        "earliest_timestamp": timestamps.min().isoformat(),
        "latest_timestamp": timestamps.max().isoformat(),
    }
    with (REPORTS / "data_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.bar(["Pass (0)", "Fail (1)"], [pass_count, fail_count], color=["#4385b1", "#d27651"])
    ax.set_ylabel("Number of manufacturing samples")
    ax.set_title("SECOM class imbalance")
    for i, count in enumerate([pass_count, fail_count]):
        ax.annotate(f"{count:,}", (i, count), xytext=(0, 5),
                    textcoords="offset points", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.hist(100 * missing_fraction, bins=20, edgecolor="white", color="#4385b1")
    ax.axvline(50, color="#d27651", linestyle="--", label="50% missing")
    ax.set_xlabel("Missing values in a feature (%)")
    ax.set_ylabel("Number of sensor features")
    ax.set_title("SECOM per-feature missingness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_missingness.png", dpi=160)
    plt.close(fig)

    print("\n=== SECOM RAW DATA AUDIT ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nSaved: reports/data_audit.json")
    print("Saved: reports/feature_statistics.csv")
    print("Saved: reports/figures/class_distribution.png")
    print("Saved: reports/figures/feature_missingness.png")
    print("\nNote: Dataset exploration is NOT model preprocessing.")
    print("In Step 2, all imputation/scaling/selection will be fit using TRAIN rows only.")


if __name__ == "__main__":
    main()
