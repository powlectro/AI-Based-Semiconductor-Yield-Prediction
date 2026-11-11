"""Step 2: training-only preprocessing and a first cost-aware ML baseline.

Run from repository root:
    .\\.venv\\Scripts\\python.exe scripts/02_prepare_features.py

Only TRAIN is used to fit preprocessors and models. VALIDATION is used for
comparison. TEST is reserved for the final, one-time evaluation.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import SecomPreprocessor, stratified_split_indices  # noqa: E402
from src.secom_io import load_secom  # noqa: E402

ARCHIVE = ROOT / "data" / "raw" / "secom.zip"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
FEATURE_COUNTS = (8, 16, 32)


def score_validation(y_true: np.ndarray, predicted: np.ndarray, p_fail: np.ndarray) -> dict:
    """In manufacturing terms, FN=escaped failure; FP=unnecessary rejection."""
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel())
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "failure_recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "pr_auc_average_precision": round(float(average_precision_score(y_true, p_fail)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_fail)), 4),
        "true_pass": tn,
        "false_alarm_fp": fp,
        "escaped_failure_fn": fn,
        "caught_failure_tp": tp,
    }


def split_summary(y: pd.Series, indices: np.ndarray) -> dict:
    split_labels = y.iloc[indices]
    return {
        "samples": int(len(indices)),
        "pass": int((split_labels == 0).sum()),
        "fail": int((split_labels == 1).sum()),
        "failure_rate_percent": round(float(100 * split_labels.mean()), 3),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X, y, timestamps = load_secom(ARCHIVE)
    split = stratified_split_indices(y, random_state=SEED)
    X_train, y_train = X.iloc[split.train], y.iloc[split.train]
    X_val, y_val = X.iloc[split.validation], y.iloc[split.validation]

    assignments = np.full(len(X), "", dtype=object)
    assignments[split.train] = "train"
    assignments[split.validation] = "validation"
    assignments[split.test] = "test"
    pd.DataFrame({
        "row_index": np.arange(len(X)),
        "split": assignments,
        "timestamp": timestamps.dt.strftime("%Y-%m-%d %H:%M:%S"),
    }).to_csv(REPORTS / "split_assignments.csv", index=False)
    summaries = {
        "seed": SEED,
        "split_strategy": "random stratified 60/20/20; no lot identifiers available",
        "train": split_summary(y, split.train),
        "validation": split_summary(y, split.validation),
        "test": split_summary(y, split.test),
        "caveat": "Time drift has NOT been tested. Holdout test rows must remain sealed until final evaluation.",
    }
    (REPORTS / "split_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # A 'predict every sample passes' model provides a deliberately naive reference.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    naive_prediction = dummy.predict(np.zeros((len(y_val), 1)))
    naive_fail_probability = dummy.predict_proba(np.zeros((len(y_val), 1)))[:, 1]
    results = [{
        "model": "always_pass",
        "features": 0,
        **score_validation(y_val.to_numpy(), naive_prediction, naive_fail_probability),
    }]

    feature_rows = []
    fit_summary = {}
    for k in FEATURE_COUNTS:
        prep = SecomPreprocessor(k=k, max_missing_fraction=0.50)
        train_matrix = prep.fit_transform(X_train, y_train)
        val_matrix = prep.transform(X_val)
        classifier = LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED
        )
        classifier.fit(train_matrix, y_train)
        predictions = classifier.predict(val_matrix)  # Fixed 0.50 threshold for Step 2.
        p_fail = classifier.predict_proba(val_matrix)[:, 1]
        results.append({
            "model": f"logistic_k{k:02d}",
            "features": k,
            **score_validation(y_val.to_numpy(), predictions, p_fail),
        })

        # This single artifact preserves the exact data-prep steps + model for Step 3.
        joblib.dump({"preprocessor": prep, "model": classifier, "threshold": 0.50},
                    ARTIFACTS / f"logistic_k{k:02d}.joblib")

        fit_summary[str(k)] = {
            "original_sensors": int(X.shape[1]),
            "dropped_for_missingness": len(prep.dropped_missing_),
            "dropped_train_constant": len(prep.dropped_constant_),
            "extra_zero_variance": prep.extra_zero_variance_,
            "kept_after_filters": len(prep.training_columns_) - prep.extra_zero_variance_,
            "selected_sensor_input_order": prep.selected_features_,
        }
        for position, feature in enumerate(prep.selected_features_):
            feature_rows.append({
                "feature_count": k,
                "input_position": position,
                "sensor": feature,
                "anova_f_score_train": round(float(prep.selected_f_scores_[feature]), 6),
                "missing_percent_train": round(100 * prep.training_missing_fraction_[feature], 3),
            })

    pd.DataFrame(feature_rows).to_csv(REPORTS / "selected_features.csv", index=False)
    result_df = pd.DataFrame(results)
    result_df.to_csv(REPORTS / "validation_baseline_metrics.csv", index=False)
    (REPORTS / "preprocessing_summary.json").write_text(
        json.dumps({"missing_threshold": 0.50, "feature_counts": fit_summary,
                    "baseline_threshold": 0.50, "baseline_class_weight": "balanced",
                    "selected_from": "training data only"}, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for _, row in result_df.iterrows():
        ax.scatter(row["false_positive_rate"], row["failure_recall"], s=70)
        ax.annotate(row["model"], (row["false_positive_rate"], row["failure_recall"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="False positive rate (unnecessary rejections / passing samples)",
           ylabel="Failure recall (caught failures / actual failures)",
           title="SECOM validation: detecting rare failures vs false alarms",
           xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_recall_vs_false_alarms.png", dpi=160)
    plt.close(fig)

    print("\n=== STEP 2: SPLITS ===")
    for name in ("train", "validation", "test"):
        print(f"{name:>10s}: {summaries[name]}")
    print("\n=== TRAIN-ONLY FEATURE PREPROCESSING ===")
    for k in FEATURE_COUNTS:
        s = fit_summary[str(k)]
        print(f"k={k:02d}: kept {s['kept_after_filters']} sensors after filtering; "
              f"selected {', '.join(s['selected_sensor_input_order'])}")
    print("\n=== VALIDATION METRICS (TEST SET NOT EVALUATED) ===")
    print(result_df[["model", "accuracy", "failure_recall", "false_positive_rate",
                     "pr_auc_average_precision", "escaped_failure_fn", "false_alarm_fp"]]
          .to_string(index=False))
    print("\nSaved reports: split_assignments.csv, split_summary.json, ")
    print("preprocessing_summary.json, selected_features.csv, validation_baseline_metrics.csv")
    print("Saved figure: reports/figures/validation_recall_vs_false_alarms.png")
    print("Saved models: artifacts/logistic_k08.joblib, k16.joblib, k32.joblib")
    print("\nIMPORTANT: TEST remains held out. No fitted values were learned from validation/test rows.")


if __name__ == "__main__":
    main()
