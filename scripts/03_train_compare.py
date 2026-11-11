
"""Step 3: Compare ML models on the SECOM dataset.

Models:
- Logistic Regression
- Random Forest
- Small Neural Network

Feature counts: 8, 16, 32

Training data: Fit preprocessing and models
Validation data: Select thresholds and compare models
Test data: Remains untouched
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_evaluation import (
    choose_threshold,
    evaluate_at_threshold,
    oversample_minority_train_only,
)

from src.preprocessing import (
    SecomPreprocessor,
    stratified_split_indices,
)

from src.secom_io import load_secom


SEED = 42
FEATURE_COUNTS = (8, 16, 32)

# Illustrative semiconductor screening costs
ESCAPE_COST = 10
FALSE_ALARM_COST = 1

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"


def make_models():
    """Create the three model types."""

    return {
        "logistic": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),

        "neural_net": MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=0.05,
            learning_rate_init=0.002,
            max_iter=300,
            n_iter_no_change=20,
            random_state=SEED,
        ),
    }


def save_curves(best_per_family, y_validation):
    """Plot precision-recall and ROC curves."""

    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

    for family, record in best_per_family.items():

        scores = record["scores"]

        precision, recall, _ = precision_recall_curve(
            y_validation,
            scores,
        )

        fpr, tpr, _ = roc_curve(
            y_validation,
            scores,
        )

        label = (
            f"{family} "
            f"(k={record['features']}, "
            f"AP={record['pr_auc']:.3f})"
        )

        ax_pr.plot(
            recall,
            precision,
            label=label,
        )

        ax_roc.plot(
            fpr,
            tpr,
            label=(
                f"{family} "
                f"(k={record['features']})"
            ),
        )

    ax_pr.axhline(
        np.mean(y_validation),
        linestyle="--",
        color="gray",
        label="Failure prevalence",
    )

    ax_pr.set(
        xlabel="Failure recall",
        ylabel="Failure precision",
        title="SECOM validation: Precision-Recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax_pr.legend(fontsize=8)
    fig_pr.tight_layout()

    fig_pr.savefig(
        FIGURES / "step3_precision_recall.png",
        dpi=160,
    )

    plt.close(fig_pr)

    ax_roc.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Random ranking",
    )

    ax_roc.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="SECOM validation: ROC curves",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()

    fig_roc.savefig(
        FIGURES / "step3_roc_curves.png",
        dpi=160,
    )

    plt.close(fig_roc)


def main():

    for folder in (REPORTS, FIGURES, ARTIFACTS):
        folder.mkdir(parents=True, exist_ok=True)

    # Load dataset using our Step 1 code
    X, y, _timestamps = load_secom(
        ROOT / "data" / "raw" / "secom.zip"
    )

    # Reuse the exact Step 2 split
    split = stratified_split_indices(
        y,
        random_state=SEED,
    )

    X_train = X.iloc[split.train]
    y_train = y.iloc[split.train].to_numpy()

    X_val = X.iloc[split.validation]
    y_val = y.iloc[split.validation].to_numpy()

    # Do not inspect or evaluate the held-out test rows.

    rows = []
    sweep_rows = []
    score_records = []

    for k in FEATURE_COUNTS:

        # Train-only sensor preprocessing
        prep = SecomPreprocessor(
            k=k,
            max_missing_fraction=0.50,
        )

        train_matrix = prep.fit_transform(
            X_train,
            y_train,
        )

        val_matrix = prep.transform(X_val)

        for family, model in make_models().items():

            # MLP minority oversampling uses TRAIN only
            if family == "neural_net":
                fit_X, fit_y = (
                    oversample_minority_train_only(
                        train_matrix,
                        y_train,
                        SEED,
                    )
                )
            else:
                fit_X, fit_y = train_matrix, y_train

            # Train the model
            model.fit(fit_X, fit_y)

            # Scores from validation only
            scores = model.predict_proba(
                val_matrix
            )[:, 1]

            # Find cost-optimal validation threshold
            selected, sweep = choose_threshold(
                y_val,
                scores,
                escape_cost=ESCAPE_COST,
                false_alarm_cost=FALSE_ALARM_COST,
            )

            default = evaluate_at_threshold(
                y_val,
                scores,
                0.50,
                ESCAPE_COST,
                FALSE_ALARM_COST,
            )

            record = {
                "model": family,
                "features": k,
                "threshold": selected["threshold"],
                "accuracy": selected["accuracy"],
                "failure_precision": selected[
                    "failure_precision"
                ],
                "failure_recall": selected[
                    "failure_recall"
                ],
                "false_positive_rate": selected[
                    "false_positive_rate"
                ],
                "f1": selected["f1"],
                "pr_auc": selected["pr_auc"],
                "roc_auc": selected["roc_auc"],
                "caught_failures_tp": selected["tp"],
                "escaped_failures_fn": selected["fn"],
                "false_alarms_fp": selected["fp"],
                "true_pass_tn": selected["tn"],
                "screening_cost": selected[
                    "screening_cost"
                ],
                "cost_at_threshold_050": default[
                    "screening_cost"
                ],
            }

            rows.append(record)

            for item in sweep:
                sweep_rows.append({
                    "model": family,
                    "features": k,
                    **item,
                })

            score_records.append({
                **record,
                "scores": scores,
            })

            # Save preprocessing + model together
            joblib.dump(
                {
                    "preprocessor": prep,
                    "model": model,
                    "threshold": selected["threshold"],
                    "selected_features": (
                        prep.selected_features_
                    ),
                    "cost_assumptions": {
                        "escape": ESCAPE_COST,
                        "false_alarm": FALSE_ALARM_COST,
                    },
                },
                ARTIFACTS
                / f"step3_{family}_k{k:02d}.joblib",
            )

            print(
                f"{family:14s} "
                f"k={k:02d} "
                f"threshold={selected['threshold']:.2f} "
                f"recall={selected['failure_recall']:.3f} "
                f"false_alarms={selected['fp']:3d} "
                f"cost={selected['screening_cost']:3d}"
            )

    # Sort by assumed screening cost
    results = pd.DataFrame(rows).sort_values(
        [
            "screening_cost",
            "false_alarms_fp",
            "features",
            "model",
        ]
    ).reset_index(drop=True)

    results.to_csv(
        REPORTS / "step3_validation_results.csv",
        index=False,
        float_format="%.5f",
    )

    pd.DataFrame(sweep_rows).to_csv(
        REPORTS / "step3_threshold_sweep.csv",
        index=False,
    )

    best = results.iloc[0].to_dict()

    # Best feature count for each model family
    best_per_family = {}

    for family in (
        "logistic",
        "random_forest",
        "neural_net",
    ):
        records = [
            r for r in score_records
            if r["model"] == family
        ]

        best_per_family[family] = min(
            records,
            key=lambda r: (
                r["screening_cost"],
                r["false_alarms_fp"],
                r["features"],
            ),
        )

    save_curves(best_per_family, y_val)

    # Confusion matrix of best validation model
    confusion_matrix = np.array([
        [
            int(best["true_pass_tn"]),
            int(best["false_alarms_fp"]),
        ],
        [
            int(best["escaped_failures_fn"]),
            int(best["caught_failures_tp"]),
        ],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Pass", "Fail"],
    ).plot(
        ax=ax,
        colorbar=False,
        values_format="d",
    )

    ax.set_title(
        f"Best validation model: "
        f"{best['model']} "
        f"k={int(best['features'])}"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "step3_best_confusion_matrix.png",
        dpi=160,
    )

    plt.close(fig)

    summary = {
        "best_validation_model": {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in best.items()
        },
        "candidate_count": len(results),
        "threshold_candidates": (
            "0.05 to 0.95 in 0.05 steps"
        ),
        "cost_assumptions": {
            "escaped_failure": ESCAPE_COST,
            "false_alarm": FALSE_ALARM_COST,
            "note": (
                "Illustrative costs, "
                "not actual fab prices."
            ),
        },
        "warning": (
            "Models and thresholds were selected "
            "using only 21 validation failures. "
            "Test remains untouched. "
            "Probability scores may be uncalibrated."
        ),
    }

    (
        REPORTS / "step3_selection.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== STEP 3: VALIDATION ONLY ==="
    )

    print(
        results[
            [
                "model",
                "features",
                "threshold",
                "failure_recall",
                "false_positive_rate",
                "pr_auc",
                "screening_cost",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSelected by ASSUMED cost: "
        f"{best['model']} with "
        f"{int(best['features'])} features"
    )

    print(
        "Saved CSV/JSON reports, 3 plots, "
        "and 9 trained model artifacts."
    )

    print(
        "IMPORTANT: Test set has not been evaluated."
    )


if __name__ == "__main__":
    main()
