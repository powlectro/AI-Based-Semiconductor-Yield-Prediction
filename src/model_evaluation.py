
"""Evaluation utilities for rare semiconductor failure detection.

Label convention:
0 = Pass
1 = Fail

False Negative (FN) = Escaped manufacturing failure
False Positive (FP) = False alarm / unnecessary rejection
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_at_threshold(
    y_true,
    scores,
    threshold,
    escape_cost=10,
    false_alarm_cost=1,
):
    """Evaluate predictions at a chosen classification threshold."""

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if (
        y_true.ndim != 1
        or scores.ndim != 1
        or len(y_true) != len(scores)
    ):
        raise ValueError(
            "y_true and scores must be equally sized 1-D arrays"
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Labels must use 0=pass and 1=fail")

    if not np.isfinite(scores).all() or np.any(
        (scores < 0) | (scores > 1)
    ):
        raise ValueError("Scores must be finite and in [0, 1]")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")

    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y_true,
            predicted,
            labels=[0, 1],
        ).ravel(),
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predicted)
        ),
        "failure_precision": float(
            precision_score(
                y_true, predicted, zero_division=0
            )
        ),
        "failure_recall": float(
            recall_score(
                y_true, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "false_positive_rate": (
            float(fp / (fp + tn))
            if (fp + tn) else 0.0
        ),
        "pr_auc": (
            float(average_precision_score(y_true, scores))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "screening_cost": int(
            escape_cost * fn + false_alarm_cost * fp
        ),
    }


def choose_threshold(
    y_validation,
    scores,
    escape_cost=10,
    false_alarm_cost=1,
):
    """Select threshold using validation data only."""

    candidates = np.round(
        np.arange(0.05, 0.951, 0.05),
        2,
    )

    metrics = [
        evaluate_at_threshold(
            y_validation,
            scores,
            float(t),
            escape_cost,
            false_alarm_cost,
        )
        for t in candidates
    ]

    selected = min(
        metrics,
        key=lambda m: (
            m["screening_cost"],
            m["fp"],
            abs(m["threshold"] - 0.5),
        ),
    )

    return selected, metrics


def oversample_minority_train_only(
    X_train,
    y_train,
    seed=42,
):
    """Balance classes by repeating minority training samples."""

    X = np.asarray(X_train)
    y = np.asarray(y_train, dtype=int)

    classes, counts = np.unique(
        y, return_counts=True
    )

    if (
        X.ndim != 2
        or X.shape[0] != len(y)
        or len(classes) != 2
    ):
        raise ValueError(
            "Expected a 2-D training matrix with both classes"
        )

    rng = np.random.default_rng(seed)

    major = np.flatnonzero(
        y == classes[np.argmax(counts)]
    )

    minor = np.flatnonzero(
        y == classes[np.argmin(counts)]
    )

    sampled_minor = rng.choice(
        minor,
        size=len(major),
        replace=True,
    )

    selected = np.concatenate([
        major,
        sampled_minor,
    ])

    rng.shuffle(selected)

    return X[selected], y[selected]
