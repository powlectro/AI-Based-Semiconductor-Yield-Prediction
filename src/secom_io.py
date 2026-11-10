"""Load the original UCI SECOM archive without changing the raw data.

SECOM's original labels are -1=pass and +1=fail.
This project maps them to 0=pass and 1=fail for binary classification.
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def load_secom(zip_path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return sensor table X, binary labels y, and measurement timestamps.

    Dataset rows are kept in their original order. No imputation, scaling,
    or feature selection occurs here: those must be fitted on training data
    only in later modeling steps to avoid data leakage.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"SECOM archive not found: {zip_path}")

    with ZipFile(zip_path) as archive:
        required = {"secom.data", "secom_labels.data"}
        absent = required.difference(archive.namelist())
        if absent:
            raise ValueError(f"Archive missing required files: {sorted(absent)}")

        with archive.open("secom.data") as sensor_file:
            X = pd.read_csv(sensor_file, sep=r"\s+", header=None, na_values=["NaN"])
        with archive.open("secom_labels.data") as label_file:
            # Each line: label "DD/MM/YYYY HH:MM:SS" (quoted timestamp)
            label_data = pd.read_csv(
                label_file,
                sep=r"\s+",
                header=None,
                names=["raw_label", "timestamp_str"],
            )

    if len(X) != len(label_data):
        raise ValueError(f"Row mismatch: X={len(X)}, labels={len(label_data)}")
    original_labels = set(label_data["raw_label"].dropna().unique())
    if original_labels != {-1, 1}:
        raise ValueError(f"Expected original labels {{-1, +1}}, got {original_labels}")

    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    y = label_data["raw_label"].map({-1: 0, 1: 1}).astype("int8").rename("failure")
    timestamps = pd.to_datetime(
        label_data["timestamp_str"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    ).rename("timestamp")

    return X, y, timestamps
