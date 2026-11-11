"""Leakage-safe SECOM split and compact feature preparation.

Nothing is fitted on validation or test data. All filters, imputation,
standardization and supervised feature selection learn only from train.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def stratified_split_indices(
    y: pd.Series | np.ndarray,
    *,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> SplitIndices:
    """Reproducible 60/20/20 split (by default) with both classes retained.

    ``validation_fraction`` and ``test_fraction`` refer to the FULL dataset,
    not the remaining training portion. Test indices are never used for fitting.
    """
    if test_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Validation + test fractions must be less than 1.")

    y_array = np.asarray(y)
    all_indices = np.arange(len(y_array))
    development, test = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y_array,
    )
    validation_share_of_development = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        development,
        test_size=validation_share_of_development,
        random_state=random_state,
        stratify=y_array[development],
    )
    return SplitIndices(train=train, validation=validation, test=test)


class SecomPreprocessor(BaseEstimator, TransformerMixin):
    """Fit train-only data cleaning and select k sensor readings.

    The returned columns are standardized sensor readings in the *hardware
    input order*. Selection is supervised by the training labels. This is
    univariate feature screening, not evidence of physical causation.

    Operations in fit order:
        1) Drop >= max_missing_fraction missing and constant-in-train sensors.
        2) Replace remaining NaNs with training medians.
        3) Drop any remaining zero-variance columns.
        4) Scale using training means and standard deviations.
        5) Keep top-k training ANOVA-F-ranked columns.
    """

    def __init__(self, k: int = 8, max_missing_fraction: float = 0.50):
        self.k = k
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with sensor column names.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be a positive integer.")
        if not 0 < self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in (0, 1].")

        self.original_columns_ = list(X.columns)
        missing_fractions = X.isna().mean()
        unique_observed = X.nunique(dropna=True)
        keep_mask = (missing_fractions < self.max_missing_fraction) & (unique_observed > 1)
        self.training_columns_ = list(X.columns[keep_mask])
        self.dropped_missing_ = list(X.columns[missing_fractions >= self.max_missing_fraction])
        self.dropped_constant_ = list(X.columns[(missing_fractions < self.max_missing_fraction) & (unique_observed <= 1)])

        if len(self.training_columns_) < self.k:
            raise ValueError(f"Only {len(self.training_columns_)} usable sensors for k={self.k}.")

        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=self.k)),
        ])
        self.pipeline_.fit(X.loc[:, self.training_columns_], np.asarray(y))

        variance_mask = self.pipeline_.named_steps["variance"].get_support()
        survivors = np.asarray(self.training_columns_, dtype=object)[variance_mask]
        selection = self.pipeline_.named_steps["selector"]
        self.selected_features_ = list(survivors[selection.get_support()])
        self.selected_f_scores_ = dict(zip(survivors.tolist(), selection.scores_.tolist()))
        self.extra_zero_variance_ = int(np.sum(~variance_mask))
        self.training_missing_fraction_ = missing_fractions.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["pipeline_", "training_columns_", "selected_features_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with sensor column names.")
        missing_columns = set(self.training_columns_) - set(X.columns)
        if missing_columns:
            raise ValueError(f"Input missing fitted sensor columns: {sorted(missing_columns)[:5]}")
        return self.pipeline_.transform(X.loc[:, self.training_columns_])

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, ["selected_features_"])
        return np.asarray(self.selected_features_, dtype=object)
