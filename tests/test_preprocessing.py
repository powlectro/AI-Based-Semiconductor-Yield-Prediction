"""Step 2 guardrails: split reproducibility and train-only transforms."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import SecomPreprocessor, stratified_split_indices


class TestStratifiedSplits(unittest.TestCase):
    def test_nonoverlap_coverage_and_class_balance(self):
        y = pd.Series([0] * 100 + [1] * 20)
        a = stratified_split_indices(y)
        b = stratified_split_indices(y)
        self.assertTrue(np.array_equal(a.train, b.train))
        self.assertTrue(np.array_equal(a.validation, b.validation))
        self.assertTrue(np.array_equal(a.test, b.test))
        self.assertEqual(len(set(a.train) & set(a.validation)), 0)
        self.assertEqual(len(set(a.train) & set(a.test)), 0)
        self.assertEqual(len(set(a.validation) & set(a.test)), 0)
        self.assertEqual(len(np.unique(np.concatenate([a.train, a.validation, a.test]))), len(y))
        for indices in (a.train, a.validation, a.test):
            self.assertEqual(set(y.iloc[indices]), {0, 1})


class TestTrainOnlyPreprocessing(unittest.TestCase):
    def setUp(self):
        self.X_train = pd.DataFrame({
            "sensor_a": [0.0, 0.2, 0.4, 0.6, 1.2, 1.4, 1.6, 1.8],
            "sensor_b": [1.0, 2.0, np.nan, 2.0, 0.0, 0.0, 1.0, 0.0],
            "constant": [7.0] * 8,
            "mostly_missing": [np.nan] * 7 + [15.0],
        })
        self.y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    def test_train_only_filtering_and_shapes(self):
        prep = SecomPreprocessor(k=2).fit(self.X_train, self.y_train)
        self.assertIn("constant", prep.dropped_constant_)
        self.assertIn("mostly_missing", prep.dropped_missing_)
        self.assertEqual(set(prep.selected_features_), {"sensor_a", "sensor_b"})
        self.assertEqual(prep.transform(self.X_train).shape, (8, 2))
        self.assertTrue(np.isfinite(prep.transform(self.X_train)).all())

    def test_validation_outliers_do_not_change_fitted_scaling(self):
        prep = SecomPreprocessor(k=2).fit(self.X_train, self.y_train)
        scaler = prep.pipeline_.named_steps["scaler"]
        means_before = scaler.mean_.copy()
        medians_before = prep.pipeline_.named_steps["imputer"].statistics_.copy()
        X_val = pd.DataFrame({
            "sensor_a": [1e12], "sensor_b": [np.nan],
            "constant": [1e12], "mostly_missing": [1e12],
        })
        self.assertEqual(prep.transform(X_val).shape, (1, 2))
        np.testing.assert_array_equal(means_before, scaler.mean_)
        np.testing.assert_array_equal(medians_before, prep.pipeline_.named_steps["imputer"].statistics_)

    def test_reject_missing_sensor_column(self):
        prep = SecomPreprocessor(k=1).fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            prep.transform(self.X_train.drop(columns="sensor_a"))


if __name__ == "__main__":
    unittest.main()
