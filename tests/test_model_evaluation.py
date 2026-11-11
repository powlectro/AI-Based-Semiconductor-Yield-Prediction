
import unittest
import numpy as np

from src.model_evaluation import (
    choose_threshold,
    evaluate_at_threshold,
    oversample_minority_train_only,
)


class TestModelEvaluation(unittest.TestCase):

    def test_confusion_matrix_and_cost(self):

        y = np.array([0, 0, 0, 1, 1])

        scores = np.array([
            0.1, 0.7, 0.3, 0.4, 0.9
        ])

        m = evaluate_at_threshold(
            y, scores, 0.5
        )

        self.assertEqual(
            (m["tn"], m["fp"], m["fn"], m["tp"]),
            (2, 1, 1, 1),
        )

        self.assertEqual(
            m["screening_cost"], 11
        )

        self.assertAlmostEqual(
            m["failure_recall"], 0.5
        )

    def test_threshold_uses_validation_scores(self):

        y = np.array([0, 0, 0, 1, 1])

        scores = np.array([
            0.1, 0.7, 0.3, 0.4, 0.9
        ])

        choice, sweep = choose_threshold(
            y, scores
        )

        self.assertEqual(len(sweep), 19)

        self.assertEqual(
            choice["screening_cost"], 1
        )

        self.assertEqual(
            choice["fn"], 0
        )

    def test_oversampling_only_repeats_train_examples(self):

        X = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ])

        y = np.array([0, 0, 0, 0, 1])

        X_bal, y_bal = (
            oversample_minority_train_only(
                X, y, seed=42
            )
        )

        self.assertEqual(
            (y_bal == 0).sum(),
            (y_bal == 1).sum(),
        )

        self.assertTrue(
            set(X_bal[:, 0]).issubset(
                set(X[:, 0])
            )
        )


if __name__ == "__main__":
    unittest.main()
