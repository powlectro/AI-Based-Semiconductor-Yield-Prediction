"""Sanity checks against the original UCI SECOM files."""
import unittest
from pathlib import Path

from src.secom_io import load_secom

ROOT = Path(__file__).resolve().parents[1]


class TestSecomLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.timestamps = load_secom(ROOT / "data" / "raw" / "secom.zip")

    def test_shapes(self):
        self.assertEqual(self.X.shape, (1567, 590))
        self.assertEqual(len(self.y), 1567)
        self.assertEqual(len(self.timestamps), 1567)

    def test_label_encoding(self):
        self.assertEqual(int((self.y == 0).sum()), 1463)
        self.assertEqual(int((self.y == 1).sum()), 104)
        self.assertEqual(set(self.y.unique()), {0, 1})

    def test_missing_values_and_timestamp(self):
        self.assertTrue(self.X.isna().any().any())
        self.assertEqual(self.timestamps.iloc[0].strftime("%d/%m/%Y %H:%M:%S"),
                         "19/07/2008 11:55:00")


if __name__ == "__main__":
    unittest.main()
