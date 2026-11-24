import unittest
import numpy as np
from src.fixed_point import (
    fractional_bits,
    make_fixed_model,
    quantize,
    quantized_linear_predict,
    sequential_mac_predict,
    to_twos_hex,
)


class LinearModel:
    coef_ = np.array([[0.75, -0.5]])
    intercept_ = np.array([-0.2])


class FixedPointTests(unittest.TestCase):
    def test_twos_complement(self):
        self.assertEqual(to_twos_hex(-1, 8), 'ff')
        self.assertEqual(to_twos_hex(-128, 8), '80')
        with self.assertRaises(ValueError):
            to_twos_hex(128, 8)

    def test_saturation_and_round_to_even(self):
        np.testing.assert_array_equal(quantize([500, -500, 0.5, 1.5, 2.5], 8, 0), [127, -128, 0, 2, 2])

    def test_linear_boundary(self):
        model = LinearModel()
        train = np.array([[-1., 0], [0, 1.], [1, -1]])
        spec = make_fixed_model(model, train, 16, 0.5)
        data = np.array([[1, 0], [0, 1], [0, 0], [-1, 0]], dtype=float)
        pred = quantized_linear_predict(quantize(data, 16, spec['xfrac']), spec['weights'], spec['offset'])
        np.testing.assert_array_equal(pred, [True, False, False, False])

    def test_dimension_error(self):
        with self.assertRaises(ValueError):
            quantized_linear_predict(np.zeros((2, 3)), [1, 2], 0)

    def test_sequential_mac_matches_parallel_dot(self):
        # Verify one-MAC-per-cycle accumulation gives identical decisions to the matrix dot product.
        rng = np.random.default_rng(0)
        data = rng.integers(-40, 40, size=(24, 8), dtype=np.int64)
        weights = rng.integers(-15, 15, size=8, dtype=np.int64)
        np.testing.assert_array_equal(
            sequential_mac_predict(data, weights, -7),
            quantized_linear_predict(data, weights, -7),
        )

    def test_scale(self):
        self.assertGreaterEqual(fractional_bits(10, 16), 0)
        with self.assertRaises(ValueError):
            fractional_bits(-1, 8)


if __name__ == '__main__':
    unittest.main()
