"""
Unit tests for the utils package.
"""
import unittest
import numpy as np
from stochrare.utils import pseudorand, one_d_method


class TestUtils(unittest.TestCase):
    __deterministic__ = True

    @pseudorand
    def test_pseudorand(self):
        results = np.array(
            [
                0.54340494,
                0.27836939,
                0.42451759,
                0.84477613,
                0.00471886,
            ]
        )
        np.testing.assert_allclose(np.random.random(5), results, atol=1e-8)

    def test_one_d_method(self):
        class mock_class:
            def __init__(self, dimension):
                self.dimension = dimension

            @one_d_method
            def method(self):
                return False

        self.assertFalse(mock_class(1).method())
        with self.assertRaises(NotImplementedError):
            mock_class(2).method()


if __name__ == "__main__":
    unittest.main()
