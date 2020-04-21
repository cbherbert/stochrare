"""
Unit tests for the diffusion module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion as diffusion

class TestDynamics(unittest.TestCase):
    def test_wiener_potential(self):
        data = np.ones(10)
        np.testing.assert_array_equal(diffusion.Wiener(1).potential(data), np.zeros_like(data))
        data = np.ones((10, 10))
        np.testing.assert_array_equal(diffusion.Wiener(2).potential(data), np.zeros_like(data))

    def test_update(self):
        dimension = 2
        wiener = diffusion.Wiener(dimension, D=0.5)
        dw = np.random.normal(size=dimension)
        np.testing.assert_array_equal(wiener.update(np.zeros(dimension), 0, dw=dw), dw)

if __name__ == "__main__":
    unittest.main()
