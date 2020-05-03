"""
Unit tests for the diffusion module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion as diffusion

class TestDynamics(unittest.TestCase):

    def test_properties(self):
        oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2)
        self.assertEqual(oup.D0, 1)
        self.assertEqual(oup.mu, 0)
        self.assertEqual(oup.theta, 1)
        np.testing.assert_allclose(oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[np.sqrt(2), 0], [0, np.sqrt(2)]]))
        oup.D0 = 0.5
        np.testing.assert_allclose(oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([-1, -1]))
        oup.theta = 2
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([-2, -2]))
        oup.mu = 1
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([0, 0]))


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
