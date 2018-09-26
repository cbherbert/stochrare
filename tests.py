"""
Unit tests
"""
import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import stochpy.dynamics.stochastic as stochastic
import stochpy.timeseries as ts

class TestPotential(unittest.TestCase):
    def test_wiener_potential(self):
        data = np.ones(10)
        np.testing.assert_array_equal(stochastic.Wiener().potential(data, 0.), np.zeros_like(data))
        data = np.ones((10, 10))
        np.testing.assert_array_equal(stochastic.Wiener().potential(data, 0.), np.zeros_like(data))

class TestRareEvents(unittest.TestCase):
    def test_blockmaximum(self):
        data = np.random.random(100)
        data[20] = 2.0
        data[69] = 3.5
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='proba')),
                         [(3.5, 0.5), (2.0, 1.0)])
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='returntime')),
                         [(3.5, 100.), (2., 50.)])

if __name__ == "__main__":
    unittest.main()
