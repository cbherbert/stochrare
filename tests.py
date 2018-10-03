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
import stochpy.ams as ams

class TestPotential(unittest.TestCase):
    def test_wiener_potential(self):
        data = np.ones(10)
        np.testing.assert_array_equal(stochastic.Wiener().potential(data, 0.), np.zeros_like(data))
        data = np.ones((10, 10))
        np.testing.assert_array_equal(stochastic.Wiener().potential(data, 0.), np.zeros_like(data))

class TestRareEvents(unittest.TestCase):
    def test_blockmaximum(self):
        data = np.random.random(101)
        data[20] = 2.0
        data[69] = 3.5
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='proba')),
                         [(3.5, 0.5), (2.0, 1.0)])
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='returntime')),
                         [(3.5, 100.), (2., 50.)])

class TestAMS(unittest.TestCase):

    def test_getlevel(self):
        data = np.random.random(100)
        data[10] = 2.
        algo = ams.TAMS(None, (lambda t, x: x), 10.)
        self.assertEqual(algo.getlevel(np.arange(100), data), 2.0)

    def test_crossingtime(self):
        data = np.random.random(100)
        data[10] = 2.
        algo = ams.TAMS(None, (lambda t, x: x), 10.)
        self.assertEqual(algo.getcrossingtime(1.5, np.arange(100), data), (10, 2.0))

    def test_selectams(self):
        levels = np.array([0.5, 1.1, 0.2, 0.6, 0.2, 0.5])
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels[:-2], npart=1)))[0], (2, ))
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels, npart=1)))[0], (2, 4))
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels[:-2], npart=2)))[0], (0, 2))
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels[:-1], npart=2)))[0], (0, 2, 4))
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels, npart=2)))[0], (0, 2, 4, 5))
        self.assertEqual(zip(*list(ams.TAMS.selectionstep(levels, npart=3)))[0], (0, 2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
