"""
Unit tests for the firstpassage module.
"""
import unittest
import numpy as np
import stochrare.firstpassage as firstpassage
from stochrare.dynamics.diffusion1d import ConstantDiffusionProcess1D

class TestFirstPassage(unittest.TestCase):
    def setUp(self):
        deterministic_drift = ConstantDiffusionProcess1D(lambda x, t: 1, 0)
        self.fpt = firstpassage.FirstPassageProcess(deterministic_drift)
        self.dt = 0.01

    def test_firstpassagetime(self):
        for A in range(10):
            t = self.fpt.firstpassagetime(0, 0, A, dt=self.dt)
            np.testing.assert_allclose(t, A, atol=self.dt)

    def test_escapetime_sample(self):
        for A in range(10):
            tarray = self.fpt.escapetime_sample(0, 0, A, dt=self.dt, ntraj=100)
            np.testing.assert_allclose(tarray, np.full_like(tarray, A), atol=self.dt)

    def test_escapetime_avg(self):
        for A in range(10):
            tmean = self.fpt.escapetime_avg(0, 0, A, dt=self.dt, ntraj=100)
            np.testing.assert_allclose(tmean, A, atol=(self.dt+1.e-5))


if __name__ == "__main__":
    unittest.main()
