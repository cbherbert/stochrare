"""
Unit tests for the instanton module.
"""
import unittest
import numpy as np
from stochrare.dynamics.diffusion1d import OrnsteinUhlenbeck1D
from stochrare.rare.instanton import InstantonSolver

class TestInstantonOU1D(unittest.TestCase):
    def setUp(self):
        model = OrnsteinUhlenbeck1D(0, 1, 0.1)
        self.theta = model.theta
        self.solver = InstantonSolver(model)

    def test_instantonivp(self):
        times = np.linspace(0, 10)
        for x0, p0 in ((0, 1), (1, 0)):
            t, x, p = self.solver.instanton_ivp(x0, p0, *times,
                                                solver='odeint', rtol=1e-8, atol=1e-9)
            xtrue = 2*p0/self.theta*np.sinh(self.theta*t)+x0*np.exp(-self.theta*t)
            ptrue = p0*np.exp(self.theta*t)
            np.testing.assert_allclose(x, xtrue, rtol=1e-6)
            np.testing.assert_allclose(p, ptrue, rtol=1e-6)

    def test_instantonbvp(self):
        times = np.linspace(0, 10)
        for x0, p0 in ((0, 1), (1, 0)):
            xtrue = 2*p0/self.theta*np.sinh(self.theta*times)+x0*np.exp(-self.theta*times)
            ptrue = p0*np.exp(self.theta*times)
            xfinal = xtrue[-1]
            t, x, p = self.solver.instanton_bvp(x0, xfinal, *times, tol=1e-8)
            np.testing.assert_allclose(t, times)
            np.testing.assert_allclose(x, xtrue, rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(p, ptrue, rtol=1e-6, atol=1e-9)

if __name__ == "__main__":
    unittest.main()
