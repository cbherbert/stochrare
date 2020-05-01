"""
Unit tests for the instanton module.
"""
import unittest
import numpy as np
import warnings
from stochrare.dynamics.diffusion1d import OrnsteinUhlenbeck1D
from stochrare.dynamics.diffusion import OrnsteinUhlenbeck
from stochrare.rare.instanton import InstantonSolver

class TestInstantonOU1D(unittest.TestCase):
    def setUp(self):
        model = OrnsteinUhlenbeck1D(0, 1, 0.1)
        self.theta = model.theta
        self.solver = InstantonSolver(model)

    def test_filtfun(self):
        t0 = np.arange(11)
        x0 = np.arange(10)
        p0 = np.arange(11)
        t, x, p = self.solver.filt_fun(t0[:-1], x0, p0[:-1], threshold=10)
        np.testing.assert_allclose(t, t0[:-1])
        np.testing.assert_allclose(x, x0)
        np.testing.assert_allclose(p, p0[:-1])
        x0 = np.append(x0, np.inf)
        t, x, p = self.solver.filt_fun(t0, x0, p0, threshold=10)
        np.testing.assert_allclose(t, t0[:-1])
        np.testing.assert_allclose(x, x0[:-1])
        np.testing.assert_allclose(p, p0[:-1])

    def test_instantonivp(self):
        times = np.linspace(0, 10)
        for x0, p0 in ((0, 1), (1, 0)):
            t, x, p = self.solver.instanton_ivp(x0, p0, *times,
                                                solver='odeint', rtol=1e-8, atol=1e-9)
            xtrue = 2*p0/self.theta*np.sinh(self.theta*t)+x0*np.exp(-self.theta*t)
            ptrue = p0*np.exp(self.theta*t)
            np.testing.assert_allclose(x, xtrue, rtol=1e-6)
            np.testing.assert_allclose(p, ptrue, rtol=1e-6)

        for x0, p0 in ((0, 1), (1, 0)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, x, p = self.solver.instanton_ivp(x0, p0, *times,
                                                    solver='odeclass', integrator='dopri5')
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

    @unittest.skip("Instanton equations in arbitrary dimensions not yet merged")
    def test_exception1d(self):
        solver = InstantonSolver(OrnsteinUhlenbeck(0, 1, 0.1, 2))
        times = np.linspace(0, 10)
        self.assertRaises(NotImplementedError, solver.instanton_bvp(0, 1, *times))

if __name__ == "__main__":
    unittest.main()
