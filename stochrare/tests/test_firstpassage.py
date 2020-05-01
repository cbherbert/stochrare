"""
Unit tests for the firstpassage module.
"""
import unittest
import numpy as np
from scipy.special import erfc
import stochrare.firstpassage as firstpassage
from stochrare.dynamics.diffusion1d import ConstantDiffusionProcess1D, Wiener1D, OrnsteinUhlenbeck1D

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

    def test_firstpassagetime_cdf(self):
        oup = OrnsteinUhlenbeck1D(0, 1, 0.1)
        tau = firstpassage.FirstPassageProcess(oup)
        _, pFP = tau.firstpassagetime_cdf(0, 0.5, *np.linspace(0, 5), dt=0.1, npts=700,
                                          bounds=(-3.0, 0.5), method='cn')
        _, pFPb = tau.firstpassagetime_cdf_adjoint(0, 0.5, *np.linspace(0, 5), dt=0.1, npts=700,
                                                   bounds=(-3.0, 0.5), method='cn')
        np.testing.assert_allclose(pFP, pFPb, rtol=1e-2)

    @unittest.skip("Source of disagreement with analytical solution should be found")
    def test_firstpassagetime_cdf_exact(self):
        wiener = Wiener1D()
        fpt = firstpassage.FirstPassageProcess(wiener)
        x0 = 0
        a = 1
        times = np.linspace(0, 1, num=10)
        _, G = fpt.firstpassagetime_cdf(x0, a, *times)
        np.testing.assert_allclose(G, 0.5*erfc((x0-a)/np.sqrt(4*wiener.D0*times)))

    def test_firstpassagetime_avg_theory(self):
        oup = OrnsteinUhlenbeck1D(0, 1, 0.1)
        fpt = firstpassage.FirstPassageProcess(oup)
        x0 = 0
        a = 0.5
        tau_th = fpt.firstpassagetime_avg_theory(x0, a, inf=-3, num=100)[1]
        tau_th2 = fpt.firstpassagetime_avg_theory2(x0, a, inf=-3, num=350)[1]
        tau_thoup = oup.mean_firstpassage_time(x0, a)
        np.testing.assert_allclose(tau_th, tau_thoup, rtol=1e-4)
        np.testing.assert_allclose(tau_th2, tau_thoup, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()
