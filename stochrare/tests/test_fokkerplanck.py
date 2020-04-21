"""
Unit tests for the fokkerplanck module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion1d as diffusion1d
import stochrare.fokkerplanck as fp

class TestFokkerPlanck(unittest.TestCase):
    def test_fpsolver_explicit_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D.from_sde(wiener)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.001, npts=400, bounds=(-20., 20.),
                                  P0=fpe.dirac1d(0, np.linspace(-20, 20, 400)),
                                  bc=('absorbing', 'absorbing'), method='explicit')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_implicit_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D.from_sde(wiener)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.05, npts=400, bounds=(-20., 20.),
                                  P0=fpe.dirac1d(0, np.linspace(-20, 20, 400)),
                                  bc=('absorbing', 'absorbing'), method='implicit')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_cranknicolson_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D.from_sde(wiener)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.025, npts=400, bounds=(-20., 20.),
                                  P0=fpe.dirac1d(0, np.linspace(-20, 20, 400)),
                                  bc=('absorbing', 'absorbing'), method='cn')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

class TestShortTimePropagator(unittest.TestCase):
    def test_transition_matrix(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.ShortTimePropagator(wiener.drift, lambda x, t: 0.5*wiener.diffusion(x, t)**2, 0.1)
        grid = np.linspace(-20, 20, 400)
        pst1 = fpe.transition_matrix(grid, 0)
        pst2 = np.array([[fpe.transition_probability(x, x0, 0) for x0 in grid] for x in grid])
        np.testing.assert_allclose(pst1, pst2)

    def test_fpsolver_shorttime_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.ShortTimePropagator(wiener.drift, lambda x, t: 0.5*wiener.diffusion(x, t)**2, 0.1)
        P0 = fp.FokkerPlanck1DAbstract.dirac1d(0, np.linspace(-20, 20, 400))
        t, X, P = fpe.fpintegrate(0, 10, npts=400, bounds=(-20., 20.), P0=P0)
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

if __name__ == "__main__":
    unittest.main()
