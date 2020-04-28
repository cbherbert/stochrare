"""
Unit tests for the fokkerplanck module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion1d as diffusion1d
import stochrare.fokkerplanck as fp
from stochrare import edpy

class TestFokkerPlanck(unittest.TestCase):
    def setUp(self):
        self.X0 = np.linspace(-20, 20, 400)
        self.P0 = fp.FokkerPlanck1DAbstract.dirac1d(0, self.X0)
        self.wiener = diffusion1d.Wiener1D()
        self.fpe = fp.FokkerPlanck1D.from_sde(self.wiener)
        self.args = {'dt': 0.001, 'npts': 400, 'bounds': (-20., 20.),
                     'bc': ('absorbing', 'absorbing')}

    def test_exceptions(self):
        fdgrid = edpy.RegularCenteredFD(-20, 20, 400)
        self.assertRaises(NotImplementedError, fp.FokkerPlanck1DAbstract._fpeq,
                          self.fpe, self.P0, self.X0, 0)
        self.assertRaises(NotImplementedError, fp.FokkerPlanck1DAbstract._fpmat,
                          self.fpe, self.X0, 0)
        self.assertRaises(NotImplementedError, fp.FokkerPlanck1DAbstract._fpbc, self.fpe, fdgrid)
        self.assertRaises(NotImplementedError, self.fpe._fpbc, fdgrid, ('strange', 'bc'))

    def test_fromsde(self):
        fpe = fp.FokkerPlanck1D(lambda x, t: 0, lambda x, t: 1)
        np.testing.assert_allclose(fpe.drift(self.X0, 0), self.fpe.drift(self.X0, 0))
        np.testing.assert_allclose(fpe.diffusion(self.X0, 0), self.fpe.diffusion(self.X0, 0))

    def test_fpintegrate_generator(self):
        self.args['method'] = 'explicit'
        P0 = self.fpe.gaussian1d(0, 0.5, self.X0)
        t1, X1, P1 = self.fpe.fpintegrate(0, 1, P0=P0, **self.args)
        t2, X2, P2 = self.fpe.fpintegrate(1, 1, P0=P1, **self.args)
        gen = self.fpe.fpintegrate_generator(1, 2, t0=0, P0=P0, **self.args)
        for (tgen, Xgen, Pgen), (tt, Xt, Pt) in zip(gen, ((t1, X1, P1), (t2, X2, P2))):
            np.testing.assert_allclose(tgen, tt)
            np.testing.assert_allclose(Xgen, Xt, atol=1e-4)
            np.testing.assert_allclose(Pgen, Pt, atol=1e-4)

    def test_fpsolver_uniform(self):
        P0 = self.fpe.uniform1d(self.X0)
        _, _, P = self.fpe.fpintegrate(0, 1, dt=0.001, npts=400, bounds=(-20, 20), P0=P0,
                                       bc=('reflecting', 'reflecting'), method='explicit')
        np.testing.assert_allclose(P, P0)

    def test_fpsolver_explicit_heat(self):
        t, X, P = self.fpe.fpintegrate(0, 10, dt=0.001, npts=400, bounds=(-20., 20.), P0=self.P0,
                                       bc=('absorbing', 'absorbing'), method='explicit')
        np.testing.assert_allclose(P, self.wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_implicit_heat(self):
        t, X, P = self.fpe.fpintegrate(0, 10, dt=0.05, npts=400, bounds=(-20., 20.), P0=self.P0,
                                  bc=('absorbing', 'absorbing'), method='implicit')
        np.testing.assert_allclose(P, self.wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_cranknicolson_heat(self):
        t, X, P = self.fpe.fpintegrate(0, 10, dt=0.025, npts=400, bounds=(-20., 20.), P0=self.P0,
                                  bc=('absorbing', 'absorbing'), method='cn')
        np.testing.assert_allclose(P, self.wiener._fpthsol(X, t), atol=1e-3)

class TestShortTimePropagator(unittest.TestCase):
    def setUp(self):
        self.wiener = diffusion1d.Wiener1D()
        self.fpe = fp.ShortTimePropagator(self.wiener.drift,
                                          lambda x, t: 0.5*self.wiener.diffusion(x, t)**2, 0.1)

    def test_transition_matrix(self):
        grid = np.linspace(-20, 20, 400)
        pst1 = self.fpe.transition_matrix(grid, 0)
        pst2 = np.array([[self.fpe.transition_probability(x, x0, 0) for x0 in grid] for x in grid])
        np.testing.assert_allclose(pst1, pst2)

    def test_fpsolver_shorttime_heat(self):
        P0 = fp.FokkerPlanck1DAbstract.dirac1d(0, np.linspace(-20, 20, 400))
        t, X, P = self.fpe.fpintegrate(0, 10, npts=400, bounds=(-20., 20.), P0=P0)
        np.testing.assert_allclose(P, self.wiener._fpthsol(X, t), atol=1e-3)
        t2, X2, P2 = self.fpe.fpintegrate_naive(0, 10, npts=400, bounds=(-20., 20.), P0=P0)
        np.testing.assert_allclose(P, P2, atol=1e-3)

if __name__ == "__main__":
    unittest.main()
