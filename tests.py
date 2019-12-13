"""
Unit tests
"""
import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import stochrare.dynamics.diffusion as diffusion
import stochrare.dynamics.diffusion1d as diffusion1d
import stochrare.fokkerplanck as fp
import stochrare.timeseries as ts
import stochrare.rare.ams as ams

class TestStochastic(unittest.TestCase):
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

class TestDynamics1D(unittest.TestCase):
    def test_update(self):
        wiener = diffusion1d.Wiener1D(D=0.5)
        dw = np.random.normal()
        self.assertEqual(wiener.update(0, 0, dw=dw), dw)

    def test_trajectory(self):
        dt_brownian = 0.001
        model = diffusion1d.DiffusionProcess1D(lambda x, t: 2*x, lambda x, t: x, deterministic=True)
        wiener = diffusion1d.Wiener1D(D=0.5, deterministic=True)
        brownian_path = wiener.trajectory(0., 0., T=1, dt=dt_brownian)
        np.testing.assert_allclose(model.trajectory(1., 0., T=1, dt=0.001,
                                                    brownian_path=brownian_path)[1],
                                   model.trajectory(1., 0., T=1, dt=0.001, deltat=dt_brownian)[1],
                                   rtol=1e-5)

class TestFokkerPlanck(unittest.TestCase):
    def test_fpsolver_explicit_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D(wiener.drift, wiener.D0)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.001, npts=400, bounds=(-20., 20.),
                                  P0='dirac', bc=('absorbing', 'absorbing'), method='explicit')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_implicit_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D(wiener.drift, wiener.D0)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.05, npts=400, bounds=(-20., 20.),
                                  P0='dirac', bc=('absorbing', 'absorbing'), method='implicit')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

    def test_fpsolver_cranknicolson_heat(self):
        wiener = diffusion1d.Wiener1D()
        fpe = fp.FokkerPlanck1D(wiener.drift, wiener.D0)
        t, X, P = fpe.fpintegrate(0, 10, dt=0.025, npts=400, bounds=(-20., 20.),
                                  P0='dirac', bc=('absorbing', 'absorbing'), method='cn')
        np.testing.assert_allclose(P, wiener._fpthsol(X, t), atol=1e-3)

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

    def test_resample(self):
        told = np.linspace(0, 1, 101)
        xold = np.random.random(101)
        algo = ams.TAMS(diffusion1d.OrnsteinUhlenbeck1D(0, 1, 0.5), (lambda t, x: x), 1.)
        tnew, xnew = algo.resample(told[51], xold[51], told, xold, dt=0.01)
        np.testing.assert_allclose(xold[:52], xnew[:52])
        np.testing.assert_allclose(told, tnew)

    def test_selectams(self):
        levels = np.array([0.5, 1.1, 0.2, 0.6, 0.2, 0.5])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels[:-2], npart=1)[0], [2])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels, npart=1)[0], [2, 4])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels[:-2], npart=2)[0], [0, 2])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels[:-1], npart=2)[0], [0, 2, 4])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels, npart=2)[0], [0, 2, 4, 5])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels, npart=3)[0], [0, 2, 3, 4, 5])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels, npart=4)[0], [])
        np.testing.assert_array_equal(ams.TAMS.selectionstep(levels, npart=5)[0], [])

    def test_initialize(self):
        algo = ams.TAMS(diffusion1d.OrnsteinUhlenbeck1D(0, 1, 0.5), (lambda t, x: x), 1.)
        algo.initialize_ensemble(10, dt=0.01)
        self.assertEqual(algo._weight, 1)
        self.assertEqual(algo._levels.size, 10)
        for ind in range(10):
            np.testing.assert_allclose(algo._ensemble[ind][0], np.linspace(0., 1.0, num=101))
            self.assertEqual(algo._ensemble[ind][0].size, algo._ensemble[ind][1].size)

    def test_mutateams(self):
        algo = ams.TAMS(diffusion1d.OrnsteinUhlenbeck1D(0, 1, 0.5), (lambda t, x: x), 1.)
        algo.initialize_ensemble(10, dt=0.01)
        kill, survive = algo.selectionstep(algo._levels)
        algo.mutationstep(kill, survive, dt=0.01)
        self.assertEqual(algo._weight, 1-float(kill.size)/10)


if __name__ == "__main__":
    unittest.main()
