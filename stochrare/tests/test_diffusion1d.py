"""
Unit tests for the diffusion1d module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion1d as diffusion1d

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
    def test_instantoneq(self):
        oup = diffusion1d.OrnsteinUhlenbeck1D(0, 1, 1)
        xvec = np.linspace(-1, 1)
        pvec = np.linspace(-1, 1)
        for x, p in zip(xvec, pvec):
            Y = np.array([x, p])
            eq_const = diffusion1d.ConstantDiffusionProcess1D._instantoneq(oup, 0, Y)
            jac_const = diffusion1d.ConstantDiffusionProcess1D._instantoneq_jac(oup, 0, Y)
            eq_diff = diffusion1d.DiffusionProcess1D._instantoneq(oup, 0, Y)
            jac_diff = diffusion1d.DiffusionProcess1D._instantoneq_jac(oup, 0, Y)
            np.testing.assert_allclose(oup._instantoneq(0, Y), eq_const)
            np.testing.assert_allclose(oup._instantoneq_jac(0, Y), jac_const, atol=1e-5)
            np.testing.assert_allclose(oup._instantoneq(0, Y), eq_diff)
            np.testing.assert_allclose(oup._instantoneq_jac(0, Y), jac_diff, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
