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

if __name__ == "__main__":
    unittest.main()
