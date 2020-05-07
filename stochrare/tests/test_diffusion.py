"""
Unit tests for the diffusion module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion as diffusion
from unittest.mock import patch

class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2, deterministic=True)
        self.wiener = diffusion.Wiener(2, D=0.5, deterministic=True)

    def test_properties(self):
        self.assertEqual(self.oup.D0, 1)
        self.assertEqual(self.oup.mu, 0)
        self.assertEqual(self.oup.theta, 1)
        np.testing.assert_allclose(self.oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[np.sqrt(2), 0], [0, np.sqrt(2)]]))
        self.oup.D0 = 0.5
        np.testing.assert_allclose(self.oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([-1, -1]))
        self.oup.theta = 2
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([-2, -2]))
        self.oup.mu = np.array([1, 1])
        np.testing.assert_allclose(self.oup.drift(np.array([1, 1]), 0), np.array([0, 0]))
        self.oup.D0 = 1
        self.oup.theta = 1
        self.oup.mu = 0

    def test_wiener_potential(self):
        data = np.ones(10)
        np.testing.assert_array_equal(diffusion.Wiener.potential(data), np.zeros_like(data))
        data = np.ones((10, 10))
        np.testing.assert_array_equal(self.wiener.potential(data), np.zeros_like(data))

    def test_update(self):
        dw = np.random.normal(size=self.wiener.dimension)
        x = np.zeros(self.wiener.dimension)
        np.testing.assert_array_equal(self.wiener.update(x, 0, dw=dw), dw)

    @patch.object(diffusion.DiffusionProcess, "_euler_maruyama")
    def test_integrate_sde(self, mock_DiffusionProcess):
        x0=0;time=10;dt=0.1;
        num = int(time/dt)+1
        tarray = np.linspace(0, time, num=num)

        for dim in range(1,5):
            x = np.full((num, dim), x0)
            dw = np.random.normal(0, np.sqrt(dt), size=(num-1, dim))
            with self.subTest(dim=dim):
                model = diffusion.DiffusionProcess(
                    lambda x, t: 2*x,
                    lambda x, t: np.identity(dim),
                    deterministic=True,
                )
                model.integrate_sde(x, tarray, dw, dt=dt, method="euler")
                model._euler_maruyama.assert_called_with(x,
                                                         tarray,
                                                         dw,
                                                         dt,
                                                         model.drift,
                                                         model.diffusion
                )
        self.assertEqual(model._euler_maruyama.call_count, 4)


    def test_integrate_sde_wrong_method(self):
        model = diffusion.DiffusionProcess(
                    lambda x, t: 2*x,
                    lambda x, t: np.identity(dim),
                    deterministic=True,
                )
        with self.assertRaises(NotImplementedError):
            model.integrate_sde(0, 0, 0, method="fancy method")


    def test_trajectory(self):
        dt_brownian = 1e-5
        diff = lambda x, t: np.array([[x[0], 0], [0, x[1]]], dtype=np.float32)
        model = diffusion.DiffusionProcess(lambda x, t: 2*x, diff, 2, deterministic=True)
        brownian_path = self.wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
        traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
        traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])
        traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=dt_brownian,
                                brownian_path=brownian_path, precision=np.float32)
        np.testing.assert_allclose(traj[1][:, 0], traj_exact1, rtol=1e-2)
        np.testing.assert_allclose(traj[1][:, 1], traj_exact2, rtol=1e-2)

    def test_trajectory_generator(self):
        traj = np.array([x for t, x in self.oup.trajectory_generator(np.array([0, 0]), 0,
                                                                     100, dt=0.01)])
        _, x = self.oup.trajectory(np.array([0, 0]), 0, dt=0.01, T=1)
        np.testing.assert_allclose(x, traj, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
