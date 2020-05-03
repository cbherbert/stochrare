"""
Unit tests for the diffusion module.
"""
import unittest
import numpy as np
import stochrare.dynamics.diffusion as diffusion

class TestDynamics(unittest.TestCase):

    def test_properties(self):
        oup = diffusion.OrnsteinUhlenbeck(0, 1, 1, 2)
        self.assertEqual(oup.D0, 1)
        self.assertEqual(oup.mu, 0)
        self.assertEqual(oup.theta, 1)
        np.testing.assert_allclose(oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[np.sqrt(2), 0], [0, np.sqrt(2)]]))
        oup.D0 = 0.5
        np.testing.assert_allclose(oup.diffusion(np.array([1, 1]), 0),
                                   np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([-1, -1]))
        oup.theta = 2
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([-2, -2]))
        oup.mu = 1
        np.testing.assert_allclose(oup.drift(np.array([1, 1]), 0), np.array([0, 0]))

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

    def test_trajectory(self):
        dt_brownian = 1e-5
        model = diffusion.DiffusionProcess(lambda x, t: 2*x,
                                           lambda x, t: np.array([[x[0], 0], [0, x[1]]]),
                                           deterministic=True)
        wiener = diffusion.Wiener(2, D=0.5, deterministic=True)
        brownian_path = wiener.trajectory(np.array([0., 0.]), 0., T=0.1, dt=dt_brownian)
        traj_exact1 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 0])
        traj_exact2 = np.exp(1.5*brownian_path[0]+brownian_path[1][:, 1])
        traj = model.trajectory(np.array([1., 1.]), 0., T=0.1, dt=dt_brownian,
                                brownian_path=brownian_path)
        np.testing.assert_allclose(traj[1][:, 0], traj_exact1, rtol=1e-2)
        np.testing.assert_allclose(traj[1][:, 1], traj_exact2, rtol=1e-2)



if __name__ == "__main__":
    unittest.main()
