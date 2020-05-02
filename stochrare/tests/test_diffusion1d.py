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

    def test_integrate_sde(self):
        oup = diffusion1d.OrnsteinUhlenbeck1D(0, 1, 1)
        dt = 0.01
        t = np.linspace(0, 1, 100, dtype=np.float32)
        x = np.full(100, 0, dtype=np.float32)
        w = np.random.normal(0, np.sqrt(dt), size=99)
        traj_euler_oup = oup.integrate_sde(x, t, w, method='euler', dt=0.01)
        traj_euler_const = diffusion1d.ConstantDiffusionProcess1D.integrate_sde(oup, x, t, w,
                                                                                method='euler',
                                                                                dt=0.01)
        traj_euler_diff = diffusion1d.DiffusionProcess1D.integrate_sde(oup, x, t, w, method='euler',
                                                                       dt=0.01)
        np.testing.assert_allclose(traj_euler_oup, traj_euler_const)
        np.testing.assert_allclose(traj_euler_oup, traj_euler_diff)
        traj_milstein_oup = oup.integrate_sde(x, t, w, method='milstein', dt=0.01)
        traj_milstein_const = diffusion1d.ConstantDiffusionProcess1D.integrate_sde(oup, x, t, w,
                                                                                   method='milstein',
                                                                                   dt=0.01)
        traj_milstein_diff = diffusion1d.DiffusionProcess1D.integrate_sde(oup, x, t, w,
                                                                          method='milstein', dt=0.01)
        np.testing.assert_allclose(traj_milstein_oup, traj_milstein_const)
        np.testing.assert_allclose(traj_milstein_oup, traj_milstein_diff)
        traj_gillespie_oup = oup.integrate_sde(x, t, w, method='gillespie', dt=0.01)
        np.testing.assert_allclose(traj_gillespie_oup, traj_milstein_oup)
        self.assertRaises(NotImplementedError, oup.integrate_sde, x, t, w, method='rk', dt=0.01)

    def test_trajectory(self):
        dt_brownian = 1e-5
        model = diffusion1d.DiffusionProcess1D(lambda x, t: 2*x, lambda x, t: x, deterministic=True)
        wiener = diffusion1d.Wiener1D(D=0.5, deterministic=True)
        brownian_path = wiener.trajectory(0., 0., T=0.1, dt=dt_brownian)
        traj_exact = np.exp(1.5*brownian_path[0]+brownian_path[1])
        traj1 = model.trajectory(1., 0., T=0.1, dt=dt_brownian, brownian_path=brownian_path)
        traj2 = model.trajectory(1., 0., T=0.1, dt=dt_brownian, deltat=dt_brownian)
        traj3 = model.trajectory(1., 0., T=0.1, dt=dt_brownian, brownian_path=brownian_path,
                                 method='milstein')
        np.testing.assert_allclose(traj1[1], traj2[1], rtol=1e-5)
        np.testing.assert_allclose(traj1[1], traj_exact, rtol=1e-2)
        np.testing.assert_allclose(traj3[1], traj_exact, rtol=1e-5)

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
