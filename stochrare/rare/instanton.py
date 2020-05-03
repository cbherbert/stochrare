"""
Module for instanton computations.

This relies on the path integral formalism for stochastic processes.
The general idea is that sample paths have a probability proportional to a factor e^{-S}, where S is
the action. The expectation value of some observable can therefore be expressed as an integral over
all possible paths, weighted by this factor.
This formalism was developed initially in the following references:
- L. Onsager and S. Machlup, Fluctuations and irreversible processes, Phys. Rev. 91, 1505 (1953).
- S. Machlup and L. Onsager, Fluctuations and irreversible processes: II. systems with kinetic
  energy, Phys. Rev. 91, 1512 (1953).
on the one hand and
- P. C. Martin, E. D. Siggia and H. A. Rose, Statistical dynamics of classical systems,
  Phys. Rev. A 8, 423 (1973).
- H. K. Janssen, On a Lagrangian for classical field dynamics and renormalization group calculations
  of dynamical critical properties, Z. Phys. B 23, 377 (1976).
- C. de Dominicis, Techniques de renormalisation de la theorie des champs et dynamique des
  phenomenes critiques, J. Phys. C: Solid State Phys. 1, 247 (1976).
on the other hand.

The path integral may be evaluated using a saddle-point approximation.
The instanton is the corresponding action minimizer.
On other words, it is the realization of the stochastic process which dominates the path integral.
The accurracy of this approximation is usually governed by a small parameter.
Such a parameter appears naturally in the Freidlin-Wentzell theory of large deviations.
It considers dynamical systems perturbed by noise.
When the amplitude of the noise goes is weak, the probability of an event, depending on the
stochastic process, satisfies a large deviation pinciple.
The rate function in this large deviation principle corresponds to the action.
See
- M. I. Freidlin and A. D. Wentzell, Random Perturbations of Dynamical Systems, Springer (1998)
for more details.

To sum up, considering a rare event for a random dynamical system, the instanton corresponds to
the typical path which realizes the event.

Instantons can be computed directly by minimizing numerically the action or by solving the
corresponding Hamilton equations. However, in practice these methods are very inefficient, except
for low dimensional systems.
Many more sophisticated methods have been designed for numerical computation of instantons,
for instance:

* String method for gradient systems
- W. E, W. Ren and E. Vanden-Eijnden, String method for the study of rare events, Phys. Rev. B 66,
  052301 (2002)
- W. Ren, W. E and E. Vanden-Eijden, Simplified and improved string method for computing the minimum
  energy paths in barrier-crossing events, J. Chem. Phys. 126, 164103 (2007).

* Minimum action method
- W. E, W. Ren and E. Vanden-Eijnden, Minimum action method for the study of rare events,
  Commun. Pure Appl. Math. 57, 637 (2004)
and related papers:
- R. Olender and R. Elber, Calculation of classical trajectories with a very large time step:
  formalism and numerical exemples, J. Chem. Phys. 105, 9299 (1996).
- X. Zhou, W. Ren and W. E, Adaptive minimum action method for the study of rare events,
  J. Chem. Phys. 128, 104111 (2008).
- W. E and X. Zhou, The gentlest ascent dynamics, Nonlinearity 24, 1831 (2011).
- X. Wan and G. Lin, Hybrid parallel computing of minimum action method, Parallel Comput. 39, 638 (2013).
- X. Wan, An adaptive high-order minimum action method, J. Comput. Phys. 230, 86669 (2011).
- M. Heymann and E. Vanden-Eijden, The geometric minimum action method: a least action principle on
  the space of curves, Commun. Pure Appl. Math. 61, 1052 (2008).
- E. Vanden-Eijden and M. Heymann, The geometric minimum action method for computing minimum energy
  paths, J. Chem. Phys. 128, 061103 (2008).
- M. Heymann and E. Vanden-Eijden, Pathways of maximum likelihood for rare events in nonequilibrium
  systems: application to nucleation in the presence of shear, Phys. Rev. Lett. 100, 140601 (2008).

* Chernykh-Stepanov algorithm
- A. I. Chernykh and M. G. Stepanov, Large negative velocity gradients in Burgers turbulence,
  Phys. Rev. E 64, 026306 (2001)


Instanton computation methods are also reviewed in
- T. Grafke, R. Grauer and T. Schafer, The instanton method and its numerical implementation in
fluid mechanics, J. Phys. A: Math. Theor. 48, 333001 (2015).

For now, this module only implements direct instanton computation by solving the Hamilton equations.
More efficient methods shall be added to the module in the future.

"""
import numpy as np
import scipy.integrate
from ..dynamics.diffusion1d import DiffusionProcess1D

class InstantonSolver:
    """
    Basic class for solving the instanton equations.
    """
    def __init__(self, model):
        self.model = model

    @property
    def instanton_eq(self):
        return self.model._instantoneq

    @property
    def instanton_jac(self):
        return self.model._instantoneq_jac

    @staticmethod
    def filt_fun(t, x, p, threshold=100):
        """
        Filter instanton trajectories to remove blowup.
        """
        filt = (x > threshold).nonzero()[0]
        if len(filt) > 0:
            maxind = filt[0]
        else:
            maxind = None
        return t[:maxind], x[:maxind], p[:maxind]


    def instanton_ivp(self, x0, p0, *args, **kwargs):
        """
        Solve the instanton equations as an initial value problem.
        x0 and p0 are the initial conditions.
        args is a sequence of time points for which to compute the position and momentum.

        Return the instanton trajectory (t, x, p).
        """
        solver = kwargs.pop('solver', 'odeint')
        scheme = kwargs.pop('integrator', 'dopri5')
        times = np.sort(args)
        init_cond = np.concatenate((np.atleast_1d(x0), np.atleast_1d(p0)))
        if solver == 'odeint':
            traj = scipy.integrate.odeint(self.instanton_eq, init_cond, times,
                                          Dfun=self.instanton_jac, tfirst=True, **kwargs)
        elif solver == 'odeclass':
            integ = scipy.integrate.ode(self.instanton_eq,
                                        jac=self.instanton_jac).set_integrator(scheme, **kwargs)
            integ.set_initial_value(init_cond, t=times[0])
            traj = np.array([integ.integrate(t) for t in times])
        x, p = np.hsplit(traj, 2)
        return times, np.squeeze(x), np.squeeze(p)

    def instanton_bvp(self, xstart, xend, *args, **kwargs):
        """
        Solve the instanton equations as a boundary value problem.
        xstart and xend are the initial and final conditions.
        Return the instanton trajectory (t, x, p).
        """
        if DiffusionProcess1D not in self.model.__class__.__mro__:
            raise NotImplementedError('Boundary value instanton problem solver not implemented in arbitrary dimension')
        def bc(Ya, Yb):
            return (Ya[0]-xstart, Yb[0]-xend)
        times = np.sort(args)
        res = scipy.integrate.solve_bvp(self.instanton_eq, bc, times,
                                        (np.linspace(xstart, xend, num=len(times)),
                                         np.zeros_like(times)), **kwargs)
        return times, res.sol(times)[0], res.sol(times)[1]
