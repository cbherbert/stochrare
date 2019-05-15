"""
Simulating 1D diffusion processes
=================================

.. currentmodule:: stochpy.dynamics.diffusion1d

This module defines the `DiffusionProcess1D` class, representing diffusion processes with
arbitrary drift and diffusion coefficients in 1D.

This class can be subclassed for specific diffusion processes for which methods can be specialized,
both to simplify the code (e.g. directly enter analytical formulae when they are available) and for
performance.
As an exemple of this mechanism, we also provide in this module the `ConstantDiffusionProcess1D`
class, for which the diffusion term is constant, the `OrnsteinUhlenbeck1D` class representing the
particular case of the Ornstein-Uhlenbeck process, and the `Wiener1D` class corresponding to
Brownian motion.
These classes form a hierarchy deriving from the base class, `DiffusionProcess1D`.

.. autoclass:: DiffusionProcess1D
   :members:

.. autoclass:: ConstantDiffusionProcess1D
   :members:

.. autoclass:: OrnsteinUhlenbeck1D
   :members:

.. autoclass:: Wiener1D
   :members:

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.misc import derivative
from .. import edpy
from .. import fokkerplanck as fp
from ..utils import pseudorand


class DiffusionProcess1D:
    r"""
    Generic class for 1D diffusion processes.

    It corresponds to the family of 1D SDEs :math:`dx_t = F(x_t, t)dt + \sigma(x_t, t)dW_t`,
    where :math:`F` is a time-dependent vector field and :math:`W` the Wiener process.

    Parameters
    ----------
    vecfield : function with two arguments
        The vector field :math:`F(x, t)`.
    sigma : function with two arguments
        The diffusion coefficient :math:`\sigma(x, t)`.
    """
    default_dt = 0.1

    def __init__(self, vecfield, sigma, **kwargs):
        """
        F and sigma are functions of two variables (x,t)
        """
        self.drift = vecfield
        self.diffusion = sigma
        self.__deterministic__ = kwargs.get('deterministic', False)

    def potential(self, X, t):
        """
        Compute the potential from which the force derives.

        Parameters
        ----------
        X : ndarray
            The points where we want to compute the potential.

        Returns
        -------
        V : ndarray
            The potential from which the force derives, at the given points.

        Notes
        -----
        We integrate the vector field to obtain the value of the underlying potential
        at the input points.
        Caveat: This works only for 1D dynamics.
        """
        fun = interp1d(X, -self.drift(X, t))
        return np.array([integrate.quad(fun, 0.0, x)[0] for x in X])

    def increment(self, x, t, **kwargs):
        """ Return F(x_t, t)dt + sigma(x_t, t)dW_t """
        dt = kwargs.get('dt', self.default_dt)
        return self.drift(x, t) * dt + self.diffusion(x, t)*np.sqrt(dt)*np.random.normal(0.0, 1.0)

    def update(self, x, t, **kwargs):
        """
        Return the next sample for the time-discretized process, using the Euler-Maruyama method

        Refs:
        - P. E. Kloeden and E. Platen, Numerical Solution of Stochastic Differential Equations,
          Springer (1992)
        - G. Maruyama, Continuous Markov processes and stochastic equations, Rend. Circ. Mat. Palermo
          4, 48-90 (1955)
        """
        return x + self.increment(x, t, **kwargs)

    @pseudorand
    def trajectory_numpy(self, x0, t0, **kwargs):
        """
        Integrate a trajectory with given initial condition (t0,x0)
        Optional arguments:
        - dt: float, the time step
        - T: float, the integration time (i.e. the duration of the trajectory)
        - finite: boolean, whether to filter output to return only finite values (default False)

        This is the fastest way I have found to build the array directly using numpy.ndarray object
        It still takes twice as much time as building a list and casting it to a numpy.ndarray.
        """
        dt = kwargs.pop('dt', self.default_dt)
        time = kwargs.pop('T', 10.0)
        if dt < 0:
            time = -time
        precision = kwargs.pop('precision', np.float32)
        num = int(time/dt)+1
        t = np.linspace(t0, t0+dt*int(time/dt), num=num, dtype=precision)
        x = np.full(num, x0, dtype=precision)
        for index in range(1, num):
            x[index] = self.update(x[index-1], t[index-1], dt=dt, **kwargs)
        if kwargs.pop('finite', False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t[t <= t0+time], x[t <= t0+time]

    @pseudorand
    def trajectory(self, x0, t0, **kwargs):
        r"""
        Integrate the SDE with given initial condition.

        Parameters
        ----------
        x0: float
            The initial position.
        t0: float
            The initial time.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the increment routine
            (default 0.1, unless overridden by a subclass).
        T: float
            The time duration of the trajectory (default 10).
        finite: bool
            Filter finite values before returning trajectory (default False).

        Returns
        -------
        t, x: ndarray, ndarray
            Time-discrete sample path for the stochastic process with initial conditions (t0, x0).
            The array t contains the time discretization and x the value of the sample path
            at these instants.
        """
        t = [t0]
        x = [x0]
        dt = kwargs.pop('dt', self.default_dt)
        time = kwargs.pop('T', 10.0)
        precision = kwargs.pop('precision', np.float32)
        if dt < 0:
            time = -time
        while t[-1] <= t0+time:
            t.append(t[-1] + dt)
            x.append(self.update(x[-1], t[-1], dt=dt, **kwargs))
        t = np.array(t, dtype=precision)
        x = np.array(x, dtype=precision)
        if kwargs.pop('finite', False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t[t <= t0+time], x[t <= t0+time]

    def trajectory_conditional(self, x0, t0, pred, **kwargs):
        r"""
        Compute sample path satisfying arbitrary condition.

        Parameters
        ----------
        x0: float
            The initial position.
        t0: float
            The initial time.
        pred: function with two arguments
            The predicate to select trajectories.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the increment routine
            (default 0.1, unless overridden by a subclass).
        T: float
            The time duration of the trajectory (default 10).
        finite: bool
            Filter finite values before returning trajectory (default False).

        Returns
        -------
        t, x: ndarray, ndarray
            Time-discrete sample path for the stochastic process with initial conditions (t0, x0).
            The array t contains the time discretization and x the value of the sample path
            at these instants.
        """
        while True:
            t, x = self.trajectory(x0, t0, **kwargs)
            if pred(t, x):
                break
        return t, x

    @classmethod
    def trajectoryplot(cls, *args, **kwargs):
        """ Plot previously computed trajectories """
        _ = plt.figure()
        ax = plt.axes()
        lines = []
        for t, x in args:
            lines += ax.plot(t, x)

        ax.grid()
        ax.set_ylim(kwargs.get('ylim', ax.get_ylim()))
        ax.set_xlim(kwargs.get('xlim', ax.get_xlim()))
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x(t)$')
        plottitle = kwargs.get('title', "")
        if plottitle != "":
            plt.title(plottitle)

        labels = kwargs.get('labels', [])
        if labels != []:
            plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        cls._trajectoryplot_decorate(*args, axis=ax, **kwargs)
        plt.show()

    @classmethod
    def _trajectoryplot_decorate(cls, *args, **kwargs):
        pass

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        return NotImplemented


class ConstantDiffusionProcess1D(DiffusionProcess1D):
    r"""
    Diffusion processes in 1D with constant diffusion coefficient.

    It corresponds to the family of SDEs :math:`dx_t = F(x_t, t)dt + \sigma dW_t`,
    where :math:`F` is a time-dependent vector field and :math:`W` the Wiener process.
    The diffusion coefficient :math:`\sigma` is independent of space and time.

    Parameters
    ----------
    vecfield : function with two arguments
        The vector field :math:`F(x, t)`.
    Damp : float
        The amplitude of the noise.

    Notes
    -----
    The diffusion coefficient is given by :math:`\sigma=\sqrt{2\text{Damp}}`.
    This convention leads to simpler expressions, for instance for the Fokker-Planck equations.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp, **kwargs):
        """
        vecfield: function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise), scalar
        """
        DiffusionProcess1D.__init__(self, vecfield, lambda x, t: np.sqrt(2*Damp), **kwargs)
        self.F = vecfield # We keep this temporarily for backward compatiblity
        self.D0 = Damp    # We keep this temporarily for backward compatiblity

    def increment(self, x, t, **kwargs):
        """ Return F(x_t,t)dt + sqrt(2*D0)dW_t """
        dt = kwargs.get('dt', self.default_dt)
        return self.F(x, t) * dt + np.sqrt(2.0*self.D0*dt)*np.random.normal(0.0, 1.0)

    def time_reversal(self):
        """ Apply time reversal and return the new model """
        return StochModel1D_T(lambda x, t: -self.F(x, -t), self.D0)

    def traj_cond_gen(self, x0, t0, tau, M, **kwargs):
        """Generate trajectories conditioned on the first-passage time tau at value M.
        Initial conditions are (x0,t0).
        Optional keyword arguments:
        - dt     -- integration timestep (default is self.default_dt)
        - ttol   -- first-passage time tolerance (default is 1% of trajectory duration)
        - num    -- number of trajectories generated (default is 10)
        - interp -- interpolate to generate unifomly sampled trajectories
        - npts   -- number of points for interpolated trajectories (default (tau-t0)/dt)
        """
        dt = kwargs.get('dt', self.default_dt)
        tau_tol = kwargs.get('ttol', 0.01*np.abs(tau-t0))
        num = kwargs.pop('num', 10)
        interp = kwargs.pop('interp', False)
        npts = kwargs.pop('npts', np.abs(tau-t0)/dt)
        while num > 0:
            x = [x0]
            t = [t0]
            while (x[-1] <= M and t[-1] <= tau):
                x += [self.update(x[-1], t[-1], dt=dt)]
                t += [t[-1] + dt]
            if (x[-1] > M and np.abs(t[-1]-tau) < tau_tol):
                num -= 1
                if interp:
                    fun = interp1d(t, x, fill_value='extrapolate')
                    t = np.linspace(t0, tau, num=npts)
                    x = fun(t)
                yield t, x

    def blowuptime(self, x0, t0, **kwargs):
        """ Compute the last time with finite values, for one realization"""
        t, x = self.trajectory(x0, t0, **kwargs)
        return t[np.isfinite(x)][-1]

    def pdfplot(self, *args, **kwargs):
        """ Plot the pdf P(x,t) at various times """
        _ = plt.figure()
        ax = plt.axes()
        t0 = kwargs.pop('t0', args[0])
        fun = kwargs.pop('integ', fp.FokkerPlanck1D(self.F, self.D0).fpintegrate)
        if kwargs.get('potential', False):
            ax2 = ax.twinx()
            ax2.set_ylabel('$V(x,t)$')
        for t in args:
            t, X, P = fun(t0, t-t0, **kwargs)
            line, = ax.plot(X, P, label='t='+format(t, '.2f'))
            if kwargs.get('th', False):
                Pth = self._fpthsol(X, t, **kwargs)
                if Pth is not None:
                    ax.plot(X, Pth, color=line.get_color(), linestyle='dotted')
            if kwargs.get('potential', False):
                ax2.plot(X, self.potential(X, t), linestyle='dashed')
            t0 = t
            kwargs['P0'] = P

        ax.grid()
        ax.set_xlabel('$x$')
        ax.set_ylabel(kwargs.get('ylabel', '$P(x,t)$'))
        plt.title(r'$\epsilon='+str(self.D0)+'$')
        ax.legend(**(kwargs.get('legend_args', {})))
        plt.show()

    def instanton(self, x0, p0, *args, **kwargs):
        """
        Numerical integration of the equations of motion for instantons.
        x0 and p0 are the initial conditions.
        Return the instanton trajectory (t,x).
        """
        def inverse(f):
            return lambda Y, t: -f(Y, -t)
        def filt_fun(t, x):
            filt = (x > 100.0).nonzero()[0]
            if len(filt) > 0:
                maxind = filt[0]
            else:
                maxind = -1
            return t[:maxind], x[:maxind]
        solver = kwargs.pop('solver', 'odeint')
        scheme = kwargs.pop('integrator', 'dopri5')
        filt_traj = kwargs.pop('filter_traj', False)
        back = kwargs.pop('backwards', False)
        times = np.sort(args)
        fun = self._instantoneq
        jac = self._instantoneq_jac
        if back:
            fun = inverse(fun)
            jac = inverse(jac)
        if solver == 'odeint':
            x = integrate.odeint(fun, (x0, p0), times, tfirst=True, **kwargs)[:, 0]
            return filt_fun(times, x) if filt_traj else (times, x)
        elif solver == 'odeclass':
            integ = integrate.ode(fun, jac=jac).set_integrator(scheme, **kwargs)
            integ.set_initial_value([x0, p0], times[0])
            return times, [integ.integrate(t)[0] for t in times]

    def _instantoneq(self, t, Y):
        """
        Equations of motion for instanton dynamics.
        These are just the Hamilton equations corresponding to the action.

        Y should be a vector (list or numpy.ndarray) with two items: x=Y[0] and p=Y[1]
        """
        x = Y[0]
        p = Y[1]
        return [2.*p+self.F(x, t),
                -p*derivative(self.F, x, dx=1e-6, args=(t, ))]

    def _instantoneq_jac(self, t, Y):
        """
        Jacobian of instanton dynamics.

        Y should be a vector (list or numpy.ndarray) with two items: x=Y[0] and p=Y[1]
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.F, x, dx=1e-6, args=(t, ))
        return np.array([[dbdx, 2.],
                         [-p*derivative(self.F, x, n=2, dx=1e-6, args=(t, )), -dbdx]])


    def action(self, *args):
        """
        Compute the action for all the trajectories given as arguments
        """
        for t, x in args:
            xdot = edpy.CenteredFD(t).grad(x)
            p = 0.5*(xdot-self.F(x[1:-1], t[1:-1]))
            yield integrate.trapz(p**2, t[1:-1])

class OrnsteinUhlenbeck1D(ConstantDiffusionProcess1D):
    r"""
    The 1D Ornstein-Uhlenbeck process.

    It corresponds to the SDE :math:`dx_t = \theta(\mu-x_t)dt + \sqrt{2D} dW_t`,
    where :math:`\theta>0` and :math:`\mu` are arbitrary coefficients
    and :math:`D>0` is the amplitude of the noise.

    Parameters
    ----------
    mu : float
        The expectation value.
    theta : float
        The inverse of the relaxation time.
    D : float
        The amplitude of the noise.

    Notes
    -----
    The Ornstein-Uhlenbeck process has been used to model many systems.
    It was initially introduced to describe the motion of a massive
    Brownian particle with friction [1]_ .
    It may also be seen as a diffusion process in a harmonic potential.

    Because many of its properties can be computed analytically, it provides a useful
    toy model for developing new methods.

    References
    ----------
    .. [1] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of Brownian Motion".
           Phys. Rev. 36, 823–841 (1930).
    """
    def __init__(self, mu, theta, D, **kwargs):
        self.theta = theta
        self.d_f = (lambda x, t: -theta)
        super(OrnsteinUhlenbeck1D, self).__init__(lambda x, t: theta*(mu-x), D, **kwargs)

    def update(self, x, t, **kwargs):
        """
        Return the next sample for the time-discretized process.

        For the Ornstein-Uhlenbeck process, there is an exact method; see
        D. T. Gillespie, Exact numerical simulation of the Ornstein-Uhlenbeck process and its
                         integral, Phys. Rev. E 54, 2084 (1996).
        """
        if kwargs.pop('method', 'gillespie') == 'gillespie':
            dt = kwargs.get('dt', self.default_dt)
            xx = x*np.exp(-self.theta*dt)+np.sqrt(self.D0/self.theta*(1-np.exp(-2*self.theta*dt)))*np.random.normal(0.0, 1.0)
        else:
            xx = ConstantDiffusionProcess1D.update(self, x, t, **kwargs)
        return xx

    def _instantoneq(self, t, Y):
        """
        Equations of motion for instanton dynamics.
        These are just the Hamilton equations corresponding to the action.

        Y should be a vector (list or numpy.ndarray) with two items: x=Y[0] and p=Y[1]
        """
        x = Y[0]
        p = Y[1]
        return [2.*p+self.F(x, t),
                -p*self.d_f(x, t)]

    def _instantoneq_jac(self, t, Y):
        """
        Jacobian of instanton dynamics.

        Y should be a vector (list or numpy.ndarray) with two items: x=Y[0] and p=Y[1]
        """
        dbdx = self.d_f(Y[0], t)
        return np.array([[dbdx, 2.],
                         [0, -dbdx]])

class Wiener1D(OrnsteinUhlenbeck1D):
    r"""
    The 1D Wiener process.

    Parameters
    ----------
    D : float, optional
        The amplitude of the noise (default is 1).

    Notes
    -----
    The Wiener process is a central object in the theory or stochastic processes,
    both from a mathematical point of view and for its applications in different scientific fields.
    We refer to classical textbooks for more information about the Wiener process
    and Brownian motion.
    """
    def __init__(self, D=1, **kwargs):
        super(Wiener1D, self).__init__(0, 0, D, **kwargs)

    @classmethod
    def potential(cls, X):
        r"""
        Compute the potential from which the force derives.

        Parameters
        ----------
        X : ndarray
            The points where we want to compute the potential.

        Returns
        -------
        V : float
            The potential from which the force derives, at the given points.

        Notes
        -----
        The Wiener Process is a trivial gradient system, with vanishing potential.
        It is useless (and potentially source of errors) to call the general potential routine,
        so we just return zero directly.
        """
        return np.zeros_like(X)

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the heat equation.
        This should depend on the boundary conditions.
        Right now, we do as if we were solving on the real axis."""
        return np.exp(-X**2.0/(4.0*self.D0*t))/np.sqrt(4.0*np.pi*self.D0*t)


class DrivenOrnsteinUhlenbeck1D(ConstantDiffusionProcess1D):
    """
    The 1D Ornstein-Uhlenbeck model driven by a periodic forcing:
        dx_t = theta*(mu-x_t)+A*sin(Omega*t+phi)+sqrt(2*D)*dW_t
    """
    def __init__(self, mu, theta, D, A, Omega, phi, **kwargs):
        ConstantDiffusionProcess1D.__init__(self, lambda x, t: theta*(mu-x)+A*np.sin(Omega*t+phi), D, **kwargs)



class StochModel1D_T(ConstantDiffusionProcess1D):
    """ Time reversal of a given model """
    def trajectory(self, x0, t0, **kwargs):
        t, x = super(StochModel1D_T, self).trajectory(x0, t0, **kwargs)
        return 2*t[0]-t, x
