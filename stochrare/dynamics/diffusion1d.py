"""
Simulating 1D diffusion processes
=================================

.. currentmodule:: stochrare.dynamics.diffusion1d

This module defines the `DiffusionProcess1D` class, representing diffusion processes with
arbitrary drift and diffusion coefficients in 1D.

This class can be subclassed for specific diffusion processes for which methods can be specialized,
both to simplify the code (e.g. directly enter analytical formulae when they are available) and for
performance.
As an exemple of this mechanism, we also provide in this module the `ConstantDiffusionProcess1D`
class, for which the diffusion term is constant (additive noise), the `OrnsteinUhlenbeck1D` class
representing the particular case of the Ornstein-Uhlenbeck process, and the `Wiener1D` class
corresponding  to Brownian motion.
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
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.misc import derivative
from .. import edpy
from .. import fokkerplanck as fp
from ..utils import pseudorand
from ..io import plot


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

    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn : float
            The current position.
        tn : float
            The current time.

        Keyword Arguments
        -----------------
        dt : float
            The time step (default 0.1 if not overriden by a subclass).
        dw : float
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : float
            The position at time tn+dt.

        Notes
        -----
        This method uses the Euler-Maruyama method [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        References
        ----------
        .. [1] G. Maruyama, Continuous Markov processes and stochastic equations, Rend. Circ. Mat.
          Palermo 4, 48-90 (1955).
        .. [2] P. E. Kloeden and E. Platen, Numerical Solution of Stochastic Differential Equations,
          Springer (1992).
        """
        dt = kwargs.get('dt', self.default_dt)
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt)))
        return xn + self.drift(xn, tn)*dt + self.diffusion(xn, tn)*dw

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
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        T: float
            The time duration of the trajectory (default 10).
        brownian_path : (ndarray, ndarray)
            A precomputed Brownian path with respect to which we integrate the SDE.
            If not provided (default behavior), one will be computed one the fly.
        deltat : float
            The time step for the Brownian path, when generated on the fly (default: dt).
        finite: bool
            Filter finite values before returning trajectory (default False).

        Returns
        -------
        t, x: ndarray, ndarray
            Time-discrete sample path for the stochastic process with initial conditions (t0, x0).
            The array t contains the time discretization and x the value of the sample path
            at these instants.
        """
        dt = kwargs.pop('dt', self.default_dt)
        time = kwargs.pop('T', 10.0)
        if dt < 0:
            time = -time
        precision = kwargs.pop('precision', np.float32)
        num = int(time/dt)+1
        t = np.linspace(t0, t0+dt*(num-1), num=num, dtype=precision)
        x = np.full(num, x0, dtype=precision)
        if 'brownian_path' in kwargs:
            tw, w = kwargs.pop('brownian_path')
            dw = np.diff(w)
            deltat = tw[1]-tw[0]
            ratio = int(np.rint(dt/deltat))
            dw = dw[:((num-1)*ratio)]
        else:
            deltat = kwargs.pop('deltat', dt)
            ratio = int(np.rint(dt/deltat))
            dw = np.random.normal(0, np.sqrt(deltat), size=(num-1)*ratio)
        dw = dw.reshape((num-1, ratio)).sum(axis=1)
        for index in range(1, num):
            x[index] = self.update(x[index-1], t[index-1], dt=dt, dw=dw[index-1])
        if kwargs.pop('finite', False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t[t <= t0+time], x[t <= t0+time]

    @pseudorand
    def _trajectory_fast(self, x0, t0, **kwargs):
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
            The time step, forwarded to the :math:`update` routine
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

        Notes
        -----
        For some reasons which are not fully elucidated yet, this old version of the integration
        routine, although less flexible than the new (it does not allow for providing the
        Brownian path) can be up to three times faster.
        Hence, until the :meth:`trajectory` method has been optimized, we keep this method
        around for cases where performance matter.

        'Premature optimization is the root of all evil' --- Donald Knuth.
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
            The time step, forwarded to the :meth:`update` routine
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

    def blowuptime(self, x0, t0, **kwargs):
        """
        Compute the last time with finite values, for one realization.

        Parameters
        ----------
        x0: float
            The initial position.
        t0: float
            The initial time.

        Returns
        -------
        The last time with finite values for a realization with initial conditions (t0, x0).
        This is a random variable.
        """
        t, x = self.trajectory(x0, t0, **kwargs)
        return t[np.isfinite(x)][-1]

    def empirical_vector(self, x0, t0, nsamples, *args, **kwargs):
        """
        Empirical vector at given times.

        Parameters
        ----------
        x0 : float
            Initial position.
        t0 : float
            Initial time.
        nsamples : int
            The size of the ensemble.
        *args : variable length argument list
            The times at which we want to estimate the empirical vector.

        Keyword Arguments
        -----------------
        **kwargs :
            Keyword arguments forwarded to :meth:`trajectory` and to :meth:`numpy.histogram`.

        Yields
        ------
        t, pdf, bins : float, ndarray, ndarray
            The time and histogram of the stochastic process at that time.

        Notes
        -----
        This method computes the empirical vector, or in other words, the relative frequency of the
        stochastic process at different times, conditioned on the initial condition.
        At each time, the empirical vector is a random vector.
        It is an estimator of the transition probability :math:`p(x, t | x_0, t_0)`.
        """
        hist_kwargs_keys = ('bins', 'range', 'weights') # hard-coded for now, we might use inspect
        hist_kwargs = {key: kwargs[key] for key in kwargs if key in hist_kwargs_keys}
        def traj_sample(x0, t0, *args, **kwargs):
            for tsample in args:
                t, x = self.trajectory(x0, t0, T=tsample-t0, **kwargs)
                t0 = t[-1]
                x0 = x[-1]
                yield tsample, x0
        for ensemble in zip(*[traj_sample(x0, t0, *args, **kwargs) for _ in range(nsamples)]):
            time, obs = zip(*ensemble)
            yield (time[0], ) + np.histogram(obs, density=True, **hist_kwargs)


    @classmethod
    def trajectoryplot(cls, *args, **kwargs):
        """
        Plot 1D  trajectories.

        Parameters
        ----------
        *args : variable length argument list
        trajs: tuple (t, x)

        Keyword Arguments
        -----------------
        fig : matplotlig.figure.Figure
            Figure object to use for the plot. Create one if not provided.
        ax : matplotlig.axes.Axes
            Axes object to use for the plot. Create one if not provided.
        **kwargs :
            Other keyword arguments forwarded to matplotlib.pyplot.axes.

        Returns
        -------
        fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
            The figure.

        Notes
        -----
        This is just an interface to the function :meth:`stochrare.io.plot.trajectory_plot1d`.
        However, it may be overwritten in subclasses to systematically include elements to
        the plot which are specific to the stochastic process.
        """
        return plot.trajectory_plot1d(*args, **kwargs)

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        return NotImplemented


class ConstantDiffusionProcess1D(DiffusionProcess1D):
    r"""
    Diffusion processes in 1D with constant diffusion coefficient (additive noise).

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

    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn : float
            The current position.
        tn : float
            The current time.

        Keyword Arguments
        -----------------
        dt : float
            The time step (default 0.1 if not overriden by a subclass).
        dw : float
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : float
            The position at time tn+dt.

        Notes
        -----
        This method uses the Euler-Maruyama method [3]_ [4]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sqrt{2D} \Delta W_n`.

        References
        ----------
        .. [3] G. Maruyama, Continuous Markov processes and stochastic equations, Rend. Circ. Mat.
          Palermo 4, 48-90 (1955).
        .. [4] P. E. Kloeden and E. Platen, Numerical Solution of Stochastic Differential Equations,
          Springer (1992).
        """
        dt = kwargs.get('dt', self.default_dt)
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt)))
        return xn + self.drift(xn, tn)*dt + np.sqrt(2.0*self.D0)*dw

    def traj_cond_gen(self, x0, t0, tau, M, **kwargs):
        """
        Generate trajectories conditioned on the first-passage time tau at value M.

        Parameters
        ----------
        x0: float
            Initial position.
        t0: float
            Initial time.
        tau: float
            The value of the first passage time required.
        M: float
            The threshold for the first passage time.

        Keyword Arguments
        -----------------
        dt: float
            The integration timestep (default is self.default_dt).
        ttol: float
            The first-passage time tolerance (default is 1% of trajectory duration).
        num: int
            The number of trajectories generated (default is 10).
        interp: bool
            Interpolate to generate unifomly sampled trajectories.
        npts: int
            The number of points for interpolated trajectories (default (tau-t0)/dt).

        Yields
        -------
        t, x: ndarray, ndarray
            Trajectories satisfying the condition on the first passage time.
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


    def pdfplot(self, *args, **kwargs):
        """
        Plot the pdf P(x,t) at various times.

        Parameters
        ----------
        args : variable length argument list
            The times at which to plot the PDF.

        Keyword Arguments
        -----------------
        t0 : float
            Initial time.
        potential : bool
            Plot potential on top of PDF.
        th : bool
            Plot theoretical solution, if it exists, on top of PDF.
        """
        fig, ax, _ = plot.pdf_plot1d(legend=False, title=r'$D='+str(self.D0)+'$')
        kw_integ = ('dt', 'npts', 'bounds', 't0', 'P0', 'bc', 'method', 'adjoint')
        fpe = fp.FokkerPlanck1D(self.drift, self.D0)
        fpgen = fpe.fpintegrate_generator(*args,
                                          **{k: v for k, v in kwargs.items() if k in kw_integ})
        for t, X, P in fpgen:
            if kwargs.get('potential', False):
                kwargs['potential'] = (X, self.potential(X, t))
            _, _, lines = plot.pdf_plot1d((np.array(X), np.array(P),
                                           {'label': 't='+format(t, '.2f')}),
                                          fig=fig, ax=ax, legend=False, **kwargs)
            if kwargs.get('th', False):
                plot.pdf_plot1d((X, self._fpthsol(X, t),
                                 {'ls': 'dotted', 'color': lines[0].get_color()}),
                                fig=fig, ax=ax, legend=False)
        fig, ax, _ = plot.pdf_plot1d(fig=fig, ax=ax, legend=True)
        return fig, ax


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
    Brownian particle with friction [5]_ .
    It may also be seen as a diffusion process in a harmonic potential.

    Because many of its properties can be computed analytically, it provides a useful
    toy model for developing new methods.

    References
    ----------
    .. [5] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of Brownian Motion".
           Phys. Rev. 36, 823–841 (1930).
    """
    def __init__(self, mu, theta, D, **kwargs):
        self.theta = theta
        self.d_f = (lambda x, t: -theta)
        super(OrnsteinUhlenbeck1D, self).__init__(lambda x, t: theta*(mu-x), D, **kwargs)

    def update(self, xn, tn, **kwargs):
        """
        Return the next sample for the time-discretized process, using the Gillespie method.

        Parameters
        ----------
        xn : float
            The current position.
        tn : float
            The current time.

        Keyword Arguments
        -----------------
        dt : float
            The time step (default 0.1 if not overriden by a subclass).
        dw : float
            The brownian increment if precomputed.
            By default, it is generated on the fly from a standard Gaussian distribution.
        method : str
            The numerical method for integration: 'gillespie' (default) or 'euler'.

        Returns
        -------
        x : float
            The position at time tn+dt.

        Notes
        -----
        For the Ornstein-Uhlenbeck process, there is an exact method, the Gillespie algorithm [6]_.
        This method is selected by default.
        If necessary, the Euler-Maruyama method can still be chosen using the ``method`` keyword
        argument.

        References
        ----------
        .. [6] D. T. Gillespie, Exact numerical simulation of the Ornstein-Uhlenbeck process and its
               integral, Phys. Rev. E 54, 2084 (1996).
        """
        if kwargs.pop('method', 'gillespie') == 'gillespie' and self.theta != 0:
            dt = kwargs.get('dt', self.default_dt)
            dw = kwargs.get('dw', np.random.normal(0.0, 1.0))
            xx = xn*np.exp(-self.theta*dt) \
                 + np.sqrt(self.D0/self.theta*(1-np.exp(-2*self.theta*dt)))*dw
        else:
            xx = ConstantDiffusionProcess1D.update(self, xn, tn, **kwargs)
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
