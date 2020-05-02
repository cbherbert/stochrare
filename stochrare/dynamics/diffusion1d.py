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

References
----------

.. [1] G. Maruyama, Continuous Markov processes and stochastic equations,
       Rend. Circ. Mat. Palermo 4, 48-90 (1955).
.. [2] P. E. Kloeden and E. Platen, Numerical Solution of Stochastic Differential Equations,
       Springer (1992).
.. [3] G. N. Milstein, Numerical Integration of Stochastic Differential Equations,
       Mathematics and Its Applications (Kluwer Academic, Norwell, MA, 1995)
.. [4] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of Brownian Motion".
       Phys. Rev. 36, 823–841 (1930).
.. [5] D. T. Gillespie, Exact numerical simulation of the Ornstein-Uhlenbeck process and its
       integral, Phys. Rev. E 54, 2084 (1996).


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
from scipy.special import erfi, erfcx
from numba import jit
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
        self.drift = jit(vecfield, nopython=True)
        self.diffusion = jit(sigma, nopython=True)
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
        fun = interp1d(X, -1*self.drift(X, t), fill_value='extrapolate')
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
        x = self.integrate_sde(x, t, dw, dt=dt, **kwargs)
        if kwargs.pop('finite', False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t[t <= t0+time], x[t <= t0+time]

    def integrate_sde(self, x, t, w, **kwargs):
        r"""
        Dispatch SDE integration for different numerical schemes

        Parameters
        ----------
        x: ndarray
            The (empty) position array
        t: ndarray
            The sample time array
        w: ndarray
            The brownian motion realization used for integration

        Keyword Arguments
        -----------------
        method: str
            The numerical scheme: 'euler' (default) or 'milstein'
        dt: float
            The time step

        Notes
        -----
        We define this method rather than putting the code in the `trajectory` method to make
        it easier to implement numerical schemes valid only for specific classes of processes.
        Then it suffices to implement the scheme and subclass this method to add the corresponding
        'if' statement, without rewriting the entire `trajectory` method.

        The implemented schemes are the following:

        - Euler-Maruyama [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        - Milstein [2]_ [3]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n + \frac{\sigma(x_n, t_n)\sigma'(x_n, t_n)}{2} \lbrack \Delta W_n^2-\Delta t\rbrack`.

        It is the next order scheme in the strong Ito-Taylor approximations.
        The Milstein scheme has strong order 1.
        """
        method = kwargs.get('method', 'euler')
        dt = kwargs.get('dt', self.default_dt)
        if method in ('euler', 'euler-maruyama', 'em'):
            x = self._euler_maruyama(x, t, w, dt, self.drift, self.diffusion)
        elif method == 'milstein':
            x = self._milstein(x, t, w, dt, self.drift, self.diffusion)
        else:
            raise NotImplementedError('SDE integration error: Numerical scheme not implemented')
        return x

    @staticmethod
    @jit(nopython=True)
    def _euler_maruyama(x, t, w, dt, drift, diffusion):
        index = 1
        for wn in w:
            xn = x[index-1]
            tn = t[index-1]
            x[index] = xn + drift(xn, tn)*dt + diffusion(xn, tn)*wn
            index = index + 1
        return x

    @staticmethod
    def _milstein(x, t, w, dt, drift, diffusion):
        index = 1
        for wn in w:
            xn = x[index-1]
            tn = t[index-1]
            a = drift(xn, tn)
            b = diffusion(xn, tn)
            db = derivative(diffusion, xn, dx=1e-6, args=(t, ))
            x[index] = xn + (a-0.5*b*db)*dt + b*wn + 0.5*b*db*wn**2
            index = index + 1
        return x

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

    def _instantoneq(self, t, Y):
        r"""
        Equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (size 2)
            The right hand side of the Hamilton equations.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/2 \int ((\dot{x}-b(x, t))/sigma(x, t))^2 dt`, i.e.
        :math:`\dot{x}=\sigma(x,t)^2*p+b(x, t)` and
        :math:`\dot{p}=-\sigma(x, t)*\sigma'(x, t)*p^2-b'(x, t)*p`.

        The Hamiltonian is :math:`H=\sigma^2(x, t)*p^2/2+b(x, t)*p`.

        Note that these equations include the diffusion coefficient, unlike those we use in the case
        of a constant diffusion process `ConstantDiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        dsigmadx = derivative(self.diffusion, x, dx=1e-6, args=(t, ))
        return np.array([p*self.diffusion(x, t)**2+self.drift(x, t),
                         -p**2*self.diffusion(x, t)*dsigmadx-p*dbdx])

    def _instantoneq_jac(self, t, Y):
        r"""
        Jacobian of the equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (shape (2, 2))
            The Jacobian of the right hand side of the Hamilton equations, i.e.
            :math:`[[d\dot{x}/dx, d\dot{x}/dp], [d\dot{p}/dx, d\dot{p}/dp]]`.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/2 \int ((\dot{x}-b(x, t))/sigma(x, t))^2 dt`, i.e.
        :math:`\dot{x}=\sigma(x,t)^2*p+b(x, t)` and
        :math:`\dot{p}=-\sigma(x, t)*\sigma'(x, t)*p^2-b'(x, t)*p`.

        The Hamiltonian is :math:`H=\sigma^2(x, t)*p^2/2+b(x, t)*p`.

        Note that these equations include the diffusion coefficient, unlike those we use in the case
        of a constant diffusion process `ConstantDiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        d2bdx2 = derivative(self.drift, x, n=2, dx=1e-5, args=(t, ))
        sigma = self.diffusion(x, t)
        dsigmadx = derivative(self.diffusion, x, dx=1e-6, args=(t, ))
        d2sigmadx2 = derivative(self.diffusion, x, n=2, dx=1e-5, args=(t, ))
        return np.array([[dbdx+2*p*sigma*dsigmadx, sigma**2],
                         [-p*d2bdx2-p**2*(dsigmadx**2+sigma*d2sigmadx2), -dbdx-2*p*sigma*dsigmadx]])

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
        This method uses the Euler-Maruyama method [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sqrt{2D} \Delta W_n`.
        """
        dt = kwargs.get('dt', self.default_dt)
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt)))
        return xn + self.drift(xn, tn)*dt + np.sqrt(2.0*self.D0)*dw

    def integrate_sde(self, x, t, w, **kwargs):
        r"""
        Dispatch SDE integration for different numerical schemes

        Parameters
        ----------
        x: ndarray
            The (empty) position array
        t: ndarray
            The sample time array
        w: ndarray
            The brownian motion realization used for integration

        Keyword Arguments
        -----------------
        method: str
            The numerical scheme: 'euler' (default) or 'milstein'
        dt: float
            The time step

        Notes
        -----
        The implemented schemes are the following:

        - Euler-Maruyama [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        - Milstein [2]_ [3]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n + \frac{\sigma(x_n, t_n)\sigma'(x_n, t_n)}{2} \lbrack \Delta W_n^2-\Delta t\rbrack`.

        It is the next order scheme in the strong Ito-Taylor approximations.
        The Milstein scheme has strong order 1.

        For processes with a constant diffusion coefficient, the Milstein method
        reduces to the Euler-Maruyama method.
        """
        method = kwargs.get('method', 'euler')
        dt = kwargs.get('dt', self.default_dt)
        if method in ('euler', 'euler-maruyama', 'em'):
            x = self._euler_maruyama_const(x, t, w, dt, self.drift, self.D0)
        else:
            x = DiffusionProcess1D.integrate_sde(self, x, t, w, **kwargs)
        return x

    @staticmethod
    @jit(nopython=True)
    def _euler_maruyama_const(x, t, w, dt, drift, D0):
        index = 1
        for wn in w:
            xn = x[index-1]
            tn = t[index-1]
            x[index] = xn + drift(xn, tn)*dt + np.sqrt(2.0*D0)*wn
            index = index + 1
        return x

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
        npts = kwargs.pop('npts', int(np.abs(tau-t0)/dt))
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
        fpe = fp.FokkerPlanck1D.from_sde(self)
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

    def _instantoneq(self, t, Y):
        r"""
        Equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (size 2)
            The right hand side of the Hamilton equations.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/4 \int ((\dot{x}-b(x, t)))^2 dt`, i.e.
        :math:`\dot{x}=2p+b(x, t)` and
        :math:`\dot{p}=-b'(x, t)*p`.

        The Hamiltonian is :math:`H=p^2/2+b(x, t)*p`.

        Note that these equations do not include the (constant) diffusion coefficient,
        unlike those we use in the case of a non-constant diffusion process `DiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        return [2.*p+self.drift(x, t), -p*dbdx]

    def _instantoneq_jac(self, t, Y):
        r"""
        Jacobian of the equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (shape (2, 2))
            The Jacobian of the right hand side of the Hamilton equations, i.e.
            :math:`[[d\dot{x}/dx, d\dot{x}/dp], [d\dot{p}/dx, d\dot{p}/dp]]`.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/4 \int ((\dot{x}-b(x, t)))^2 dt`, i.e.
        :math:`\dot{x}=2p+b(x, t)` and
        :math:`\dot{p}=-b'(x, t)*p`.

        The Hamiltonian is :math:`H=p^2/2+b(x, t)*p`.

        Note that these equations do not include the (constant) diffusion coefficient,
        unlike those we use in the case of a non-constant diffusion process `DiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        dbdx = derivative(self.drift, x, dx=1e-6, args=(t, ))
        d2bdx2 = derivative(self.drift, x, n=2, dx=1e-5, args=(t, ))
        return np.array([[dbdx, 2.], [-p*d2bdx2, -dbdx]])


    def action(self, *args):
        """
        Compute the action for all the trajectories given as arguments
        """
        for t, x in args:
            xdot = edpy.CenteredFD(t).grad(x)
            p = 0.5*(xdot-self.drift(x[1:-1], t[1:-1]))
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
    Brownian particle with friction [4]_ .
    It may also be seen as a diffusion process in a harmonic potential.

    Because many of its properties can be computed analytically, it provides a useful
    toy model for developing new methods.
    """
    def __init__(self, mu, theta, D, **kwargs):
        self.mu = mu
        self.theta = theta
        super(OrnsteinUhlenbeck1D, self).__init__(lambda x, t: theta*(mu-x), D, **kwargs)

    def __str__(self):
        label = "1D Ornstein-Uhlenbeck process"
        eq = "dx_t = theta(mu-x_t)dt + sqrt(2D) dW_t"
        return f"{label}: {eq}, with theta={self.theta}, mu={self.mu} and D={self.D0}."

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
        For the Ornstein-Uhlenbeck process, there is an exact method, the Gillespie algorithm [5]_.
        This method is selected by default.
        If necessary, the Euler-Maruyama method can still be chosen using the ``method`` keyword
        argument.
        """
        if kwargs.pop('method', 'gillespie') == 'gillespie' and self.theta != 0:
            dt = kwargs.get('dt', self.default_dt)
            dw = kwargs.get('dw', np.random.normal(0.0, 1.0))
            xx = xn*np.exp(-self.theta*dt) \
                 + np.sqrt(self.D0/self.theta*(1-np.exp(-2*self.theta*dt)))*dw
        else:
            xx = ConstantDiffusionProcess1D.update(self, xn, tn, **kwargs)
        return xx

    def integrate_sde(self, x, t, w, **kwargs):
        r"""
        Dispatch SDE integration for different numerical schemes

        Parameters
        ----------
        x: ndarray
            The (empty) position array
        t: ndarray
            The sample time array
        w: ndarray
            The brownian motion realization used for integration

        Keyword Arguments
        -----------------
        method: str
            The numerical scheme: 'euler' (default) or 'milstein'
        dt: float
            The time step

        Notes
        -----
        The implemented schemes are the following:

        - Euler-Maruyama [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        - Milstein [2]_ [3]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n + \frac{\sigma(x_n, t_n)\sigma'(x_n, t_n)}{2} \lbrack \Delta W_n^2-\Delta t\rbrack`.

        It is the next order scheme in the strong Ito-Taylor approximations.
        The Milstein scheme has strong order 1.

        For processes with a constant diffusion coefficient, the Milstein method
        reduces to the Euler-Maruyama method.

        - Gillespie:
        For the Ornstein-Uhlenbeck process, there is an exact method, the Gillespie algorithm [5]_.
        """
        method = kwargs.get('method', 'euler')
        dt = kwargs.get('dt', self.default_dt)
        if method == 'gillespie':
            x = self._gillespie(x, w, dt, self.theta, self.D0)
        else:
            x = ConstantDiffusionProcess1D.integrate_sde(self, x, t, w, **kwargs)
        return x

    @staticmethod
    @jit(nopython=True)
    def _gillespie(x, w, dt, theta, D0):
        index = 1
        D1 = np.exp(-theta*dt)
        D2 = np.sqrt(D0/theta*(1-D1**2))
        for wn in w:
            xn = x[index-1]
            x[index] = D1*xn + D2*wn
            index = index + 1
        return x

    def mean_firstpassage_time(self, x0, a):
        r"""
        Return the mean first-passage time for the 1D Ornstein-Uhlenbeck process (exact formula).

        Parameters
        ----------
        x0: float
            Initial position
        a: float
            Threshold

        Return
        ------
        tau: float
            Mean first-passage time

        Notes
        -----
        The first passage time is defined by :math:`\tau_a(x_0)=\inf \{t>0, X_t>a | X_0=x_0\}`.
        It is a random variable. Here, we compute only its expectation value, for which an
        analytical formula is known.

        General methods for first-passage time conputations are avaiblable in the
        `stochrare.firstpassage` module.
        """
        if self.mu != 0:
            raise NotImplementedError("The theoretical formula has not been checked for nonzero mu")
        if x0 > a:
            tau = 0
        else:
            k = np.sqrt(self.theta/(2*self.D0))
            u = np.linspace(k*x0, k*a)
            integral = np.sqrt(np.pi)/self.theta*np.trapz(erfcx(u), u)
            tau = np.pi/self.theta*(erfi(k*a)-erfi(k*x0))-integral
        return tau

    def _instantoneq(self, t, Y):
        r"""
        Equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (size 2)
            The right hand side of the Hamilton equations.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/4 \int ((\dot{x}-b(x, t)))^2 dt`, i.e.
        :math:`\dot{x}=2p+b(x, t)` and
        :math:`\dot{p}=-b'(x, t)*p`.

        The Hamiltonian is :math:`H=p^2/2+b(x, t)*p`.

        Note that these equations do not include the (constant) diffusion coefficient,
        unlike those we use in the case of a non-constant diffusion process `DiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        x = Y[0]
        p = Y[1]
        return [2.*p+self.theta*(self.mu-x), p*self.theta]

    def _instantoneq_jac(self, t, Y):
        r"""
        Jacobian of the equations of motion for instanton dynamics.

        Parameters
        ----------
        t: float
            The time.
        Y: ndarray or list
            Vector with two elements: x=Y[0] the position and p=Y[1] the impulsion.

        Returns
        -------
        xdot, pdot: ndarray (shape (2, 2))
            The Jacobian of the right hand side of the Hamilton equations, i.e.
            :math:`[[d\dot{x}/dx, d\dot{x}/dp], [d\dot{p}/dx, d\dot{p}/dp]]`.

        Notes
        -----
        These are the Hamilton equations corresponding to the following action:
        :math:`A=1/4 \int ((\dot{x}-b(x, t)))^2 dt`, i.e.
        :math:`\dot{x}=2p+b(x, t)` and
        :math:`\dot{p}=-b'(x, t)*p`.

        The Hamiltonian is :math:`H=p^2/2+b(x, t)*p`.

        Note that these equations do not include the (constant) diffusion coefficient,
        unlike those we use in the case of a non-constant diffusion process `DiffusionProcess1D`.
        Hence, for constant diffusion coefficients, the two only coincide when D=1.
        Otherwise, it amounts at a rescaling of the impulsion.
        """
        return np.array([[-self.theta, 2.], [0, self.theta]])

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
