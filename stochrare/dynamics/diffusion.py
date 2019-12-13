"""
Simulating diffusion processes in arbitrary dimensions
=================================================================

.. currentmodule:: stochrare.dynamics.diffusion

This module defines the `DiffusionProcess` class, representing generic diffusion processes with
arbitrary drift and diffusion coefficients, in arbitrary dimension.

This class can be subclassed for specific diffusion processes for which methods can be specialized,
both to simplify the code (e.g. directly enter analytical formulae when they are available) and for
performance.
As an exemple of this mechanism, we also provide in this module the `ConstantDiffusionProcess`
class, for which the diffusion term is constant and proportional to the identity matrix,
the `OrnsteinUhlenbeck` class representing the particular case of the Ornstein-Uhlenbeck process,
and the `Wiener` class corresponding to Brownian motion.
These classes form a hierarchy deriving from the base class, `DiffusionProcess`.

.. autoclass:: DiffusionProcess
   :members:

.. autoclass:: ConstantDiffusionProcess
   :members:

.. autoclass:: OrnsteinUhlenbeck
   :members:

.. autoclass:: Wiener
   :members:

"""
import numpy as np
from ..utils import pseudorand

class DiffusionProcess:
    r"""
    Generic class for diffusion processes in arbitrary dimensions.

    It corresponds to the family of SDEs :math:`dx_t = F(x_t, t)dt + \sigma(x_t, t)dW_t`,
    where :math:`F` is a time-dependent :math:`N`-dimensional vector field
    and :math:`W` the :math:`M`-dimensional Wiener process.
    The diffusion matrix sigma has size NxM.

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
        vecfield: vector field
        sigma: diffusion coefficient (noise)

        vecfield and sigma are functions of two variables (x,t).
        """
        self.drift = vecfield
        self.diffusion = sigma
        self.__deterministic__ = kwargs.get('deterministic', False)

    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn: ndarray
            A n-dimensional vector (in :math:`\mathbb{R}^n`).
        tn: float
            The current time.

        Keyword Arguments
        ------------------
        dt : float
            The time step.
        dw : ndarray
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : ndarray
            The position at time tn+dt.

        Notes
        -----
        This method uses the Euler-Maruyama method [1]_ [2]_:
        :math:`x_{n+1} = x_n + F(x_n, t_n)\Delta t + \sigma(x_n, t_n) \Delta W_n`,
        for a fixed time step :math:`\Delta t`, where :math:`\Delta W_n` is a random vector
        distributed according to the standard normal distribution [1]_ [2]_.

        It is the straightforward generalization to SDEs of the Euler method for ODEs.

        The Euler-Maruyama method has strong order 0.5 and weak order 1.

        References
        ----------
        .. [1] G. Maruyama, "Continuous Markov processes and stochastic equations",
           Rend. Circ. Mat. Palermo 4, 48-90 (1955).
        .. [2] P. E. Kloeden and E. Platen,
           "Numerical solution of stochastic differential equations", Springer (1992).
        """
        dt = kwargs.get('dt', self.default_dt)
        dim = len(xn)
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt), dim))
        return xn + self.drift(xn, tn)*dt+self.diffusion(xn, tn)@dw

    @pseudorand
    def trajectory(self, x0, t0, **kwargs):
        r"""
        Integrate the SDE with given initial condition.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.

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
        x = [x0]
        dt = kwargs.get('dt', self.default_dt) # Time step
        time = kwargs.get('T', 10.0)   # Total integration time
        if dt < 0:
            time = -time
        tarray = np.linspace(t0, t0+time, num=np.floor(time/dt)+1)
        for t in tarray[1:]:
            x += [self.update(x[-1], t, dt=dt)]
        x = np.array(x)
        if kwargs.get('finite', False):
            tarray = tarray[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return tarray, x

    @pseudorand
    def trajectory_generator(self, x0, t0, nsteps, **kwargs):
        r"""
        Integrate the SDE with given initial condition, generator version.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.
        nsteps: int
            The number of samples to generate.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        observable: function with two arguments
            Time-dependent observable :math:`O(x, t)` to compute (default :math:`O(x, t)=x`)

        Yields
        -------
        t, y: ndarray, ndarray
            Time-discrete sample path (or observable) for the stochastic process with initial
            conditions (t0, x0).
            The array t contains the time discretization and y=O(x, t) the value of the observable
            (it may be the stochastic process itself) at these instants.
        """
        x = x0
        t = t0
        dt = kwargs.get('dt', self.default_dt) # Time step
        obs = kwargs.get('observable', lambda x, t: x)
        for _ in range(nsteps):
            t = t + dt
            x = self.update(x, t, dt=dt)
            yield t, obs(x, t)

    def sample_mean(self, x0, t0, nsteps, nsamples, **kwargs):
        r"""
        Compute the sample mean of a time dependent observable, conditioned on initial conditions.

        Parameters
        ----------
        x0: ndarray
            The initial position (in :math:`\mathbb{R}^n`).
        t0: float
            The initial time.
        nsteps: int
            The number of samples in each sample path.
        nsamples: int
            The number of sample paths in the ensemble.

        Keyword Arguments
        -----------------
        dt: float
            The time step, forwarded to the :meth:`update` routine
            (default 0.1, unless overridden by a subclass).
        observable: function with two arguments
            Time-dependent observable :math:`O(x, t)` to compute (default :math:`O(x, t)=x`)

        Yields
        -------
        t, y: ndarray, ndarray
            Time-discrete ensemble mean for the observable, conditioned on the initial
            conditions (t0, x0).
            The array t contains the time discretization and :math:`y=\mathbb{E}[O(x, t)]`
            the value of the sample mean of the observable (it may be the stochastic process itself)
            at these instants.
        """
        for ensemble in zip(*[self.trajectory_generator(x0, t0, nsteps, **kwargs)
                              for _ in range(nsamples)]):
            time, obs = zip(*ensemble)
            yield np.average(time, axis=0), np.average(obs, axis=0)

class ConstantDiffusionProcess(DiffusionProcess):
    r"""
    Diffusion processes, in arbitrary dimensions, with constant diffusion coefficient.

    It corresponds to the family of SDEs :math:`dx_t = F(x_t, t)dt + \sigma dW_t`,
    where :math:`F` is a time-dependent :math:`N`-dimensional vector field
    and :math:`W` the :math:`N`-dimensional Wiener process.
    The diffusion coefficient :math:`\sigma` is independent of the stochastic process
    (additive noise) and time, and we further assume that it is proportional to the identity matrix:
    all the components of the noise are independent.

    Parameters
    ----------
    vecfield : function with two arguments
        The vector field :math:`F(x, t)`.
    Damp : float
        The amplitude of the noise.
    dim : int
        The dimension of the system.

    Notes
    -----
    The diffusion coefficient is given by :math:`\sigma=\sqrt{2\text{Damp}}`.
    This convention leads to simpler expressions, for instance for the Fokker-Planck equations.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp, dim, **kwargs):
        """
        vecfield: vector field, function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise), scalar
        dim: dimension of the system

        In this class of stochastic processes, the diffusion matrix is proportional to identity.
        """
        DiffusionProcess.__init__(self, vecfield, (lambda x, t: np.sqrt(2*Damp)*np.eye(dim)),
                                  **kwargs)
        self.D0 = Damp
        self.dimension = dim

    def update(self, xn, tn, **kwargs):
        r"""
        Return the next sample for the time-discretized process.

        Parameters
        ----------
        xn: ndarray
            A n-dimensional vector (in :math:`\mathbb{R}^n`).
        tn: float
            The current time.

        Keyword Arguments
        ------------------
        dt : float
            The time step.
        dw : ndarray
            The brownian increment if precomputed.
            By default, it is generated on the fly from a Gaussian
            distribution with variance :math:`dt`.

        Returns
        -------
        x : ndarray
            The position at time tn+dt.

        See Also
        --------
        :meth:`DiffusionProcess.update` : for details about the Euler-Maruyama method.

        Notes
        -----
        This is the same as the :meth:`DiffusionProcess.update` method from the parent class
        :class:`DiffusionProcess`, except that a matrix product is no longer necessary.
        """
        dt = kwargs.get('dt', self.default_dt)
        if len(xn) != self.dimension:
            raise ValueError('Input vector does not have the right dimension.')
        dw = kwargs.get('dw', np.random.normal(0.0, np.sqrt(dt), self.dimension))
        return xn + self.drift(xn, tn)*dt+np.sqrt(2*self.D0)*dw


class OrnsteinUhlenbeck(ConstantDiffusionProcess):
    r"""
    The Ornstein-Uhlenbeck process, in arbitrary dimensions.

    It corresponds to the SDE :math:`dx_t = \theta(\mu-x_t)dt + \sqrt{2D} dW_t`,
    where :math:`\theta>0` and :math:`\mu \in \mathbb{R}^n` are arbitrary coefficients
    and :math:`D>0` is the amplitude of the noise.

    Parameters
    ----------
    mu : ndarray
        The expectation value.
    theta : float
        The inverse of the relaxation time.
    D : float
        The amplitude of the noise.
    dim : int
        The dimension of the system.

    Notes
    -----
    The Ornstein-Uhlenbeck process has been used to model many systems.
    It was initially introduced to describe the motion of a massive
    Brownian particle with friction [3]_ .
    It may also be seen as a diffusion process in a harmonic potential.

    Because many of its properties can be computed analytically, it provides a useful
    toy model for developing new methods.

    References
    ----------
    .. [3] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of Brownian Motion".
           Phys. Rev. 36, 823–841 (1930).
    """
    def __init__(self, mu, theta, D, dim, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(lambda x, t: theta*(mu-x), D, dim, **kwargs)
        self.theta = theta
        self.mu = mu

    def potential(self, x):
        r"""
        Compute the potential from which the force derives.

        Parameters
        ----------
        x : ndarray
            The point where we want to compute the potential

        Returns
        -------
        V : float
            The potential from which the force derives, at the given point.

        Notes
        -----
        Not all diffusion processes derive from a potential, but the Ornstein Uhlenbeck does.
        It is a gradient system, with a quadratic potential:
        :math:`dx_t = -\nabla V(x_t)dt + \sqrt{2D} dW_t`, with
        :math:`V(x) = \theta(\mu-x)^2/2`.
        """
        y = self.mu-x
        return self.theta*y.dot(y)/2

class Wiener(OrnsteinUhlenbeck):
    r"""
    The Wiener process, in arbitrary dimensions.

    Parameters
    ----------
    dim : int
        The dimension of the system.
    D : float, optional
        The amplitude of the noise (default is 1).

    Notes
    -----
    The Wiener process is a central object in the theory or stochastic processes,
    both from a mathematical point of view and for its applications in different scientific fields.
    We refer to classical textbooks for more information about the Wiener process
    and Brownian motion.
    """
    def __init__(self, dim, D=1, **kwargs):
        super(Wiener, self).__init__(0, 0, D, dim, **kwargs)

    @classmethod
    def potential(cls, x):
        r"""
        Compute the potential from which the force derives.

        Parameters
        ----------
        x : ndarray
            The point where we want to compute the potential.

        Returns
        -------
        V : float
            The potential from which the force derives, at the given point.

        Notes
        -----
        The Wiener Process is a trivial gradient system, with vanishing potential.
        It is useless (and potentially source of errors) to call the general potential routine,
        so we just return zero directly.
        """
        return np.zeros_like(x)
