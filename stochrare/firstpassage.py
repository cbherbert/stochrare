"""
First-passage processes
=======================

.. currentmodule:: stochrare.firstpassage

This module defines a class corresponding to the random variable defined as the first-passage time
in a given set for a given stochastic process.

.. autoclass:: FirstPassageProcess
   :members:

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from numba import jit
from . import edpy
from . import fokkerplanck as fp

class FirstPassageProcess:
    """
    Represents a first-passage time random variable associated to a stochastic process and a given
    set.

    Parameters
    ----------
    model : stochrare.dynamics.DiffusionProcess1D
        The stochastic process to which the first-passage time is associated

    CAUTION: methods only tested with ConstantDiffusionProcess1D class, not DiffusionProcess1D!
    """

    def __init__(self, model):
        self.model = model

    def firstpassagetime(self, x0, t0, A, **kwargs):
        """
        Computes the first passage time, defined by $\tau_A = inf{t>t0 | x(t)>A}$,
        for one realization.

        Parameters
        ----------
        x0: float
            The initial position
        t0: float
            The initial time
        A: float
            The threshold

        Returns
        -------
        t: float
            A realization of the first-passage time
        """
        dt = kwargs.get('dt', self.model.default_dt)
        return self._fpt_euler(x0, t0, A, dt, self.model.drift, self.model.diffusion)

    @staticmethod
    @jit(nopython=True)
    def _fpt_euler(x0, t0, A, dt, drift, diffusion):
        x = x0
        t = t0
        while x <= A:
            w = np.random.normal(0, np.sqrt(dt))
            x = x + drift(x, t)*dt + diffusion(x, t)*w
            t = t + dt
        return t

    def escapetime_sample(self, x0, t0, A, **kwargs):
        """
        Computes realizations of the first passage time, defined by $\tau_A = inf{t>t0 | x(t)>A}$,
        using direct Monte-Carlo simulations.
        This method can be overwritten by subclasses to call compiled code for better performance.
        """
        ntraj = kwargs.pop('ntraj', 100000)
        dtype = kwargs.pop('dtype', np.float32)
        return np.array([self.firstpassagetime(x0, t0, A, **kwargs) for _ in range(ntraj)],
                        dtype=dtype)

    def escapetime_avg(self, x0, t0, A, **kwargs):
        """ Compute the average escape time for given initial condition (x0,t0) and threshold A """
        return np.mean(self.escapetime_sample(x0, t0, A, **kwargs))

    @classmethod
    def escapetime_pdf(cls, samples, **kwargs):
        """
        Compute the probability distribution function of the first-passage time
        based on the input samples
        """
        avg, std = {True: (np.mean(samples), np.std(samples)),
                    False: (0., 1.0)}.get(kwargs.get('standardize', False))
        hist, rc = np.histogram((samples-avg)/std, bins=kwargs.get('bins', 'doane'), density=True)
        rc = rc[:-1] + 0.5*(rc[1]-rc[0])
        return rc, hist

    @classmethod
    def escapetime_pdfplot(cls, *args, **kwargs):
        """ Plot previously computed pdf of first passage time """
        _ = plt.figure()
        ax = plt.axes()
        lines = []
        for t, p in args:
            lines += ax.plot(t, p, linewidth=2)

        ax.grid()
        ax.set_xlabel(r'$\tau_M$')
        ax.set_ylabel(r'$p(\tau_M)$')
        ax.set_yscale(kwargs.get('yscale', 'linear'))

        plottitle = kwargs.get('title', "")
        if plottitle != "":
            plt.title(plottitle)

        labels = kwargs.get('labels', [])
        if labels != []:
            plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()


    def firstpassagetime_cdf(self, x0, A, *args, **kwargs):
        """
        Computes the CDF of the first passage time, :math:`Prob_{x0,t0}[\tau_A<t]`,
        or its derivatives, by solving the Fokker-Planck equation.
        """
        t0 = kwargs.pop('t0', 0.0)
        if 'P0' in kwargs:
            del kwargs['P0']
        if 'bc' not in kwargs:
            kwargs['bc'] = ('reflecting', 'absorbing')
        bnds = (kwargs.pop('bounds', (-10.0, 0.0))[0], A)
        time = np.sort([t0]+list(args))
        time = time[time >= t0]
        G = [1.0 if x0 < A else 0.0]
        fpe = fp.FokkerPlanck1D.from_sde(self.model)
        P0 = fpe.dirac1d(x0, np.linspace(bnds[0], bnds[1], num=kwargs.get('npts', 100)))
        t, X, P = fpe.fpintegrate(t0, 0.0, P0=P0, bounds=bnds, **kwargs)
        for t in time[1:]:
            t, X, P = fpe.fpintegrate(t0, t-t0, P0=P, bounds=bnds, **kwargs)
            G += [integrate.trapz(P[X < A], X[X < A])]
            t0 = t
        G = np.array(G)
        output = {'cdf': (time, 1.0-G), 'G': (time, G),
                  'pdf': (time[1:-1], -edpy.CenteredFD(time).grad(G)),
                  'lambda': (time[1:-1], -edpy.CenteredFD(time).grad(np.log(G)))}
        return output.get(kwargs.get('out', 'G'))

    def firstpassagetime_cdf_adjoint(self, x0, A, *args, **kwargs):
        """
        Computes the CDF of the first passage time, :math:`Prob_{x0,t0}[\tau_A<t]`,
        or its derivatives, by solving the adjoint Fokker-Planck equation.
        """
        t0 = kwargs.pop('t0', 0.0)
        if 'P0' in kwargs:
            del kwargs['P0']
        if 'bc' not in kwargs:
            kwargs['bc'] = ('reflecting', 'absorbing')
        time = np.sort([t0]+list(args))
        time = time[time >= t0]
        bnds = (kwargs.pop('bounds', (-10.0, 0.0))[0], A)
        fpe = fp.FokkerPlanck1DBackward(self.model.drift,
                                        lambda x, t: 0.5*self.model.diffusion(x, t)**2)
        Gloc = [1.0 if x0 < A else 0.0]
        G0 = np.ones(kwargs.get('npts', 100))
        G0[np.linspace(bnds[0], bnds[1], kwargs.get('npts', 100)) >= A] = 0.0
        t, X, G = fpe.fpintegrate(t0, 0.0, P0=G0, bounds=bnds, **kwargs)
        for t in time[1:]:
            t, X, G = fpe.fpintegrate(t0, t-t0, P0=G, bounds=bnds, **kwargs)
            Gloc += [G[X <= x0][-1]]
            t0 = t
        Gloc = np.array(Gloc)
        output = {'cdf': (time, 1.0-Gloc), 'G': (time, Gloc),
                  'pdf': (time[1:-1], -edpy.CenteredFD(time).grad(Gloc)),
                  'lambda': (time[1:-1], -edpy.CenteredFD(time).grad(np.log(Gloc)))}
        return output.get(kwargs.get('out', 'G'))

    def firstpassagetime_avg_theory(self, x0, *args, **kwargs):
        r"""
        Compute the mean first-passage time using the theoretical formula:

        :math:`\mathbb{E}[\tau_M] = \frac{1}{D} \int_{x_0}^{M} dx e^{V(x)/D} \int_{-\infty}^x e^{-V(y)/D}dy.`

        This formula is valid for a homogeneous process, conditioned on the initial position :math:`x_0`.

        Parameters
        ----------
        x0 : float
            Initial position
        """
        t0 = kwargs.pop('t0', 0.0)
        inf = kwargs.pop('inf', -10.0)
        # args have to be sorted in increasing order:
        args = np.sort(args)
        # remove the values of args which are <= x0:
        badargs, args = (args[args <= x0], args[args > x0])

        def exppot_int(a, b, sign=-1, fun=lambda z: 1):
            z = np.linspace(a, b, **kwargs)
            return integrate.trapz(np.exp(sign*self.model.potential(z, t0)/self.model.D0)*fun(z), z)
        # compute the inner integral and interpolate:
        y = np.linspace(x0, args[-1], **kwargs)
        arr = np.array([exppot_int(*u) for u in [(inf, y[0])]+list(zip(y[:-1], y[1:]))])
        ifun = interp1d(y, arr.cumsum(), fill_value='extrapolate')
        # now compute the outer integral by chunks
        oint = np.array([exppot_int(*bds, sign=1, fun=ifun) for bds in [(x0, args[0])]+list(zip(args[:-1], args[1:]))]).cumsum()/self.model.D0
        return np.concatenate((badargs, args)), np.concatenate((np.zeros_like(badargs), oint))

    def firstpassagetime_avg_theory2(self, x0, *args, **kwargs):
        r"""
        Compute the mean first-passage time using the theoretical formula:

        :math:`\mathbb{E}[\tau_M] = \frac{1}{D} \int_{x_0}^{M} dx e^{V(x)/D} \int_{-\infty}^x e^{-V(y)/D}dy.`

        This formula is valid for a homogeneous process, conditioned on the initial position :math:`x_0`.

        Parameters
        ----------
        x0 : float
            Initial position
        """
        t0 = kwargs.pop('t0', 0.0)
        inf = kwargs.pop('inf', -10.0)
        # args have to be sorted in increasing order:
        args = np.sort(args)
        # remove the values of args which are <= x0:
        badargs, args = (args[args <= x0], args[args > x0])

        def exppot(y, sign=-1, fun=lambda z: 1):
            return np.exp(sign*self.model.potential(y, t0)/self.model.D0)*fun(y)
        # compute the inner integral and interpolate:
        z = np.linspace(inf, args[-1], **kwargs)
        iarr = integrate.cumtrapz(exppot(z), z, initial=0)
        ifun = interp1d(z, iarr)
        # now compute the outer integral by chunks
        y = np.linspace(x0, args[-1], **kwargs)
        oarr = integrate.cumtrapz(exppot(y, sign=1, fun=ifun), y, initial=0)/self.model.D0
        ofun = interp1d(y, oarr)
        return np.concatenate((badargs, args)), np.concatenate((np.zeros_like(badargs), ofun(args)))
