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
        for one realization
        """
        x = x0
        t = t0
        dt = kwargs.get('dt', self.model.default_dt)
        while x <= A:
            x = self.model.update(x, t, dt=dt)
            t += dt
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

    @classmethod
    def traj_fpt(cls, M, *args):
        """ Compute the first passage time for each trajectory given as argument """
        for tt, xx in args:
            for t, x in zip(tt, xx):
                if x > M:
                    yield t
                    break


    def firstpassagetime_cdf(self, x0, A, *args, **kwargs):
        """
        Computes the CDF of the first passage time, Prob_{x0,t0}[\tau_A<t]
        by solving the Fokker-Planck equation
        """
        t0 = kwargs.pop('t0', 0.0)
        if 'P0' in kwargs:
            del kwargs['P0']
        if 'P0center' in kwargs:
            del kwargs['P0center']
        if 'bc' not in kwargs:
            kwargs['bc'] = ('reflecting', 'absorbing')
        bnds = (kwargs.pop('bounds', (-10.0, 0.0))[0], A)
        time = np.sort([t0]+list(args))
        time = time[time >= t0]
        G = [1.0 if x0 < A else 0.0]
        fpe = fp.FokkerPlanck1D(self.model.F, self.model.D0)
        t, X, P = fpe.fpintegrate(t0, 0.0, P0='dirac', P0center=x0, bounds=bnds, **kwargs)
        for t in time[1:]:
            t, X, P = fpe.fpintegrate(t0, t-t0, P0=P, bounds=bnds, **kwargs)
            G += [integrate.trapz(P[X < A], X[X < A])]
            t0 = t
        G = np.array(G)
        output = {'cdf': (time, 1.0-G), 'G': (time, G),
                  'pdf': (time[1:-1], -edpy.CenteredFD(time).grad(G)),
                  'lambda': (time[1:-1], -edpy.CenteredFD(time).grad(np.log(G)))}
        return output.get(kwargs.get('out', 'G'))

    def firstpassagetime_moments(self, x0, A, *args, **kwargs):
        """
        Computes the moments of the first passage time, <tau_A^n>_{x0,t0},
        by solving the Fokker-Planck equation
        """
        t0 = kwargs.get('t0', 0.0)
        tmax = kwargs.pop('tmax', 10.0)
        nt = kwargs.pop('nt', 10)
        times = np.linspace(t0, tmax, num=nt)
        _, cdf = self.firstpassagetime_cdf(x0, A, *times, out='cdf', **kwargs)
        Mn = []
        for n in args:
            Mn += [t0**n + n*integrate.trapz(cdf*times**(n-1), times)]
        return Mn

    def firstpassagetime_avg(self, x0, *args, **kwargs):
        """
        Compute the mean first passage time by one of the following methods:
        solving the FP equation, its adjoint, or using the theoretical solution.

        x0 is the initial condition (at t0), and 'args' contains the list of
        threshold values for which to compute the first passage time.

        The theoretical formula is valid only for an homogeneous process;
        for the computation, we 'freeze' the potential at t=t0.
        """
        src = kwargs.pop('src', 'FP')
        tmax = kwargs.pop('tmax', 100.0)
        nt = kwargs.pop('nt', 10)
        t0 = kwargs.pop('t0', 0.0)
        inf = kwargs.pop('inf', -10.0)
        # args have to be sorted in increasing order:
        args = np.sort(args)
        # remove the values of args which are <= x0:
        badargs, args = (args[args <= x0], args[args > x0])
        if src == 'theory':
            def exppot_int(a, b, sign=-1, fun=lambda z: 1):
                z = np.linspace(a, b)
                return integrate.trapz(np.exp(sign*self.model.potential(z, t0)/self.model.D0)*fun(z), z)
            # compute the inner integral and interpolate:
            y = np.linspace(x0, args[-1])
            arr = np.array([exppot_int(*u) for u in [(inf, y[0])]+zip(y[:-1], y[1:])])
            ifun = interp1d(y, arr.cumsum())
            # now compute the outer integral by chunks
            return np.concatenate((badargs, args)), np.array(len(badargs)*[0.]+[exppot_int(*bds, sign=1, fun=ifun) for bds in [(x0, args[0])]+zip(args[:-1], args[1:])]).cumsum()/self.model.D0
        elif src == 'theory2':
            def exppot(y, sign=-1, fun=lambda z: 1):
                return np.exp(sign*self.model.potential(y, t0)/self.model.D0)*fun(y)
            # compute the inner integral and interpolate:
            z = np.linspace(inf, args[-1])
            iarr = integrate.cumtrapz(exppot(z), z, initial=0)
            ifun = interp1d(z, iarr)
            # now compute the outer integral by chunks
            y = np.linspace(x0, args[-1])
            oarr = integrate.cumtrapz(exppot(y, sign=1, fun=ifun), y, initial=0)/self.model.D0
            ofun = interp1d(y, oarr)
            return np.concatenate((badargs, args)), np.concatenate((len(badargs)*[0.], ofun(args)))
        elif src == 'adjoint':
            # here we need to solve the adjoint FP equation for each threshold value,
            # so this is much more expensive than the theoretical formula of course.
            def interp_int(G, t):
                logG = interp1d(t, np.log(G), fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)), 0.0, np.inf)[0] # careful: this is not the right expression for arbitrary t0 !!
            integ_method = {True: interp_int,
                            False: integrate.trapz}.get(kwargs.pop('interpolate', True))
            return np.concatenate((badargs, args)), np.array(len(badargs)*[0.]+[integ_method(*(self.firstpassagetime_cdf(x0, A, *np.linspace(0.0, tmax, num=nt), t0=t0, out='G', src='adjoint', **kwargs)[::-1])) for A in args])
#        elif src in ('FP','quad'):
        elif src == 'FP':
            # here we need to solve the FP equation for each threshold value,
            # so this is much more expensive than the theoretical formula of course.
            def interp_int(G, t):
                logG = interp1d(t, np.log(G), fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)), t0, np.inf)[0]
            integ_method = {True: interp_int,
                            False: integrate.trapz}.get(kwargs.pop('interpolate', True))
            return np.concatenate((badargs, args)), np.array(len(badargs)*[0.]+[t0+integ_method(*(self.firstpassagetime_cdf(x0, A, *np.linspace(t0, tmax, num=nt), t0=t0, out='G', **kwargs)[::-1])) for A in args])
        else:
            raise NotImplementedError('Unrecognized method for computing first passage time')
