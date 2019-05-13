"""
Generic class for 1D stochastic processes
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
    """
    Generic class for 1D diffusion processes.

    It corresponds to the family of 1D SDEs dx_t = F(x_t, t)dt + sigma(x_t, t)dW_t,
    where F is a time-dependent vector field and W the Wiener process.
    """
    default_dt = 0.1

    def __init__(self, F, sigma, **kwargs):
        """
        F and sigma are functions of two variables (x,t)
        """
        self.drift = F
        self.diffusion = sigma
        self.__deterministic__ = kwargs.get('deterministic', False)

    def increment(self, x, t, **kwargs):
        """ Return F(x_t, t)dt + sigma(x_t, t)dW_t """
        dt = kwargs.get('dt', self.default_dt)
        return self.drift(x, t) * dt + self.diffusion(x, t)*np.sqrt(dt)*np.random.normal(0.0, 1.0)

    def update(self, x, t, **kwargs):
        """
        Return the next sample for the time-discretized process
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
        """
        Integrate a trajectory with given initial condition (t0,x0)
        Optional arguments:
        - dt: float, the time step
        - T: float, the integration time (i.e. the duration of the trajectory)
        - finite: boolean, whether to filter output to return only finite values (default False)
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
        """
        Return a trajectory satisfying a condition defined by the predicate pred.
        """
        while True:
            t, x = self.trajectory(x0, t0, **kwargs)
            if pred(t, x):
                break
        return t, x


class ConstantDiffusionProcess1D(DiffusionProcess1D):
    """
    Diffusion process with constant diffusion coefficient.

    It corresponds to the family of 1D SDEs dx_t = F(x_t,t)dt + sqrt(2*D0)dW_t,
    where F is a time-dependent vector field and W the Wiener process.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp, **kwargs):
        """
        vecfield: function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise), scalar
        """
        DiffusionProcess1D.__init__(self, vecfield, lambda x,t: np.sqrt(2*Damp), **kwargs)
        self.F = vecfield # We keep this temporarily for backward compatiblity
        self.D0 = Damp    # We keep this temporarily for backward compatiblity

    def potential(self, X, t):
        """
        Integrate the vector field to obtain the value of the underlying potential
        at the input points.
        Caveat: This works only because we restrict ourselves to 1D models.
        """
        fun = interp1d(X, -self.F(X, t))
        return np.array([integrate.quad(fun, 0.0, x)[0] for x in X])

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

    @classmethod
    def traj_fpt(cls, M, *args):
        """ Compute the first passage time for each trajectory given as argument """
        for tt, xx in args:
            for t, x in zip(tt, xx):
                if x > M:
                    yield t
                    break

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

    def blowuptime(self, x0, t0, **kwargs):
        """ Compute the last time with finite values, for one realization"""
        t, x = self.trajectory(x0, t0, **kwargs)
        return t[np.isfinite(x)][-1]

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        pass

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

    # First passage time problems:
    def firstpassagetime(self, x0, t0, A, **kwargs):
        """
        Computes the first passage time, defined by $\tau_A = inf{t>t0 | x(t)>A}$,
        for one realization
        """
        x = x0
        t = t0
        dt = kwargs.get('dt', self.default_dt)
        while x <= A:
            x = self.update(x, t, dt=dt)
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
        fpe = fp.FokkerPlanck1D(self.F, self.D0)
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
                return integrate.trapz(np.exp(sign*self.potential(z, t0)/self.D0)*fun(z), z)
            # compute the inner integral and interpolate:
            y = np.linspace(x0, args[-1])
            arr = np.array([exppot_int(*u) for u in [(inf, y[0])]+zip(y[:-1], y[1:])])
            ifun = interp1d(y, arr.cumsum())
            # now compute the outer integral by chunks
            return np.concatenate((badargs, args)), np.array(len(badargs)*[0.]+[exppot_int(*bds, sign=1, fun=ifun) for bds in [(x0, args[0])]+zip(args[:-1], args[1:])]).cumsum()/self.D0
        elif src == 'theory2':
            def exppot(y, sign=-1, fun=lambda z: 1):
                return np.exp(sign*self.potential(y, t0)/self.D0)*fun(y)
            # compute the inner integral and interpolate:
            z = np.linspace(inf, args[-1])
            iarr = integrate.cumtrapz(exppot(z), z, initial=0)
            ifun = interp1d(z, iarr)
            # now compute the outer integral by chunks
            y = np.linspace(x0, args[-1])
            oarr = integrate.cumtrapz(exppot(y, sign=1, fun=ifun), y, initial=0)/self.D0
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


class Wiener1D(ConstantDiffusionProcess1D):
    """ The 1D Wiener process """
    def __init__(self, D=1, **kwargs):
        super(Wiener1D, self).__init__(lambda x, t: 0, D, **kwargs)

    def potential(self, X, t):
        """
        Useless (and potentially source of errors) to call the general potential routine
        since it is trivially zero here
        """
        return np.zeros_like(X)

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the heat equation.
        This should depend on the boundary conditions.
        Right now, we do as if we were solving on the real axis."""
        return np.exp(-X**2.0/(4.0*self.D0*t))/np.sqrt(4.0*np.pi*self.D0*t)


class OrnsteinUhlenbeck1D(ConstantDiffusionProcess1D):
    """
    The 1D Ornstein-Uhlenbeck model:
        dx_t = theta*(mu-x_t)+sqrt(2*D)*dW_t
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
