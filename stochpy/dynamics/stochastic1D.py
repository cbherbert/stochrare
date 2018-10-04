"""
Generic class for 1D stochastic processes
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.sparse as sps
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from .. import edpy, data

class StochModel1D(object):
    """
    The generic class from which all the models I consider derive.
    It corresponds to the family of 1D SDEs dx_t = F(x_t,t)dt + sqrt(2*D0)dW_t,
    where F is a time-dependent vector field and W the Wiener process.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp):
        """
        vecfield: function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise)
        """
        self.F = vecfield
        self.D0 = Damp

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
        dt = kwargs.get('dt', self.default_dt)
        time = kwargs.get('T', 10.0)
        if dt < 0:
            time = -time
        while t[-1] < t0+time:
            t += [t[-1] + dt]
            x += [x[-1] + self.increment(x[-1], t[-1], dt=dt)]
        t = np.array(t)
        x = np.array(x)
        if kwargs.get('finite', False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t[t <= t0+time], x[t <= t0+time]

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
                x += [x[-1] + self.increment(x[-1], t[-1], dt=dt)]
                t += [t[-1] + dt]
            if (x[-1] > M and np.abs(t[-1]-tau) < tau_tol):
                num -= 1
                if interp:
                    fun = interp1d(t, x, fill_value='extrapolate')
                    t = np.linspace(t0, tau, num=npts)
                    x = fun(t)
                yield t, x

    @classmethod
    def traj_fpt(self, M, *args):
        """ Compute the first passage time for each trajectory given as argument """
        for tt, xx in args:
            for t, x in zip(tt, xx):
                if x > M:
                    yield t
                    break

    @classmethod
    def trajectoryplot(cls, *args, **kwargs):
        """ Plot previously computed trajectories """
        fig = plt.figure()
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

    def _fpeq(self, P, X, t):
        """ Right hand side of the Fokker-Planck equation associated to the stochastic process """
        return -X.grad(self.F(X.grid, t)*P) + self.D0*X.laplacian(P)

    def _fpadj(self, G, X, t):
        """
        The adjoint of the Fokker-Planck operator, useful for instance
        in first passage time problems for homogeneous processes.
        """
        return self.F(X.grid, t)[1:-1]*X.grad(G)+self.D0*X.laplacian(G)

    def _fpmat(self, X, t):
        """
        Sparse matrix representation of the linear operator
        corresponding to the RHS of the FP equation
        """
        return -X.grad_mat()*sps.dia_matrix((self.F(X.grid, t), np.array([0])), shape=(X.N, X.N)) + self.D0*X.lapl_mat()

    def _fpadjmat(self, X, t):
        """ Sparse matrix representation of the adjoint of the FP operator """
        return sps.dia_matrix((self.F(X.grid, t)[1:-1], np.array([0])), shape=(X.N-2, X.N-2))*X.grad_mat() + self.D0*X.lapl_mat()

    def _fpbc(self, fdgrid, bc=('absorbing', 'absorbing'), **kwargs):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dx = fdgrid.dx
        dic = {('absorbing', 'absorbing'): edpy.DirichletBC([0, 0]),
               ('absorbing', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [0,Y[-2]/(1-self.F(X[-1], t)*dx/self.D0)]),
               ('reflecting', 'absorbing'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.F(X[0], t)*dx/self.D0),0]),
               ('reflecting', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.F(X[0], t)*dx/self.D0), Y[-2]/(1-self.F(X[-1], t)*dx/self.D0)])}
        if bc in dic:
            return edpy.DirichletBC([0, 0]) if self.D0 == 0 else dic[bc]
        else:
            return bc

    def _fpthsol(self, X, t, **kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        pass

    def fpintegrate(self, t0, T, **kwargs):
        """
        Numerical integration of the associated Fokker-Planck equation, or its adjoint.
        Optional arguments are the following:
        - bounds=(-10.0,10.0); domain where we should solve the equation
        - npts=100;            number of discretization points in the domain (i.e. spatial resolution)
        - dt;                  timestep (default choice suitable for the heat equation with forward scheme)
        - bc;                  boundary conditions (either a BoundaryCondition object or a tuple sent to _fpbc)
        - method=euler;        numerical scheme: explicit (default), implicit, or crank-nicolson
        - adj=False;           integrate the adjoint FP rather than the forward FP?
        """
        # Get computational parameters:
        B, A = kwargs.pop('bounds', (-10.0, 10.0))
        Np = kwargs.pop('npts', 100)
        fdgrid = edpy.RegularCenteredFD(B, A, Np)
        dt = kwargs.pop('dt', 0.25*(np.abs(B-A)/(Np-1))**2/self.D0)
        bc = self._fpbc(fdgrid, **kwargs)
        method = kwargs.pop('method', 'euler')
        adj = kwargs.pop('adjoint', False)
        # Prepare initial P(x):
        P0 = kwargs.pop('P0', 'gauss')
        if P0 is 'gauss':
            P0 = np.exp(-0.5*((fdgrid.grid-kwargs.get('P0center', 0.0))/kwargs.get('P0std', 1.0))**2)/(np.sqrt(2*np.pi)*kwargs.get('P0std', 1.0))
            P0 /= integrate.trapz(P0, fdgrid.grid)
        if P0 is 'dirac':
            P0 = np.zeros_like(fdgrid.grid)
            np.put(P0, len(fdgrid.grid[fdgrid.grid < kwargs.get('P0center', 0.0)]), 1.0)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        if P0 is 'uniform':
            P0 = np.ones_like(fdgrid.grid)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        # Numerical integration:
        if T > 0:
            if method in ('impl', 'implicit', 'bwd', 'backward', 'cn', 'cranknicolson', 'crank-nicolson'):
                fpmat = {False: self._fpmat, True: self._fpadjmat}.get(adj)
                return edpy.EDPLinSolver().edp_int(fpmat, fdgrid, P0, t0, T, dt, bc, scheme=method)
            else:
                fpfun = {False: self._fpeq, True: self._fpadj}.get(adj)
                return edpy.EDPSolver().edp_int(fpfun, fdgrid, P0, t0, T, dt, bc)
        else:
            return t0, fdgrid.grid, P0

    def pdfgen(self, *args, **kwargs):
        """ Generate the pdf solution of the FP equation at various times """
        t0 = kwargs.pop('t0', args[0])
        fun = kwargs.pop('integ', self.fpintegrate)
        for t in args:
            t, X, P = fun(t0, t-t0, **kwargs)
            t0 = t
            kwargs['P0'] = P
            yield t, X, P


    def pdfplot(self, *args, **kwargs):
        """ Plot the pdf P(x,t) at various times """
        fig = plt.figure()
        ax = plt.axes()
        t0 = kwargs.pop('t0', args[0])
        fun = kwargs.pop('integ', self.fpintegrate)
        if kwargs.get('potential', False):
            ax2 = ax.twinx()
            ax2.set_ylabel('$V(x,t)$')
        for t in args:
            t, X, P = fun(t0, t-t0, **kwargs)
            line, = ax.plot(X, P, label='t='+format(t, '.2f'))
            if kwargs.get('th', False):
                Pth = self._fpthsol(X, t, **kwargs)
                if Pth is not None: ax.plot(X, Pth, color=line.get_color(), linestyle='dotted')
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
        while (x <= A):
            x += self.increment(x, t, dt=dt)
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
        return np.array([firstpassagetime(x0, t0, A, **kwargs) for n in xrange(ntraj)], dtype=dtype)

    def escapetime_avg(self, x0, t0, A, **kwargs):
        """ Compute the average escape time for given initial condition (x0,t0) and threshold A """
        return np.mean(self.escapetime_sample(x0, t0, A, **kwargs))

    @classmethod
    def escapetime_pdf(self, samples, **kwargs):
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
    def escapetime_pdfplot(self, *args, **kwargs):
        """ Plot previously computed pdf of first passage time """
        fig = plt.figure()
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
        if 'P0' in kwargs: del kwargs['P0']
        if 'P0center' in kwargs: del kwargs['P0center']
        if 'bc' not in kwargs: kwargs['bc'] = ('reflecting', 'absorbing')
        bnds = (kwargs.pop('bounds', (-10.0, 0.0))[0], A)
        time = np.sort([t0]+list(args))
        time = time[time >= t0]
        G = [1.0 if x0 < A else 0.0]
        t, X, P = self.fpintegrate(t0, 0.0, P0='dirac', P0center=x0, bounds=bnds, **kwargs)
        for t in time[1:]:
            t, X, P = self.fpintegrate(t0, t-t0, P0=P, bounds=bnds, **kwargs)
            G += [integrate.trapz(P[X < A], X[X < A])]
            t0 = t
        G = np.array(G)
        output = {'cdf': (time, 1.0-G), 'G': (time, G),
                  'pdf': (time[1:-1], -edpy.CenteredFD(time).grad(G)),
                  'lambda': (time[1:-1], -edpy.CenteredFD(time).grad(np.log(G)))}
        return output.get(kwargs.get('out', 'G'))

    def firstpassagetime_moments(self, x0, A, *args, **kwargs):
        """
        Computes the moments of the first passage time, $\langle \tau_A^n \rangle_{x0,t0}$,
        by solving the Fokker-Planck equation
        """
        t0 = kwargs.get('t0', 0.0)
        tmax = kwargs.pop('tmax', 10.0)
        nt = kwargs.pop('nt', 10)
        times = np.linspace(t0, tmax, num=nt)
        t, cdf = self.firstpassagetime_cdf(x0, A, *times, out='cdf', **kwargs)
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
            return np.concatenate((badargs, args)),np.array(len(badargs)*[0.]+[exppot_int(*bds, sign=1, fun=ifun) for bds in [(x0, args[0])]+zip(args[:-1], args[1:])]).cumsum()/self.D0
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
            # here we need to solve the adjoint FP equation for each threshold value, so this is much more expensive than the theoretical formula of course.
            def interp_int(G, t):
                logG = interp1d(t, np.log(G), fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)), 0.0, np.inf)[0] # careful: this is not the right expression for arbitrary t0 !!
            integ_method = {True: interp_int,
                            False: integrate.trapz}.get(kwargs.pop('interpolate', True))
            return np.concatenate((badargs, args)), np.array(len(badargs)*[0.]+[integ_method(*(self.firstpassagetime_cdf(x0, A, *np.linspace(0.0, tmax, num=nt), t0=t0, out='G', src='adjoint', **kwargs)[::-1])) for A in args])
#        elif src in ('FP','quad'):
        elif src == 'FP':
            # here we need to solve the FP equation for each threshold value, so this is much more expensive than the theoretical formula of course.
            def interp_int(G, t):
                logG = interp1d(t, np.log(G), fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)), t0, np.inf)[0]
            integ_method = {True: interp_int,
                            False: integrate.trapz}.get(kwargs.pop('interpolate', True))
            return np.concatenate((badargs, args)),np.array(len(badargs)*[0.]+[t0+integ_method(*(self.firstpassagetime_cdf(x0, A, *np.linspace(t0, tmax, num=nt), t0=t0, out='G', **kwargs)[::-1])) for A in args])
        else:
            pass


    def instanton(self, x0, p0, *args, **kwargs):
        """
        Numerical integration of the equations of motion for instantons.
        x0 and p0 are the initial conditions.
        Return the instanton trajectory (t,x).
        """
        def rev(f):
            return lambda t, Y: f(Y, t)
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
        var = kwargs.pop('change_vars', 'None')
        filt_traj = kwargs.pop('filter_traj', False)
        back = kwargs.pop('backwards', False)
        times = np.sort(args)
        fun = self._instantoneq
        jac = self._instantoneq_jac
        if var == 'log':
            fun = self._instantoneq_log
            jac = self._instantoneq_jac_log
        if back:
            fun = inverse(fun)
            jac = inverse(jac)
        if solver == 'odeint':
            x = integrate.odeint(fun, (x0, p0), times, **kwargs)[:, 0]
            return filt_fun(times, x) if filt_traj else (times, x)
        elif solver == 'odeclass':
            integ = integrate.ode(rev(fun), jac=rev(jac)).set_integrator(scheme, **kwargs)
            integ.set_initial_value([x0, p0], times[0])
            return times, [integ.integrate(t)[0] for t in times]


class Wiener1D(StochModel1D):
    """ The 1D Wiener process """
    def __init__(self, D=1):
        super(Wiener1D, self).__init__(lambda x, t: 0, D)

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


class OrnsteinUhlenbeck1D(StochModel1D):
    """
    The 1D Ornstein-Uhlenbeck model:
        dx_t = theta*(mu-x_t)+sqrt(2*D)*dW_t
    """
    def __init__(self, mu, theta, D):
        super(OrnsteinUhlenbeck1D, self).__init__(lambda x, t: theta*(mu-x), D)


class StochModel1D_T(StochModel1D):
    """ Time reversal of a given model """
    def trajectory(self, x0, t0, **kwargs):
        t, x = super(StochModel1D_T, self).trajectory(x0, t0, **kwargs)
        return 2*t[0]-t, x
