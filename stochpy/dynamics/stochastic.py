"""
Generic class for stochastic processes in arbitrary dimensions
"""
import numpy as np
import scipy.integrate

class StochModel(object):
    """
    Generic class for stochastic processes in arbitrary dimensions.

    More precisely, this class is currently limited to a special case of stochastic differential
    equations: deterministic vector fields perturbed by white noise.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp):
        """
        vecfield: vector field, function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise)

        For now we assume that the diffusion matrix is proportional to identity.
        This is a strong limitation which hopefully should be relaxed soon.
        """
        self.F = vecfield
        self.D0 = Damp

    def increment(self, x, t, **kwargs):
        """
        Return F(x_t, t)dt + sqrt(2*D0)dW_t
        x is a n-dimensional vector (in R^n)
        t is the time (a real number)
        """
        dt = kwargs.get('dt', self.default_dt)
        dim = len(x)
        return self.F(x, t) * dt + np.sqrt(2.0*self.D0*dt)*np.random.normal(0.0, 1.0, dim)

    def trajectory(self, x0, t0, **kwargs):
        """
        Integrate a trajectory with given initial condition (t0, x0)
        Optional arugments:
        - dt: the timestep, forwarded to the increment routine
              (default 0.1, unless overridden by a subclass)
        - T: the time duration of the trajectory (default 10)
        - finite: filter finite values before returning trajectory
        """
        x = [x0]
        dt = kwargs.get('dt', self.default_dt) # Time step
        time = kwargs.get('T', 10.0)   # Total integration time
        if dt < 0:
            time = -time
        tarray = np.linspace(t0, t0+time, num=np.floor(time/dt)+1)
        for t in tarray[1:]:
            x += [x[-1] + self.increment(x[-1], t, dt=dt)]
        x = np.array(x)
        if kwargs.get('finite', False):
            tarray = tarray[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return tarray, x

    def generator(self, x0, t0, nsteps, **kwargs):
        """
        A generator for the stochastic process
        """
        x = x0
        t = t0
        dt = kwargs.get('dt', self.default_dt) # Time step
        obs = kwargs.get('observable', lambda x: x)
        for _ in range(nsteps):
            t = t + dt
            x = x + self.increment(x, t, dt=dt)
            yield t, obs(x)

    def sample_mean(self, x0, t0, nsteps, nsamples, **kwargs):
        """
        Compute the sample mean of a time dependent observable
        """
        gens = [self.generator(x0, t0, nsteps, **kwargs) for _ in range(nsamples)]
        while True:
            time, obs = zip(*[next(gen) for gen in gens])
            yield np.average(time, axis=0), np.average(obs, axis=0)

    def instanton(self, x0, p0, *args, **kwargs):
        """
        Numerical integration of the equations of motion for instantons.
        x0 and p0 are the initial conditions.
        Return the instanton trajectory (t,x).
        """
        solver = kwargs.pop('solver', 'odeint')
        scheme = kwargs.pop('integrator', 'dopri5')
        times = np.sort(args)
        if solver == 'odeint':
            x = scipy.integrate.odeint(self._instantoneq, np.concatenate((x0, p0)), times,
                                       Dfun=self._instantoneq_jac, tfirst=True, **kwargs)
        elif solver == 'odeclass':
            integ = scipy.integrate.ode(self._instantoneq,
                                        jac=self._instantoneq_jac).set_integrator(scheme, **kwargs)
            integ.set_initial_value(np.concatenate((x0, p0)), t=times[0])
            x = np.array([integ.integrate(t) for t in times])
        return times, x

    def _instantoneq(self, t, canonical_coords):
        """
        Equations of motion for instanton dynamics.
        These are just the Hamilton equations corresponding to the action.

        canonical_coords should be a 2n-dimensional numpy.ndarray
        containing canonical coordinates and momenta.

        Return a numpy.ndarray of length 2n containing the derivatives xdot and pdot.
        """
        n = len(canonical_coords)/2
        x = canonical_coords[:n]
        p = canonical_coords[n:]
        raise NotImplementedError("Generic instanton equations not yet implemented")
    #     return np.concatenate((2.*p+self.F(x, t),
    #                            -np.dot(p, derivative(self.F, x, dx=1e-6, args=(t, )))))

    def _instantoneq_jac(self, t, canonical_coords):
        """
        Jacobian of instanton dynamics.

        canonical_coords should be a 2n-dimensional numpy.ndarray
        containing canonical coordinates and momenta.
        """
        n = len(canonical_coords)/2
        x = canonical_coords[:n]
        p = canonical_coords[n:]
        raise NotImplementedError("Generic instanton equations Jacobian not yet implemented")

        # dbdx = derivative(self.F, x, dx=1e-6, args=(t, ))
        # return np.array([[dbdx, 2.],
        #                  [-p*derivative(self.F, x, n=2, dx=1e-6, args=(t, )), -dbdx]])

class Wiener(StochModel):
    """ The Wiener process """
    def __init__(self, D=1):
        super(Wiener, self).__init__(lambda x, t: 0, D)

    @classmethod
    def potential(cls, X, t):
        """
        Useless (and potentially source of errors) to call the general potential routine
        since it is trivially zero here
        """
        return np.zeros_like(X)

class OrnsteinUhlenbeck(StochModel):
    """ The Ornstein-Uhlenbeck model """
    def __init__(self, mu, theta, D):
        super(OrnsteinUhlenbeck, self).__init__(lambda x, t: theta*(mu-x), D)

    def potential(self, X, t):
        """
        Return the potential from which the force derives
        """
        y = self.F(X, t)
        return y.dot(y)/2
