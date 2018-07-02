"""
Generic class for stochastic processes in arbitrary dimensions
"""
import numpy as np

class StochModel(object):
    """
    Generic class for stochastic processes in arbitrary dimensions
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
        """ Integrate a trajectory with given initial condition (t0,x0) """
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
        for _ in xrange(nsteps):
            t = t + dt
            x = x + self.increment(x, t, dt=dt)
            yield t, obs(x)

    def sample_mean(self, x0, t0, nsteps, nsamples, **kwargs):
        """
        Compute the sample mean of a time dependent observable
        """
        gens = [self.generator(x0, t0, nsteps, **kwargs) for _ in xrange(nsamples)]
        while True:
            time, obs = zip(*[next(gen) for gen in gens])
            yield np.average(time, axis=0), np.average(obs, axis=0)

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
