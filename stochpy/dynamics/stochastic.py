"""
Generic class for stochastic processes in arbitrary dimensions
"""
import numpy as np

class DiffusionProcess:
    """
    Generic class for diffusion processes in arbitrary dimensions.

    It corresponds to the family of SDEs dx_t = F(x_t, t)dt + sigma(x_t, t)dW_t,
    where F is a time-dependent N-dimensional vector field
    and W the M-dimensional Wiener process.
    The diffusion matrix sigma has size NxM.
    """

    default_dt = 0.1

    def __init__(self, vecfield, sigma):
        """
        vecfield: vector field
        sigma: diffusion coefficient (noise)

        vecfield and sigma are functions of two variables (x,t)
        """
        self.drift = vecfield
        self.diffusion = sigma

    def increment(self, x, t, **kwargs):
        """
        Return F(x_t, t)dt + sigma(x_t, t)dW_t
        x is a n-dimensional vector (in R^n)
        t is the time (a real number)
        """
        dt = kwargs.get('dt', self.default_dt)
        dim = len(x)
        return self.drift(x, t)*dt+np.sqrt(dt)*self.diffusion(x, t)@np.random.normal(0.0, 1.0, dim)

    def trajectory(self, x0, t0, **kwargs):
        """
        Integrate a trajectory with given initial condition (t0, x0)
        Optional arguments:
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
        for ensemble in zip(*[self.generator(x0, t0, nsteps, **kwargs) for _ in range(nsamples)]):
            time, obs = zip(*ensemble)
            yield np.average(time, axis=0), np.average(obs, axis=0)

class ConstantDiffusionProcess(DiffusionProcess):
    """
    Diffusion process with constant diffusion coefficient.
    """

    default_dt = 0.1

    def __init__(self, vecfield, Damp, dim):
        """
        vecfield: vector field, function of two variables (x,t)
        Damp: amplitude of the diffusion term (noise), scalar
        dim: dimension of the system

        In this class of stochastic processes, the diffusion matrix is proportional to identity.
        """
        DiffusionProcess.__init__(self, vecfield, (lambda x, t: np.sqrt(2*Damp)*np.eye(dim)))
        self.D0 = Damp
        self.dimension = dim

    def increment(self, x, t, **kwargs):
        """
        Return F(x_t, t)dt + sqrt(2*D0)dW_t
        x is a n-dimensional vector (in R^n)
        t is the time (a real number)
        """
        dt = kwargs.get('dt', self.default_dt)
        if len(x) != self.dimension:
            raise ValueError('Input vector does not have the right dimension.')
        return self.drift(x, t)*dt+np.sqrt(2*self.D0*dt)*np.random.normal(0, 1, self.dimension)


class Wiener(ConstantDiffusionProcess):
    """ The Wiener process """
    def __init__(self, dim, D=1):
        super(Wiener, self).__init__(lambda x, t: 0, D, dim)

    @classmethod
    def potential(cls, X, t):
        """
        Useless (and potentially source of errors) to call the general potential routine
        since it is trivially zero here
        """
        return np.zeros_like(X)

class OrnsteinUhlenbeck(ConstantDiffusionProcess):
    """ The Ornstein-Uhlenbeck model """
    def __init__(self, mu, theta, D, dim):
        super(OrnsteinUhlenbeck, self).__init__(lambda x, t: theta*(mu-x), D, dim)

    def potential(self, X, t):
        """
        Return the potential from which the force derives
        """
        y = self.drift(X, t)
        return y.dot(y)/2
