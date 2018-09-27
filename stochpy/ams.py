"""
Module for the Adaptive Multilevel Splitting algorithm
"""
import numpy as np

class TAMS(object):
    """
    Implement the TAMS algorithm as described in
    Lestang, Ragone, Brehier, Herbert and Bouchet, J. Stat. Mech. 2018
    """
    def __init__(self, model, scorefun, duration):
        """
        - dynamics: stochpy.dynamics.StochModel object (or a subclass of it)
                    The dynamical model; so far we are restricted to SDEs of the form
                        dX_t = F(X_t, t) + sqrt(2D)dW_t
                    We only used the trajectory method of the dynamics object.
        - score: a scalar function with two arguments.
                 The score function Xi(t, x)
        - duration: float
                    The fixed duration for each trajectory
        """
        self.dynamics = model
        self.score = scorefun
        self.duration = duration

    def getcrossingtime(self, level, times, traj):
        """
        Return the time and position at which the trajectory reaches the specified level
        - level: float
        - times: numpy.ndarray
                 Sampling times for the trajectory.
        - traj: numpy.ndarray
                Position of the system at the sampling times.
        """
        for t, x in zip(times, traj):
            if self.score(t, x) > level:
                return t, x
        return np.nan, np.nan

    def getlevel(self, times, traj):
        """
        Return the maximum reached by the score function over the trajectory
        - times: numpy.ndarray
                 Sampling times for the trajectory.
        - traj: numpy.ndarray
                Position of the system at the sampling times.
        """
        return np.max([self.score(t, x) for t, x in zip(times, traj)])

    def resample(self, time, pos, told, xold, **kwargs):
        """
        Resample a trajectory after a given time
        """
        tnew, xnew = self.dynamics.trajectory(pos, time, T=self.duration-time, **kwargs)
        tnew = np.concatenate((told[told < time], tnew), axis=0)
        xnew = np.concatenate((xold[told < time], xnew), axis=0)
        return tnew, xnew

    def tams_run(self, ntraj, niter, **kwargs):
        """
        Generate trajectories
        - ntraj: the number of trajectories in the initial ensemble
        - niter: the number of iterations of the algorithm

        Optional arguments can be passed to the "trajectory" method of the dynamics object.
        """
        # For now we fix the initial conditions:
        x0 = 0
        t0 = 0
        # generate the initial ensemble:
        ensemble = [self.dynamics.trajectory(x0, t0, T=self.duration, **kwargs)
                    for _ in xrange(ntraj)]
        weight = 1
        for _ in xrange(niter):
            # compute the maximum of the score function over each trajectory:
            levels = np.array([self.getlevel(*traj) for traj in ensemble])
            # select the trajectory to be killed:
            kill_ind = levels.argmin()
            yield ensemble[kill_ind], weight
            # select the trajectory on which we clone the killed trajectory
            clone_ind = np.random.choice([i for i in xrange(ntraj) if i != kill_ind])
            # compute the time from which we resample
            t_resamp, x_resamp = self.getcrossingtime(levels[kill_ind], *ensemble[clone_ind])
            # resample the trajectory
            ensemble[kill_ind] = self.resample(t_resamp, x_resamp, *ensemble[clone_ind], **kwargs)
            # update the weight
            weight = weight*(1-1./ntraj)
        for traj in ensemble:
            yield traj, weight

    def tams_returntimes(self, ntraj, niter, **kwargs):
        """
        Here the observable is the score function itself (temporary only)
        """
        tamsgen = self.tams_run(ntraj, niter, **kwargs)
        blockmax = np.array([(self.getlevel(*traj), wght) for traj, wght in tamsgen])
        blockmax[:, 1] = blockmax[:, 1]/np.sum(blockmax[:, 1])
        blockmax = blockmax[blockmax[:, 0].argsort()[::-1]]
        return blockmax[:, 0], -self.duration/np.log(1-np.cumsum(blockmax[:, 1]))
