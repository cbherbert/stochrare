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
        self._ensemble = []
        self._levels = []
        self._weight = 0

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

    @staticmethod
    def selectionstep(levels, npart=1):
        """
        Selection step of the AMS algorithm:
        Generator returning the trajectories in the ensemble which performed worse,
        i.e. the trajectories for which the maximum value of the score function
        over the trajectory is minimum.
        - levels: list of levels reached by the ensemble members
        - npart: the number of levels to select
                 npart=1 corresponds to the last particle method
                 Note that one level can correspond to several trajectories in the ensemble.
        """
        kill_threshold = np.unique(levels)[npart-1]
        # What do we do if npart > len(np.unique(levels)) ? i.e. survivor_pool == []
        survivor_pool = np.flatnonzero(levels > kill_threshold)
        killed_pool = np.flatnonzero(levels <= kill_threshold)
        return killed_pool, survivor_pool

    def mutationstep(self, killed_pool, survivor_pool, **kwargs):
        """
        Mutation step for the AMS algorithm

        This is the only method which modifies the state of the ensemble (the TAMS object).
        """
        for kill_ind in killed_pool:
            # compute the time from which we resample
            clone_ind = np.random.choice(survivor_pool)
            tcross, xcross = self.getcrossingtime(self._levels[kill_ind],
                                                  *self._ensemble[clone_ind])
            # resample the trajectory
            self._ensemble[kill_ind] = self.resample(tcross, xcross, *self._ensemble[clone_ind],
                                                     **kwargs)
            self._levels[kill_ind] = self.getlevel(*self._ensemble[kill_ind])
        # update the weight
        self._weight = self._weight*(1-float(len(killed_pool))/len(self._ensemble))

    def initialize_ensemble(self, x0, t0, ntraj, **kwargs):
        """
        Generate the initial ensemble.
        """
        self._ensemble = [self.dynamics.trajectory(x0, t0, T=self.duration, **kwargs)
                          for _ in xrange(ntraj)]
        self._weight = 1
        # compute the maximum of the score function over each trajectory:
        self._levels = np.array([self.getlevel(*traj) for traj in self._ensemble])


    def tams_run(self, ntraj, niter, **kwargs):
        """
        Generate trajectories
        - ntraj: the number of trajectories in the initial ensemble
        - niter: the number of iterations of the algorithm

        The generator yields (trajectory, weight) pairs which allows to compute easily the
        probability associated to each sampled trajectory.
        This method yields first the killed trajectories as the algorithm is iterated,
        then the trajectories in the final ensemble.

        Optional arguments can be passed to the "trajectory" method of the dynamics object.
        """
        # For now we fix the initial conditions:
        self.initialize_ensemble(0, 0, ntraj, **kwargs)
        for _ in xrange(niter):
            killed_pool, survivor_pool = self.selectionstep(self._levels)
            for kill_ind in killed_pool:
                yield self._ensemble[kill_ind], self._weight
            self.mutationstep(killed_pool, survivor_pool, **kwargs)
        for traj in self._ensemble:
            yield traj, self._weight

    def run_resamp(self, ntraj, niter, **kwargs):
        """
        Generate trajectories
        - ntraj: the number of trajectories in the initial ensemble
        - niter: the number of iterations of the algorithm

        The generator yields (trajectory, weight) pairs which allows to compute easily the
        probability associated to each sampled trajectory.
        This method yields first the trajectories in the initial ensemble, then the resampled
        trajectories as the algorithm is iterated.

        Optional arguments can be passed to the "trajectory" method of the dynamics object.
        """
        # For now we fix the initial conditions:
        self.initialize_ensemble(0, 0, ntraj, **kwargs)
        for traj in self._ensemble:
            yield traj, self._weight
        for _ in xrange(niter):
            killed_pool, survivor_pool = self.selectionstep(self._levels)
            self.mutationstep(killed_pool, survivor_pool, **kwargs)
            for kill_ind in killed_pool:
                yield self._ensemble[kill_ind], self._weight

    def tams_returntimes(self, ntraj, niter, **kwargs):
        """
        Here the observable is the score function itself (temporary only)
        """
        method = kwargs.get('method', self.tams_run)
        tamsgen = method(ntraj, niter, **kwargs)
        blockmax = np.array([(self.getlevel(*traj), wght) for traj, wght in tamsgen])
        blockmax[:, 1] = blockmax[:, 1]/np.sum(blockmax[:, 1])
        blockmax = blockmax[blockmax[:, 0].argsort()[::-1]]
        return blockmax[:, 0], -self.duration/np.log(1-np.cumsum(blockmax[:, 1]))
