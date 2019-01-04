"""
Module for rare event algorithms of the "Adaptive Multilevel Splitting" family.

There are two kinds of variants of the AMS algorithms.
On the one hand, there are "scientific" variants, corresponding to different formulations of the
algorithm (e.g. AMS vs TAMS).
On the other hand, there are "technical" variants, corresponding to different implementations: for
instance, keeping all the trajectories in memory or storing them on disk (necessary for applications
to complex systems)
"""
import numpy as np

class AMS(object):
    """
    Implement the original version of the 'Adaptive Multilevel Splitting Algorithm'

    TODO: add references

    The algorithm evolves an ensemble of trajectories in an interactive manner,
    using selection and mutation steps.
    The algorithm requires two sets A and B, and a 'reactive coordinate' or 'score function' xi,
    measuring the distance between the two.
    In fact, we require that xi vanishes over the boundary of A, and takes unit value over the
    boundary of B.

    * Initilization:
    The ensemble is initialized by running N trajectories until they reach set A or set B.

    * Selection
    Then at each iteration, the maximum value of the score function over each member of the ensemble
    is computed. The q trajectories with lowest score function are 'killed'.

    * Mutation
    For each trajectory killed, we pick a random trajectory among the 'survivors'.
    We clone that trajectory until it reaches the level of the killed trajectory for the first time,
    then we restart it from that point until it reaches set A or B.

    The algorithm is iterated until all trajectories reach set B.
    """
    def __init__(self, model, scorefun):
        """
        - model: stochpy.dynamics.StochModel object (or a subclass of it)
                 The dynamical model; so far we are restricted to SDEs of the form
                     dX_t = F(X_t, t) + sqrt(2D)dW_t
                 We only use the increment method of the dynamics object.
        - scorefun: a scalar function with two arguments.
                    The score function Xi(t, x)
        """
        self.dynamics = model
        self.score = scorefun
        self._ensemble = []
        self._levels = []
        self._weight = 0

    ###
    #   Tools used by the AMS algorithm to handle trajectories
    #   These could almost be private methods.
    ###

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
        tnew, xnew = self.simul_trajectory(pos, time, **kwargs)
        tnew = np.concatenate((told[told < time], tnew), axis=0)
        xnew = np.concatenate((xold[told < time], xnew), axis=0)
        return tnew, xnew

    def simul_trajectory(self, x0, t0, **kwargs):
        """
        Simulate a trajectory with initial conditions (t0, x0), until it reaches either set A
        (score <= 0) or set B (score >= 1).
        """
        t = [t0]
        x = [x0]
        dt = kwargs.get('dt', self.dynamics.default_dt)
        while 0 < self.score(t[-1], x[-1]) < 1:
            t += [t[-1] + dt]
            x += [x[-1] + self.dynamics.increment(x[-1], t[-1], dt=dt)]
        return np.array(t), np.array(x)

    ###
    #   selectionstep and mutationstep are the two building blocks for the AMS algorithm
    ###

    def initialize_ensemble(self, x0, t0, ntraj, **kwargs):
        """
        Generate the initial ensemble.
        """
        self._ensemble = [self.simul_trajectory(x0, t0, **kwargs) for _ in range(ntraj)]
        self._weight = 1
        # compute the maximum of the score function over each trajectory:
        self._levels = np.array([self.getlevel(*traj) for traj in self._ensemble])

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
        if npart > len(levels):
            raise RuntimeError("Cannot kill more particles than the ensemble size!")
        lev_tmp = np.unique(levels)
        kill_threshold = lev_tmp[min(npart, lev_tmp.size)-1]
        survivor_pool = np.flatnonzero(levels > kill_threshold)
        killed_pool = np.flatnonzero(levels <= kill_threshold)
        # If survivor_pool is empty, we kill no particle instead of killing them all
        if survivor_pool.size == 0:
            killed_pool, survivor_pool = survivor_pool, killed_pool
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


    ###
    #    Below are the methods which actually implement the whole algorithm.
    #    There are actually several of them, corresponding to small variants
    #    (trajectory enumeration order, stopping condition).
    ###

    def run_iter(self, ntraj, niter, **kwargs):
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
        for _ in range(niter):
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
        for _ in range(niter):
            killed_pool, survivor_pool = self.selectionstep(self._levels)
            self.mutationstep(killed_pool, survivor_pool, **kwargs)
            for kill_ind in killed_pool:
                yield self._ensemble[kill_ind], self._weight

    def run_level(self, ntraj, target_lev, **kwargs):
        """
        Generate trajectories
        - ntraj: the number of trajectories in the initial ensemble
        - target_lev: the target level

        The generator yields (trajectory, weight) pairs which allows to compute easily the
        probability associated to each sampled trajectory.
        This method yields first the killed trajectories as the algorithm is iterated,
        then the trajectories in the final ensemble.

        Optional arguments can be passed to the "trajectory" method of the dynamics object.
        """
        # For now we fix the initial conditions:
        self.initialize_ensemble(0, 0, ntraj, **kwargs)
        while np.min(self._levels) < target_lev:
            killed_pool, survivor_pool = self.selectionstep(self._levels)
            for kill_ind in killed_pool:
                yield self._ensemble[kill_ind], self._weight
            self.mutationstep(killed_pool, survivor_pool, **kwargs)
        for traj in self._ensemble:
            yield traj, self._weight



class TAMS(AMS):
    """
    Implement the TAMS algorithm as described in
    Lestang, Ragone, Brehier, Herbert and Bouchet, J. Stat. Mech. 2018

    This implementation keeps all the information in memory: this should not be suitable for complex
    dynamics.
    Similarly, the algorithm is not parallelized, even if the dynamics itself may be.
    """
    def __init__(self, model, scorefun, duration):
        """
        - dynamics: stochpy.dynamics.StochModel object (or a subclass of it)
                    The dynamical model; so far we are restricted to SDEs of the form
                        dX_t = F(X_t, t) + sqrt(2D)dW_t
                    We only use the trajectory method of the dynamics object.
        - score: a scalar function with two arguments.
                 The score function Xi(t, x)
        - duration: float
                    The fixed duration for each trajectory
        """
        AMS.__init__(self, model, scorefun)
        self.duration = duration

    def resample(self, time, pos, told, xold, **kwargs):
        """
        Resample a trajectory after a given time
        """
        kwargs['T'] = told[0] + self.duration-time
        return AMS.resample(self, time, pos, told, xold, **kwargs)

    def simul_trajectory(self, x0, t0, **kwargs):
        """
        Simulate a trajectory with initial conditions (t0, x0) for a fixed duration.
        """
        if 'T' not in kwargs:
            kwargs['T'] = self.duration
        return self.dynamics.trajectory(x0, t0, **kwargs)


    ###
    #   Methods to estimate properties of observables based on the trajectories
    #   sampled by the algorithm
    ###

    def average(self, ntraj, niter, observable, **kwargs):
        """
        Estimate the average of an observable using AMS sampling
        - ntraj: number of initial trajectories in the ensemble
        - niter: number of iterations of the AMS algorithm
        - observable: a function of the form O(t, x), where t and x are numpy arrays.
                      It should itself return a numpy array.
                      For instance, it could be a time-independent function of the type
                         (lambda t, x: x**2)
                      or a functional of the trajectory such as
                         (lambda t, x: np.array([np.max(x**2)])
                      Note that in the latter case it is crucial to convert the scalar to an array.
        - method (optional): the method used to sample the trajectories with the AMS algorithm
                             It can be one of TAMS().run_iter or TAMS().run_resamp.
        - condition (optional): predicate for conditional averaging.
                                It should be of the form pred((t,x)) in (True, False).
        """
        method = kwargs.pop('method', self.run_iter)
        pred = kwargs.pop('condition', (lambda X: True))
        tamsgen = method(ntraj, niter, **kwargs)
        obs = np.array([0])
        norm = 0
        for traj, wght in tamsgen:
            if pred(traj):
                obstraj = observable(*traj)
                # The length of trajectories may differ because of rounding errors
                # Hence, we truncate to the smallest size of the sample trajectories
                size = min(obs.size, obstraj.size) if obs.size > 1 else obstraj.size
                obs = obs[:size] + wght*obstraj[:size]
                norm = norm + wght
        return obs/norm

    def returntimes(self, ntraj, niter, **kwargs):
        """
        Estimate the return time of an observable using AMS sampling
        """
        method = kwargs.get('method', self.run_iter)
        obs = kwargs.get('observable', self.score)
        tamsgen = method(ntraj, niter, **kwargs)
        blockmax = np.array([(np.max([obs(t, x) for t, x in zip(*traj)]), wght)
                             for traj, wght in tamsgen])
        blockmax[:, 1] = blockmax[:, 1]/np.sum(blockmax[:, 1])
        blockmax = blockmax[blockmax[:, 0].argsort()[::-1]]
        return blockmax[:, 0], -self.duration/np.log(1-np.cumsum(blockmax[:, 1]))
