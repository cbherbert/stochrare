"""
Tools for time series analysis.
These tools should apply in particular to trajectories generated by stochastic processes
using the dynamics subpackage.
"""
import numpy as np

def running_mean(x, N):
    """
    Return the running mean of a time series
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def transitionrate(x, threshold, **kwargs):
    """
    Count the number of times a given trajectory goes across a given threshold.
    A typical use case is to study transitions from one attractor to the other.

    x: the time series (a numpy array)
    threshold: the threshold (e.g. separating the two attractors); a float
    avg (optional): averaging window for smoothing timeseries before computing transition rate

    Without smoothing (avg=1), the result should coincide with the number of items in the generator
    levelscrossing(x,0) when starting with the right transition,
    or that number +1 if we use the wrong isign in levelscrossing.
    """
    window = kwargs.get('avg', 1)
    y = running_mean(x, window) if window > 1 else x
    return float(((y[1:]-threshold)*(y[:-1]-threshold) < 0).sum())/len(y)

def levelscrossing(x, threshold, **kwargs):
    """
    Maps the stochastic process x(t) onto a stochastic process {t_i}
    where the 't_i's correspond to crossing levels +- c
    """
    # By default we start by detecting the transition below the -c threshold
    sign = kwargs.get('isign', 1)
    if sign == 0:
        sign = 1
    if not abs(sign) == 1:
        sign /= abs(sign)
    for i in xrange(len(x)-1):
        if (threshold+sign*x[i]) > 0 and (threshold+sign*x[i+1]) < 0:
            sign *= -1
            yield i

def residencetimes(x, threshold):
    transtimes = np.array([t for t in levelscrossing(x, threshold)])
    return transtimes[1:]-transtimes[:-1]