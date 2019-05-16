"""
Plotting routines
=================

.. currentmodule:: stochpy.io.plot

This module contains several functions for making quick plots.

.. autofunction:: returntime_plot
"""
import matplotlib.pyplot as plt

def returntime_plot(*args):
    """
    Make return time plot: amplitude a as a function of the return time r(a)

    Parameters
    ----------
    *args: variable length argument list
           Pairs of the form (a, r(a))

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure.
    """
    fig = plt.figure()
    ax = plt.axes()
    for amp, ret in args:
        ax.plot(ret, amp)
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel(r'$r(a)$')
    ax.set_ylabel(r'$a$')
    return fig, ax
