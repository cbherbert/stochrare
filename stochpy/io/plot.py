"""
Plotting routines
=================

.. currentmodule:: stochpy.io.plot

This module contains several functions for making quick plots.

.. autofunction:: pdf_plot1d

.. autofunction:: returntime_plot
"""
import matplotlib.pyplot as plt

def pdf_plot1d(*args, **kwargs):
    """
    Plot PDFs.

    Parameters
    ----------
    *args : variable length argument list
        PDFs: tuple (X, P) or (X, P, kwargs_dict)

    Keyword Arguments
    -----------------
    potential : ndarray 2-tuple
        X, V where V is the value of the potential at the sample points X.
        Default (None, None).
    fig : matplotlig.figure.Figure
        Figure object to use for the plot. Create one if not provided.
    ax : matplotlig.axes.Axes
        Axes object to use for the plot. Create one if not provided.
    **kwargs :
        Other keyword arguments forwarded to matplotlib.pyplot.axes.

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure.
    """
    X, V = kwargs.pop('potential', (None, None))
    if 'fig' in kwargs:
        fig = kwargs.pop('fig')
    else:
        fig = plt.figure()
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.axes(**kwargs)
    for pdf in args:
        ax.plot(pdf[0], pdf[1], **(pdf[2] if len(pdf) > 2 else {}))
    if V is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel('$V(x,t)$')
        ax2.plot(X, V, linestyle='dashed')
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel(kwargs.get('ylabel', '$P(x,t)$'))
    #ax.legend(**(kwargs.get('legend_args', {})))
    ax.legend()
    return fig, ax

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
