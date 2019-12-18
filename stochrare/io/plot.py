"""
Plotting routines
=================

.. currentmodule:: stochrare.io.plot

This module contains several functions for making quick plots.

.. autofunction:: trajectory_plot1d

.. autofunction:: pdf_plot1d

.. autofunction:: returntime_plot
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def trajectory_plot1d(*args, **kwargs):
    """
    Plot 1D  trajectories.

    Parameters
    ----------
    *args : variable length argument list
        trajs: tuple (t, x) or (t, x, kwargs_dict)

    Keyword Arguments
    -----------------
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
    labels = kwargs.pop('labels', [])
    if 'fig' in kwargs:
        fig = kwargs.pop('fig')
    else:
        fig = plt.figure()
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.axes(**kwargs)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x(t)$')

    lines = []
    for traj in args:
        lines += ax.plot(traj[0], traj[1], **(traj[2] if len(traj) > 2 else {}))

    if labels != []:
        ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return fig, ax


def pdf_plot1d(*args, legend=True, **kwargs):
    """
    Plot 1D PDFs.

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
    legend : bool
        Add legend (default True).
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
        ax.grid()
        ax.set_xlabel('$x$')
        ax.set_ylabel(kwargs.get('ylabel', '$P(x,t)$'))
    lines = []
    for pdf in args:
        line, = ax.plot(pdf[0], pdf[1], **(pdf[2] if len(pdf) > 2 else {}))
        lines += [line]
    if V is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel('$V(x,t)$')
        ax2.plot(X, V, linestyle='dashed')
    if legend:
        #ax.legend(**(kwargs.get('legend_args', {})))
        ax.legend()
    return fig, ax, lines

def ensemble_plot1d_box(*args, **kwargs):
    """
    Plot an ensemble of 1D trajectories in a 3D box.

    Parameters
    ----------
    *args : variable length argument list
        trajs: tuple (t, x) or (t, x, kwargs_dict)

    Keyword Arguments
    -----------------
    fig : matplotlig.figure.Figure
        Figure object to use for the plot. Create one if not provided.
    ax : matplotlig.axes.Axes
        Axes object to use for the plot. Create one if not provided.

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure.
    """
    plt.style.use('dark_background')
    if 'fig' in kwargs:
        fig = kwargs.pop('fig')
    else:
        fig = plt.figure()
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$t$', color='white')
        ax.set_zlabel(r'$x$', color='white')
        ax.grid(False)
        ax.set_yticks([])
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
    default_opts = {'zdir': 'y', 'color': 'steelblue', 'alpha': 0.6}
    for z, traj in enumerate(args):
        opts = default_opts.copy()
        if len(traj) > 2:
            opts.update(traj[2])
        ax.plot(traj[0], traj[1], z, **opts)
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
