#!/usr/bin/env python
"""
Interactive presentation of the AMS algorithm for pedagogical purposes
"""
import numpy as np
import matplotlib.pyplot as plt
import stochrare as sr

class InteractiveAMS:
    # pylint: disable=too-few-public-methods
    """
    The object for interative interation of the AMS algorithm.

    It is initialized with a TAMS object, an ensemble size and a matplotlib.axes.Axes object
    for displaying the trajectories.

    The figure first displays the initial ensemble, then pressing the right arrow key iterates
    the algorithm, distinguishing three steps at each iteration of the algorithm:
    - selection phase: highlight the trajectory with the smallest level
    - mutation phase: highlight the resampled trajectory. The killed trajectory is not displayed.
    - iteration phase: prepare for next iteration by removing all colors

    Note: this object does not call any function from the matplotlib.pyplot API, it only relies on
          the object oriented interface through the matplotlib.axes.Axes object
          It may be included in the stochpy.ams module without the need to import matplotlib.

          Also, in the future, it might be interesting to be able to rewind the algorithm
          by pressing the left arrow key. This would require keeping in memory all the trajectories
          generated over the iterations of the algorithms.
    """
    def __init__(self, tams, ntraj, ax):
        self.tams = tams
        self.tams.initialize_ensemble(ntraj, dt=0.01)
        self.ax = ax
        self.lines = []
        self.maxdots = []
        for t, x in self.tams._ensemble:
            line, = self.ax.plot(t, x, color='grey')
            self.lines += [line]
            self.maxdots += [self.ax.scatter((t[np.argmax(x)], ),
                                             (x[np.argmax(x)], ),
                                             color='red')]
        _ = self.ax.figure.canvas.mpl_connect('key_press_event', self)
        self.step = 'selection'
        self.killed = np.array([])
        self.survivors = np.array([])

    def __call__(self, event):
        if event.key == 'right':
            if self.step == 'selection':
                self.killed, self.survivors = self.tams.selectionstep(self.tams._levels)
                for kill_ind in self.killed:
                    self.lines[kill_ind].set_color('C0')
                    self.lines[kill_ind].set_zorder(np.max([line.zorder for line in self.lines])+1)
                    #self.ax.axhline(y=np.max(self.tams._ensemble[kill_ind][1]), ls='dashed')
                self.step = 'mutation'
            elif self.step == 'mutation':
                self.tams.mutationstep(self.killed, self.survivors, dt=0.01)
                for kill_ind in self.killed:
                    t, x = self.tams._ensemble[kill_ind]
                    self.lines[kill_ind].set_data(t, x)
                    self.lines[kill_ind].set_color('C1')
                    self.ax.collections.remove(self.maxdots[kill_ind])
                    self.maxdots[kill_ind] = self.ax.scatter((t[np.argmax(x)], ),
                                                             (x[np.argmax(x)], ),
                                                             color='red')
                self.step = 'iter'
            elif self.step == 'iter':
                for kill_ind in self.killed:
                    self.lines[kill_ind].set_color('grey')
                self.step = 'selection'
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.ax.figure.canvas.draw()

def main():
    """
    Create the figure and the associated interactive AMS object.
    """
    plt.figure(figsize=(9, 5))
    ax = plt.axes()
    ax.grid(True)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    oup = sr.dynamics.diffusion1d.OrnsteinUhlenbeck1D(0, 1, 0.5)
    _ = InteractiveAMS(sr.rare.ams.TAMS(oup, (lambda t, x: x), 5.), 3, ax)
    plt.show()

if __name__ == '__main__':
    main()
