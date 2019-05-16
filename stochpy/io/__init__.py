"""
Input/Output
============

.. currentmodule:: stochpy.io

Stochpy generates two kinds of outputs:
- data (results of computations that we wish to store on disk for future use)
- plots (results fo computations that we wish to represent in a graphic manner)

Hence the :mod:`io` module is organized into two submodules, :mod:`stochpy.io.data`
and :mod:`stochpy.io.plot`.

.. autosummary::
   :toctree:

   data
   plot

"""
from . import data, plot
