"""
Input/Output
============

.. currentmodule:: stochrare.io

Stochpy generates two kinds of outputs:
- data (results of computations that we wish to store on disk for future use)
- plots (results fo computations that we wish to represent in a graphic manner)

Hence the :mod:`io` module is organized into two submodules, :mod:`stochrare.io.data`
and :mod:`stochrare.io.plot`.

.. autosummary::
   :toctree:

   data
   plot

"""
from . import data, plot
