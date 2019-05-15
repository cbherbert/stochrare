"""
Sample stochastic processes
===========================

.. currentmodule:: stochpy.dynamics

This is the core module of StochPy.
It contains submodules for simulating trajectories corresponding to different stochastic dynamics.

For now, only diffusion processes are available.

.. autosummary::
   :toctree:

   diffusion
   diffusion1d

"""
from . import diffusion, diffusion1d
