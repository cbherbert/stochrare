# stochpy

This Python package aims at providing tools to study stochastic processes:
- numerical integration of SDEs
- numerical solver for the Fokker-Planck equations
- first-passage time computation
- instanton computation
- rare event algorithms

Stochastic models arise in many scientific fields: physics, chemistry, biology, finance...
Although the package was initially developed with an out-of-equilibrium statistical physics point of view, it aims at providing tools to support research with stochastic processes in any of these fields.

## Scope

We have identified two main use cases:
- interactive study of low-dimensional stochastic processes. This may be useful to study a model of a particular phenomenon arising in a research context, to develop new methods and algorithms, or for pedagogical use.
- framework for designing streamlined workflows for reproducible numerical experiments. In this context, the package may be interfaced with more complex simulation codes (e.g. computational fluid dynamics, molecular dynamics,...), acting mostly as a wrapper providing flexibility while the heavy lifting is done by the underlying code.

Until now the code has been mostly developed and tested in the first context, but this should change soon...

The package may also contain code which could be of interest to study deterministic dynamical systems, although this is not the primary goal.

## Documentation

Documentation in the *doc* directory can be compiled in various formats.

In the *demo* directory, some Jupyter notebooks illustrate the basic features of the package. 
More should be added soon, in particular for metastability or loss of stability problem and noise-induced transitions.
These notebooks will include references to the relevant litterature.

