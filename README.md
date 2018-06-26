# stochpy

This Python package aims at providing tools to study stochastic processes:
- numerical integration of SDEs
- numerical solver for the Fokker-Planck equations
- first-passage time computation
- instanton computation
- rare event algorithms (coming soon)

For now the code is restricted to 1D processes, which makes it essentially useful for tests and pedagogical purposes, but this should change soon.
Sub-packages shall be added for specific models of particular interest, such as a double well potential or a saddle-node bifurcation.

In the *demo* directory, I will put Jupyter notebooks illustrating the features of the package, in particular metastability or loss of stability problem and noise-induced transitions.
These notebooks will include references to the relevant litterature.

The package may also contain code which could be of interest to study deterministic dynamical systems, although this is not the primary goal.
