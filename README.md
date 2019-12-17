# stochrare

[![Documentation Status](https://readthedocs.org/projects/stochrare/badge/?version=latest)](https://stochrare.readthedocs.io/en/latest/?badge=latest)

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

## Citation

We should soon write a metapaper describing the project.
In the meantime, if you use the package for your research, we would appreciate if you could give credit by citing one of the research papers where the development of stochrare started:

	@article{Herbert2017,
	Author = {Herbert, Corentin and Bouchet, Freddy},
	Doi = {10.1007/BF01106788},
	Journal = {Phys. Rev. E},
	Pages = {030201(R)},
	Title = {{Predictability of escape for a stochastic saddle-node bifurcation: when rare events are typical}},
	Volume = {96},
	Year = {2017}}

for the core features, and

	@article{Lestang2018,
	Author = {Lestang, Thibault and Ragone, Francesco and Br{\'e}hier, Charles-Edouard and Herbert, Corentin and Bouchet, Freddy},
	Doi = {10.1088/1742-5468/aab856},
	Title = {{Computing return times or return periods with rare event algorithms}},
	Journal = {J. Stat. Mech.},
	Volume = {043213},
	Year = {2018}}
	
for the AMS algorithm and return time computations.
