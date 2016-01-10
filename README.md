# Tracking global optima in dynamic environments with efficient global optimization

Morales-Enciso, Sergio and Branke, Juergen, Tracking global optima in dynamic environments
with global optimization. In European Journal of Operational Research, Volume 242, Issue 3, 
1 May 2015, Pages 744–755. [[1]](http://www.sciencedirect.com/science/article/pii/S0377221714009515)

# Abstract
Many practical optimization problems are dynamically changing, and require a tracking of the global optimum over time. However, tracking usually has to be quick, which excludes re-optimization from scratch every time the problem changes. Instead, it is important to make good use of the history of the search even after the environment has changed. In this paper, we consider Efficient Global Optimization (EGO), a global search algorithm that is known to work well for expensive black box optimization problems where only few function evaluations are possible. It uses metamodels of the objective function for deciding where to sample next. We propose and compare four methods of incorporating old and recent information in the metamodels of EGO in order to accelerate the search for the global optima of a noise-free objective function stochastically changing over time. As we demonstrate, exploiting old information as much as possible significantly improves the tracking behavior of the algorithm.

# Keywords
* Heuristics
* Dynamic global optimization
* Efficient global optimization 
* Gaussian processes
* Response surfaces

# Highlights
* Metamodel-based optimization for expensive dynamic black box functions.
* Novel adaptation of efficient global optimization to dynamic environments.
* Four approaches to decrease reliance on old information empirically compared.
* Comparisons with naive approaches of re-optimization or ignoring change show significant improvement.

# License
This is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License.
* movpeaks.c and movpeaks.h - Copyright (C) 1999 Juergen Branke. 
* All other files - Copyright (C) 2014 Sergio Morales.


# Citation
If using this code, please cite as: 

Morales-Enciso, S. and Branke, J., Tracking global optima in dynamic environments with global optimization. In European Journal of Operational Research, Volume 242, Issue 3,1 May 2015, Pages 744–755.
