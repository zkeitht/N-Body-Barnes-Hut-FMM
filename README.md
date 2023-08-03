# N-Body-Barnes-Hut-FMM

N-body simulation using Hierarchical Methods: the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) (BH) method and the [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method) (FMM).

The aim of the project is to reduce the complexity of pairwise interactions in N-body systems to below $O(N^2)$. Both algorithms are implemented in python. The time complexity and error as a function of the number of particles $N$ and other parameters in both algorithms are explored.

Important packages: 

    classes/     # to create objects: grid, particle, box

    functions/   # the BH and FMM algorithms

    simulations/ # uses the functions to operate on the objects under varying conditions

See *Hierarchical N-body simulation.ipynb* for a report on an exploration of the hierarchical methods and examples of how to use the code. A summary of the repository layout is also given in the introductory subsection of the report.
