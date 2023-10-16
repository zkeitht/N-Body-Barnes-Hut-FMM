# N-Body-Barnes-Hut-FMM

N-body simulation using Hierarchical Methods: the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) (BH) method and the [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method) (FMM).

The aim of the project is to reduce the time complexity of pairwise interactions in N-body systems to below $O(N^2)$. Both algorithms are implemented in python. The time complexity and error as a function of the number of particles $N$ and other parameters in both algorithms are explored.

Important packages: 

    classes/     # to create objects: grid, particle, box

    functions/   # the BH and FMM algorithms

    simulations/ # uses the functions to operate on the objects under varying conditions

See [*Hierarchical N-body simulation.ipynb*](/Hierarchical%20N-body%20simulation.ipynb) for a report on an exploration of the hierarchical methods and examples of how to use the code. A summary of the repository layout is also given in the introductory subsection of the report.

The Barnes-Hut Method
![The Barnes Hut method visualised.](/Figures/BH%20eval%20step%202.png)

The FMM
![The 6 Steps of FMM.](/Figures/FMMOverview.jpg)
