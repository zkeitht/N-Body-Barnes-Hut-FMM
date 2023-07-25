# FMM: N variation and steps
"""
This shows the step variation of number of levels as a function of N.
No detailed calculation will be made as only tree construction is required.
"""
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.fmm_n_variation_steps
   Note: -m tells Python to load fmm_n_variation_steps.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)
--------------------
In the event of ModuleNotFoundError:
1. CHANGE "classes" to "..classes" in helperfunctions/fmm_functions.py
2. Perform step 1 of above "In the event of ImportError".
3. After running, CHANGE  "..classes" back to "classes" for the module to work in Jupyter notebook.

   
   
""")
import numpy as np
import matplotlib.pyplot as plt
import time

from ..classes import GridComplex
from ..helperfunctions import construct_tree_fmm

N_range = np.logspace(2,5,10).astype('i') # [100, 100000]; takes within 1m
ptcmax_range = []
lvlss = []

# loop FMM over different N
for n in N_range:
    # initialise fixed parameters
    p = 10
    ptcmax = 10
    ptcmax_range.append(ptcmax)
    lvls = int(np.ceil(np.emath.logn(4, n/ptcmax))) #+1

    print()
    print()
    print()
    very_start = time.time()
    print(f"------ p = {p}, n = {n}, lvls = {lvls}, ptcmax = {ptcmax} ------")

    # initialise particles
    np.random.seed(4)
    gridcomplex = GridComplex(size=128)
    all_coords = [i for i in range(n)]
    all_q = np.random.random(len(all_coords))*10+50
    all_particles = gridcomplex.create_particles(len(all_coords), all_coords=None, all_q=all_q)
    # print('Grid and particles initialised.')
    # print('Time elapsed:', time.time() - very_start)

    # FMM tree construction
    tic = time.perf_counter()
    tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, ptcmax, p)
    toc = time.perf_counter()
    if crowded:
        while crowded:
            lvls+=1
            tic = time.perf_counter()
            tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, ptcmax, p)
            toc = time.perf_counter()
        print(f'lvls readjusted to {lvls}.')
    lvlss.append(lvls)
    print(f'Tree constructed.')
    print('Time elapsed:', time.time() - very_start)

print()

# FMM plot: lvls vs N
if input("Generate lvls vs N plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,6))
    xlabel = 'N_range'
    x = eval(xlabel)
    y = lvlss
    title = f'FMM Number of levels vs log $N$; $ptcmax$ = {ptcmax}'
    ax.set_title(title)
    ax.plot(np.log10(x), y)
    ax.set_xlabel(f'log$_{{{10}}}$ ${xlabel[0]}$')
    ax.set_ylabel('lvls')
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()