# BH: N variation
"""The below code runs the BH code for different $N$ (number of particles).
$\theta$ is fixed at $0.5$. The timing and error as a function of $N$ will
be examined.
"""
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.bh_n_variation
   Note: -m tells Python to load bh_n_variation.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)

   
   
""")
import numpy as np
import matplotlib.pyplot as plt
import time

from ..classes import Grid
from ..functions import bh_create_tree, bh_calc_phi, grid_direct_sum

N_range = np.logspace(2,3.5,5).astype('i') # [100, 3162]; takes around 2m30s to run
# N_range = np.logspace(2,3,4).astype('i') # [100, 1000]; takes < 1m to run
depths = []
max_errs = []

# time keeper for different components (keys)
keys = ['create_tree', 'bh_calc', 'direct_sum']
times = {key:[] for key in keys}

# loop BH over different N
for n in N_range:
    # initialise fixed parameters
    theta = 0.5

    print()
    print()
    print()
    print(f"------ n = {n}, theta = {theta} ------")
    very_start = time.time()
    print("Timer elapsed reset.")

    # initialise particles
    np.random.seed(4)
    grid = Grid(size=1024)
    all_coords = [i for i in range(n)]
    all_q = np.random.random(len(all_coords))*10+50
    all_particles = grid.create_particles(len(all_coords), all_coords=None, all_q=all_q)
    # print('Grid and particles initialised.')
    # print('Time elapsed:', time.time() - very_start)

    # BH tree construction
    tic = time.perf_counter()
    bh_create_tree(grid, grid.particles)
    toc = time.perf_counter()
    times[keys[0]].append(toc-tic)
    # print('Time elapsed:', time.time() - very_start)

    # BH tree depth determination (optional)
    # depth = find_bh_tree_depth(grid.rootbox, grid.size)
    # depths.append(depth)
    # print("BH tree depth:", depth)
    # print('Time elapsed:', time.time() - very_start)

    # BH calculation
    tic = time.perf_counter()
    bh_calc_phi(grid, theta)
    toc = time.perf_counter()
    times[keys[1]].append(toc-tic)
    print('BH calculated.')
    print('Time elapsed:', time.time() - very_start)

    # BH calculation output stored in "bh"
    bh = np.array(grid.get_all_phi())
    # print('Time elapsed:', time.time() - very_start)

    # direct calculation
    tic = time.perf_counter()
    grid_direct_sum(grid)
    toc = time.perf_counter()
    times[keys[2]].append(toc-tic)
    print('Directly calculated.')
    print('Time elapsed:', time.time() - very_start)

    # direct calculation output stored in "exact"
    exact = np.array(grid.get_all_phi())
    
    # (absolute) fractional error calculation
    bh_errs = (bh-exact)/exact
    data = abs(bh_errs)
    max_errs.append(max(data))
    
# BH plot: t vs N
if input("Generate t vs N plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,7))
    xlabel = 'N_range'
    x = eval(xlabel)

    # selecting components (keys) to plot
    excluded_keys = ()
    step_keys = [key for key in keys if key not in excluded_keys]

    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        X = np.log10(x[x!=0])
        Y = np.log10(y[x!=0])
        a, b = np.polyfit(X, Y, 1)
        ax.plot(X, Y, label = f'{keys[i]}, experimental')
        ax.plot(X, a*X+b, label=f'{keys[i]}, fit  $t \propto N^{{{a:.2f}}}$', linestyle='--')

    title = f'BH log$_{{{10}}}$ $t$ vs log$_{{{10}}}$ ${xlabel[0]}$; $θ$ = {theta}'
    ax.set_title(title)
    ax.set_xlabel('log$_{10}$ $N$')
    ax.set_ylabel('log$_{10}$ $t$')
    plt.legend(loc='upper left')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()


# BH plot: max_err vs N
if input("Generate max_err vs N plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,7))
    xlabel = 'N_range'
    x = eval(xlabel)
    y = np.array(max_errs)
    title = f'BH max_err vs ${xlabel[0]}$; $θ$ = {theta}'
    ax.set_title(title)
    ax.plot(x, y)
    ax.set_xlabel('$N$')
    ax.set_ylabel('max_err')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)