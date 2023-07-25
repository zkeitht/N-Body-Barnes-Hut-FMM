# BH: theta variation
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.bh_theta_variation
   Note: -m tells Python to load bh_theta_variation.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)

   
   
""")
import numpy as np
import matplotlib.pyplot as plt
import time

from ..classes import Grid
from ..helperfunctions import bh_create_tree, bh_calc_phi, grid_direct_sum

theta_range = np.arange(0, 19, 0.5)
depths = []
max_errs = []
all_data = []

# time keeper for different components (keys)
keys = ['create_tree', 'bh_calc', 'direct_sum']
times = {key:[] for key in keys}

# loop BH over different theta
for theta in theta_range:
    # initialise fixed parameters
    n = 100

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

    # (absolute) relative error calculation
    bh_errs = (bh-exact)/exact
    data = abs(bh_errs)
    max_errs.append(max(data))
    all_data.append(data)

# BH plot: t vs theta
if input("Generate t vs theta plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,7))
    xlabel = 'theta_range'
    x = eval(xlabel)

    # selecting components (keys) to plot
    excluded_keys = ()
    step_keys = [key for key in keys if key not in excluded_keys]
    
    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        ax.plot(x, np.log10(y), label = f'{keys[i]}, experimental')

    title = f'BH log$_{{{10}}}$ $t$ vs $θ$'
    ax.set_title(title)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('log$_{10}$ $t$')
    plt.legend(loc='upper left')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()


# BH plot: error distribution with varying theta
if input("Generate error distribution with varying theta plot? (y/n) ") == 'y':
    r = 5
    s = 1
    for data, theta in zip(all_data[:r:s], theta_range[:r:s]):
        # if zeros exist:
        if any(data==0):
            tot = len(data)
            data = data[data!=0]
            tot_no0 = len(data)

        # create logspace bins
        minexpnt = np.floor(np.log10(min(data)))
        maxexpnt = np.ceil(np.log10(max(data)))
        bins = np.logspace(minexpnt, maxexpnt, 10)

        # create histogram with logscale bins
        plt.hist(data, bins=bins, log=True, label=f'θ={theta}')

    title = f'BH error distribution with varying $θ$'
    plt.title(title)
    plt.xlabel('error')
    plt.ylabel('frequency')
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend()
    fig = plt.gcf()
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# BH plot: max_err vs theta
if input("Generate max_err vs theta plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,7))
    xlabel = 'theta_range'
    x = eval(xlabel)
    y = np.array(max_errs)
    title = f'BH max_err vs $θ$'
    ax.set_title(title)
    ax.plot(x, y)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('max_err')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# BH plot: show particles and com spatially on grid (optional)
if input("Show particles and com on grid? (y/n) ") == 'y':
    grid.draw()
    grid.ax.scatter(*grid.rootbox.com_R, color='k', label='COM')
    grid.ax.legend(loc='lower right')
    grid.draw_squares(4)
    fig = grid.fig
    dummy = plt.figure(figsize=(6,6))
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    title = 'BH com'
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# BH plot: spatial distribution of error
if input("Show spatial distribution of error? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize=(6,5))
    x, y = zip(*grid.get_all_coords())
    data = abs(bh_errs)
    sc = plt.scatter(x, y, c=data, vmin = min(data), vmax=max(data), cmap = 'cividis_r')
    title = f'BH spatial distribution of error, $N$ = {n}, $θ$ =  {theta_range[-1]}'
    plt.title(title)
    plt.colorbar(sc, label= '(abs.) fractional error')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)