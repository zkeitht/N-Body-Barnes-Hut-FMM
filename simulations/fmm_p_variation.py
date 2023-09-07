# FMM: p variation
"""The below code runs the FMM code for different $p$ (order of expansion).
$N$ and $m$ are fixed. The timing and error as a function of $p$ will 
be examined.
"""
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.fmm_p_variation
   Note: -m tells Python to load fmm_p_variation.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)
--------------------
In the event of ModuleNotFoundError:
1. CHANGE "classes" to "..classes" in functions/fmm_functions.py
2. Perform step 1 of above "In the event of ImportError".
3. After running, CHANGE  "..classes" back to "classes" for the module to work in Jupyter notebook.

   
   
""")
import numpy as np
import matplotlib.pyplot as plt
import time

from ..classes import GridComplex
from ..functions import lvls_fmm, construct_tree_fmm, fmm_calc_phi, grid_direct_sum_complex

p_range = np.arange(0,35).astype('i')
m_range = []
max_errs = []
lvlss = []

# time keeper for different components (keys)
innerkeys = ['S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
keys = ['construct_tree', 'fmm_calc', 'direct_sum'] + innerkeys
times = {key:[] for key in keys}

for p in p_range:
    # initialise fixed parameters
    n = 500 # takes arund 1m30s to run
    m = 10
    m_range.append(m)
    lvlextra = 0
    lvls = lvls_fmm(n, m, lvlextra)
    
    print()
    print()
    print()
    very_start = time.time()
    print(f"------ p = {p}, n = {n}, lvls = {lvls}, m = {m} ------")
    print("Timer elapsed reset.")

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
    tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, m, p)
    toc = time.perf_counter()
    lvlss.append(lvls)
    times[keys[0]].append(toc-tic)
    print(f'{keys[0]} ed.')
    # print('Time elapsed:', time.time() - very_start)

    # FMM calculation
    tic = time.perf_counter()
    innertimes = fmm_calc_phi(tree, idx_helpers, lvls, p)
    toc = time.perf_counter()

    # store component timings
    times[keys[1]].append(toc-tic)
    for key in innerkeys:
        if innertimes[key]:
            times[key].append(innertimes[key][0])
    # print('Time elapsed:', time.time() - very_start)

    # FMM calculation output stored in "fmm"
    fmm = gridcomplex.get_all_phi()
    # print('Time elapsed:', time.time() - very_start)

    # direct calculation
    tic = time.perf_counter()
    grid_direct_sum_complex(gridcomplex)
    toc = time.perf_counter()
    times[keys[2]].append(toc-tic)
    print('Directly calculated.')
    print('Time elapsed:', time.time() - very_start)

    # direct calculation output stored in "exactcomplex"
    exactcomplex = np.array(gridcomplex.get_all_phi())

    # (absolute) fractional error calculation
    fmm_errs = (fmm-exactcomplex).real/exactcomplex.real
    data = abs(fmm_errs)
    max_errs.append(max(data))
print()

# FMM plot: t vs p
if input("Generate t vs p plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,6))
    xlabel = 'p_range'
    x = eval(xlabel)

    # select all components to plot except contruct_tree and direct_sum
    excluded_keys = ['construct_tree', 'direct_sum']
    step_keys = [key for key in keys if key not in excluded_keys]

    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        # fit to obtain power law exponent
        X = np.log(x[x!=0])
        Y = np.log(y[x!=0])
        a, b = np.polyfit(X, Y, 1)
        
        # default linestyle, label, scaling (x10 as most values are too small)
        linestyle = '-'
        label = step_keys[i]

        # special cases:
        if step_keys[i] == 'fmm_calc':
            linestyle = '--'
        elif step_keys[i] == 'M2L':
            linestyle = ':'
        elif step_keys[i] == 'P2P':
            linestyle = '-.'
        ax.plot(x, y, label = f'{label},  $t \propto {xlabel[0]}^{{{a:.2f}}}$', linestyle=linestyle)

    title = f'FMM $t$ vs $p$;   $N$={n}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel('log $N$')
    ax.set_xlabel('$p$ (order of expansion)')
    ax.set_ylabel('log $t$')
    ax.set_ylabel('$t$ / s')
    plt.legend()
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# FMM plot: log t vs p
if input("Generate log t vs p plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,8))
    xlabel = 'p_range'
    x = eval(xlabel)

    # selecting components (keys) to plot
    # all keys: [construct_tree, fmm_calc, direct_sum, S2M, M2M, M2L, L2L, L2P, P2P]
    # excluded_keys = ['direct_sum', 'construct_tree', 'S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
    excluded_keys = ['direct_sum', 'construct_tree'] # to compare components 
    step_keys = [key for key in keys if key not in excluded_keys]

    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        X = np.log10(x[x!=0])
        Y = np.log10(y[x!=0])
        a, b = np.polyfit(X, Y, 1)
        linestyle = '-'
        label = step_keys[i]
        if step_keys[i] == 'fmm_calc':
            linestyle = '--'
            ax.plot(10**X, a*X+b, 
                    label=f'{label}, $t \propto {xlabel[0]}^{{{a:.2f}}}$ (fit)', 
                    linestyle='--')
        elif step_keys[i] in ['S2M', 'L2P', 'P2P']:
            linestyle = ':'
        ax.plot(10**X, Y, label=f'{label}, $t \propto {xlabel[0]}^{{{a:.2f}}}$',
                 linestyle=linestyle, lw=3)
        # if step_keys[i] in ['fmm_calc', 'direct_sum', 'construct_tree']:

    title = f'FMM log $t$ vs ${xlabel[0]}$;   $N$={n}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel(f'${xlabel[0]}$')
    ax.set_ylabel('log$_{10}$ $t$')
    plt.legend(loc='upper left')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# FMM plot: error vs p
if input("Generate error vs p plot? (y/n) ") == 'y':
    fig, ax = plt.subplots()
    x = p_range
    y = np.array(max_errs)
    Y = np.log10(y[x!=0])
    ax.plot(x[x>0], Y, label = 'max_error')

    # polynomial fitting - only fit those before machine precision
    fit_upper = 26
    a, b = np.polyfit(x[:fit_upper], Y[:fit_upper], 1)
    ax.plot(x[x>0][:fit_upper], a*x[:fit_upper] + b, label=f'fit: max_error $\propto e^{{{a:.2f}p}}$')

    title = f'FMM log max_error vs $p$;   $N$={n}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel('$p$')
    ax.set_ylabel('log$_{10}$ max_error')
    plt.legend()
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# FMM plot: t_M2L vs p
if input("Generate t_M2L vs p plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (8,6))
    xlabel = 'p_range'
    x = eval(xlabel)
    y = np.array(times['M2L'])
    ax.plot(x, y, label = 'expm  $t_{M2L}$')
    X = np.log(x[x!=0])
    Y = np.log(y[x!=0])
    a, b = np.polyfit(X, Y, 1)
    ax.plot(x, np.exp(b)*x**a, label = f'Power-law fit  $t \propto {xlabel[0]}^{{{a:.2f}}}$', linestyle = '--')
    # exponential fit: meaningless
    # X = x[x!=0]
    # Y = np.log(y[x!=0])
    # a, b = np.polyfit(X, Y, 1)
    # ax.plot(x, np.exp(b)*np.exp(a*x), label = f'Exponential fit  $t \propto e^{{{{{a:.2f}}}{xlabel[0]}}}$', linestyle = ':')

    title = f'FMM $t_{{M2L}}$ vs $p$;   $N$={n}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel('$p$ (order of expansion)')
    ax.set_ylabel('$t_{M2L}$ / s')
    plt.legend()
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()
