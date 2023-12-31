# FMM: N variation
"""The below code runs the FMM code for different $N$ (number of particles).
$p$ and $m$ are fixed. The timing as a function of $N$ will be examined.
"""
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.fmm_n_variation
   Note: -m tells Python to load fmm_n_variation.py as a module (instead of a top-level script
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

# timing estimations uses p = 10
# N_range = np.logspace(2,3,5).astype('i') # [100, 1000]; takes around 30s to run; 1m45s with direct
# N_range = np.logspace(2,4,8).astype('i') # [100, 10000]; takes around 9m to run; 50m! with direct
# N_range = np.logspace(2,4.5,10).astype('i') # [100, 31622]; took around 6m to run!
# N_range = np.logspace(2, 5.5, 8).astype('i') # estimated runtime: 1 hour (without direct) # ran overnight? interrupted
# N_range = np.logspace(2, 5.3, 4).astype('i') # [100, 199526] took around 50 m to run (without direct)
# N_range = np.logspace(2, 6, 12).astype('i') # too ambitious for p = 10

# post lvls_fmm:
N_range = np.logspace(2, 4.5, 10).astype('i') # [100, 31622]; took around 2m to run!
# N_range = np.logspace(2, 5, 10).astype('i') # [100, 100000]; 11m

# N_range = np.logspace(2, 3.5, 6).astype('i') # 1m with direct
# N_range = np.logspace(2, 4, 6).astype('i') # 8m with direct

p_range = []
m_range = []
max_errs = []
lvlss = []

# skip direct calculation to save time (set run_direct to False)
run_direct = False
if run_direct:
    print('Direct calculation will be executed - this will take a while to run.')

# time keeper for different components (keys)
innerkeys = ['S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
keys = ['construct_tree', 'fmm_calc', 'direct_sum'] + innerkeys
times = {key:[] for key in keys}

# loop FMM over different N
for n in N_range:
    # initialise fixed parameters
    p = 5
    p_range.append(p)
    m = 10
    m_range.append(m)
    lvlextra = 0
    lvls = lvls_fmm(n, m, lvlextra)

    print()
    print()
    print()
    print(f"------ p = {p}, n = {n}, lvls = {lvls}, m = {m} ------")
    very_start = time.time()
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

    # skip direct calculation by default as that takes much more time
    if run_direct:
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

# FMM plot: lvls vs N
if input("Generate lvls vs N plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,6))
    xlabel = 'N_range'
    x = eval(xlabel)
    y = lvlss
    title = 'FMM Number of levels vs log $N$'
    ax.set_title(title)
    ax.plot(np.log10(x), y)
    ax.set_xlabel(f'log$_{{{10}}}$ ${xlabel[0]}$')
    ax.set_ylabel('lvls')
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + '.jpg', dpi=300)
print()

# FMM plot: t vs N (linear)
if input("Generate t vs N (linear) plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,6))
    xlabel = 'N_range'
    x = eval(xlabel)

    # selecting components (keys) to plot
    excluded_keys = ('direct_sum')
    # excluded_keys = ('direct_sum', 'fmm_calc', 'M2L')
    if run_direct:
        excluded_keys = ()
    step_keys = [key for key in keys if key not in excluded_keys]

    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        # fit to obtain power law exponent
        X = np.log(x[y!=0])
        Y = np.log(y[y!=0])
        a, b = np.polyfit(X, Y, 1)

        # default linestyle, label, scaling (x10 for construct_tree)
        linestyle = '-'
        label = step_keys[i]
        
        # special cases:
        if step_keys[i] == 'fmm_calc':
            linestyle = '--'
        elif step_keys[i] == 'M2L':
            linestyle = ':'
        elif step_keys[i] == 'P2P':
            linestyle = '-.'
        elif step_keys[i] == 'construct_tree':
            y *= 10
            label = step_keys[i] + ' (×10)'
        ax.plot(x, y, label=f'{label},  $t \propto {xlabel[0]}^{{{a:.2f}}}$', linestyle=linestyle)

    title = f'FMM $t$ vs ${xlabel[0]}$;   $p$={p}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel(f'log ${xlabel[0]}$')
    ax.set_xlabel('$N$ particles')
    ax.set_ylabel('log $t$')
    ax.set_ylabel('$t$ / s')
    plt.legend(loc='upper left')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        if run_direct:
            title += ' (direct included)'
        fig.savefig(title + '.jpg', dpi=300)
print()

# FMM plot: t vs N (log) (main components only)
if input("Generate t vs N (log) plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,8))
    xlabel = 'N_range'
    x = eval(xlabel)

    # selecting components (keys) to plot
    # all keys: [construct_tree, fmm_calc, direct_sum, S2M, M2M, M2L, L2L, L2P, P2P]
    excluded_keys = ['direct_sum', 'S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
    # excluded_keys = ['direct_sum'] # to compare components 
    if run_direct:
        excluded_keys.remove('direct_sum')
    step_keys = [key for key in keys if key not in excluded_keys]

    # plot each component
    for i in range(len(step_keys)):
        y = np.array(times[step_keys[i]])
        X = np.log10(x[y!=0])
        Y = np.log10(y[y!=0])
        a, b = np.polyfit(X, Y, 1)
        linestyle = '-'
        label = step_keys[i]
        if step_keys[i] == 'fmm_calc':
            linestyle = '--'
        elif step_keys[i] == 'M2L':
            linestyle = ':'
        ax.plot(X, Y, label=f'{label}, experimental', linestyle=linestyle, lw=3)
        if step_keys[i] in ['fmm_calc', 'direct_sum', 'construct_tree']:
            ax.plot(X, a*X+b, label=f'{label}, fit: $t \propto {xlabel[0]}^{{{a:.2f}}}$', linestyle='--')

    title = f'FMM log $t$ vs log ${xlabel[0]}$;   $p$={p}, $m$={m}'
    ax.set_title(title)
    ax.set_xlabel(f'log$_{{{10}}}$ ${xlabel[0]}$')
    ax.set_ylabel('log$_{10}$ $t$')
    plt.legend(loc='upper left')
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        if run_direct:
            title += ' (direct included)'
        fig.savefig(title + '.jpg', dpi=300)
print()
