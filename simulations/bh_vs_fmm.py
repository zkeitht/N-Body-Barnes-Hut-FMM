# BH vs FMM
"""The below code runs the BH and the FMM code for different N.
θ, p, and m are fixed. The timing and difference between the two output as a 
function of N will be examined. 
If direct sum is also run, the errors of both will be plotted.
"""
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.bh_vs_fmm
   Note: -m tells Python to load bh_theta_variation.py as a module (instead of a
   top-level script which relative imports like ..classes won't work)

   
   
""")
import numpy as np
import matplotlib.pyplot as plt
import time

from ..classes import Grid, GridComplex
from ..functions import bh_create_tree, bh_calc_phi, grid_direct_sum
from ..functions import lvls_fmm, construct_tree_fmm, fmm_calc_phi


# N_range = np.logspace(2, 3.2, 6).astype('i') # 1m with direct
# N_range = np.logspace(2, 3.5, 6).astype('i') # [100, 3162]; 3m with direct
# N_range = np.logspace(2, 4, 6).astype('i') # 21m with direct
N_range = np.logspace(2, 4, 10).astype('i') # 35m with direct
# N_range = np.logspace(2, 4.5, 6).astype('i') # 21m with direct 2m without (p=3)
# N_range = np.logspace(2, 4.5, 6).astype('i') # 21m with direct 5m without (p=6)
# N_range = np.logspace(2, 5, 6).astype('i') # don't try with direct; 7m without (p=6)
# N_range = np.logspace(2, 5, 10).astype('i') # don't try with direct; 22m without (p=6)
keys_bh = ['create_tree', 'bh_calc', 'direct_sum']
times_bh = {key:[] for key in keys_bh}

innerkeys = ['S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
keys_fmm = ['construct_tree', 'fmm_calc', 'direct_sum'] + innerkeys
times_fmm = {key:[] for key in keys_fmm}
max_errs_bh = []
max_errs_fmm = []
diffs = []
direct = 1

nrange_start = time.time()

for n in N_range:
    # initialise fixed parameters bh
    theta = 0.5

    # initialise fixed parameters fmm
    p = 6
    m = 10
    lvlextra = 0
    lvls = lvls_fmm(n, m, lvlextra)

    print()
    print()
    print()
    print(f"------ n={n}, θ={theta}, p={p}, m={m} ------")
    very_start = time.time()
    print("Timer elapsed reset.")

    # initialise particles bh
    np.random.seed(4)
    grid_bh = Grid()
    grid_bh.create_particles(n)

    # initialise particles fmm
    np.random.seed(4)
    grid_fmm = GridComplex()
    grid_fmm.create_particles(n)

    # BH tree construction
    tic = time.perf_counter()
    bh_create_tree(grid_bh, grid_bh.particles)
    toc = time.perf_counter()
    times_bh[keys_bh[0]].append(toc-tic)
    # print('Time elapsed:', time.time() - very_start)

    # FMM tree construction
    tic = time.perf_counter()
    tree, idx_helpers, crowded = construct_tree_fmm(lvls, grid_fmm, m, p)
    toc = time.perf_counter()
    times_fmm[keys_fmm[0]].append(toc-tic)
    # print(f'{keys_fmm[0]} ed.')
    
    # BH calculation
    tic = time.perf_counter()
    bh_calc_phi(grid_bh, theta)
    toc = time.perf_counter()
    times_bh[keys_bh[1]].append(toc-tic)
    print('BH calculated.')
    print('Time elapsed:', time.time() - very_start)

    # BH calculation output stored in "bh"
    bh = np.array(grid_bh.get_all_phi())
    # print('Time elapsed:', time.time() - very_start)

    # FMM calculation
    tic = time.perf_counter()
    innertimes = fmm_calc_phi(tree, idx_helpers, lvls, p)
    toc = time.perf_counter()
    # store component timings
    times_fmm[keys_fmm[1]].append(toc-tic)
    for key in innerkeys:
        if innertimes[key]:
            times_fmm[key].append(innertimes[key][0])
    # print('Time elapsed:', time.time() - very_start)

    # FMM calculation output stored in "fmm"
    fmm = np.array(grid_fmm.get_all_phi())
    # print('Time elapsed:', time.time() - very_start)
    
    # FMM - BH 
    # expect FMM to be much more accurate than BH,
    # and BH's error decreases as N increases
    # hence the difference graph will decrease as N increases.
    diff = (fmm-bh).real/fmm.real
    data = abs(diff)
    diffs.append(max(data))

    if direct:
        # direct calculation
        tic = time.perf_counter()
        grid_direct_sum(grid_bh)
        toc = time.perf_counter()
        times_bh[keys_bh[2]].append(toc-tic)
        print('Directly calculated.')
        print('Time elapsed:', time.time() - very_start)

        # direct calculation output stored in "exact"
        exact = np.array(grid_bh.get_all_phi())
        
        # (absolute) fractional error calculation bh
        bh_errs = (bh-exact)/exact
        data = abs(bh_errs)
        max_errs_bh.append(max(data))
        
        # (absolute) fractional error calculation fmm
        fmm_errs = (fmm-exact).real/fmm.real
        data = abs(fmm_errs)
        max_errs_fmm.append(max(data))
print(f'\n\nNtime: {(time.time() - nrange_start)/60:.2f} mins\n')

if input("Generate FMM vs BH (difference or error) plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (8,5))
    if direct:
        ax.plot(np.log10(N_range), max_errs_bh, label='bh')
        ax.plot(np.log10(N_range), max_errs_fmm, label='fmm')
        title = f"BH vs FMM error;   $θ$={theta}, $p$={p}, $m$={m}"
    else:
        ax.plot(np.log10(N_range), diffs, label='FMM-BH')
        title = f"FMM - BH;   $θ$={theta}, $p$={p}, $m$={m}"

    ax.set_title(title)
    ax.set_xlabel('log$_{10}$ $N$')
    ax.set_ylabel('max_frac_error')
    plt.legend()
    plt.show()
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + f" N up to {N_range[-1]}"+".jpg", dpi=300)
print()

if input("Generate t vs N (log) plot? (y/n) ") == 'y':
    fig, ax = plt.subplots(figsize = (10,7))
    xlabel = 'N_range'
    x = eval(xlabel)

    # plot bh
    # selecting components (keys) to plot
    excluded_keys_bh = ['direct_sum', 'create_tree']
    if direct:
        excluded_keys_bh.remove('direct_sum')
    step_keys_bh = [key for key in keys_bh if key not in excluded_keys_bh]

    # plot each component bh
    for step_key in step_keys_bh:
        y = np.array(times_bh[step_key])
        X = np.log10(x[x!=0])
        Y = np.log10(y[x!=0])
        a, b = np.polyfit(X, Y, 1)
        ax.plot(X, Y, label = f'{step_key}, bh experimental', lw=2)
        ax.plot(X, a*X+b, label=f'{step_key}, fit  $t \propto N^{{{a:.2f}}}$', linestyle=':')

    # plot fmm
    excluded_keys_fmm = ['direct_sum', 'construct_tree', 'S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
    # excluded_keys_fmm = ['direct_sum', 'S2M', 'M2M', 'fmm_calc', 'L2L', 'L2P', 'P2P']
    step_keys = [key for key in keys_fmm if key not in excluded_keys_fmm]

    # plot each component fmm
    for step_key in step_keys:
        y = np.array(times_fmm[step_key])
        X = np.log10(x[x!=0])
        Y = np.log10(y[x!=0])
        a, b = np.polyfit(X, Y, 1)
        linestyle = '-'
        if step_key == 'fmm_calc':
            linestyle = '--'
        label = step_key
        ax.plot(X, Y, label=f'{label}, fmm experimental', linestyle=linestyle, lw=2)
        ax.plot(X, a*X+b, label=f'{label}, fit: $t \propto {xlabel[0]}^{{{a:.2f}}}$', linestyle=':')

    title = f'BH vs FMM log $t$ vs log ${xlabel[0]}$;   $θ$ = {theta}, $p$ = {p}, $m$ = {m}'
    ax.set_title(title)
    ax.set_xlabel('log$_{10}$ $N$')
    ax.set_ylabel('log$_{10}$ $t$')
    plt.legend(loc='upper left')
    plt.show()
    if direct:
        title += ' (direct included)'
    if input("save plot? (y/n) ") == 'y':
        fig.savefig(title + f" N up to {N_range[-1]}"+".jpg", dpi=300)
print()
