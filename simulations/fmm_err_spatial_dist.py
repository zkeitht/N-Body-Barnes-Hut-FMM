# FMM plot: spatial distribution of error
# This simulation gives a visualisation of the spatial distribution of the 
# errors on the particles.
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.fmm_err_spatial_dist
   Note: -m tells Python to load fmm_err_spatial_dist.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)
--------------------
In the event of ModuleNotFoundError:
1. CHANGE "classes" to "..classes" in functions/fmm_functions.py
2. Perform step 1 of above "In the event of ImportError".
3. After running, CHANGE  "..classes" back to "classes" for the module to work in Jupyter notebook.


""")
import numpy as np
import time
import matplotlib.pyplot as plt

from ..classes import GridComplex
from ..functions import lvls_fmm, construct_tree_fmm, fmm_calc_phi, grid_direct_sum_complex

# initialise fixed parameters
p = 15 # n=5000, p=15 takes around 6 mins to run
p = 10
n = 5000 
m = 10
lvlextra = 0
lvls = lvls_fmm(n, m, lvlextra)

very_start = time.time()
print(f"------ p = {p}, n = {n}, lvls = {lvls}, m = {m} ------")
print("Timer elapsed reset.")

# initialise particles
np.random.seed(4)
gridcomplex = GridComplex(size=128)
all_coords = [i for i in range(n)]
all_q = np.random.random(len(all_coords))*10+50
all_particles = gridcomplex.create_particles(len(all_coords), all_coords=None, all_q=all_q)
print('Grid and particles initialised.')

# FMM tree construction
tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, m, p)
# if crowded:
#     while crowded:
#         lvls+=1
#         tic = time.perf_counter()
#         tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, m, p)
#         toc = time.perf_counter()
#     print(f'lvls readjusted to {lvls}.')
print('Tree constructed.')

# FMM calculation
innertimes = fmm_calc_phi(tree, idx_helpers, lvls, p)

# FMM calculation output stored in "fmm"
fmm = gridcomplex.get_all_phi()

# direct calculation
grid_direct_sum_complex(gridcomplex)
print(f'Directly calculated.')
print('Time elapsed:', time.time() - very_start)

# direct calculation output stored in "exactcomplex"
exactcomplex = np.array(gridcomplex.get_all_phi())

# (absolute) fractional error calculation
fmm_errs = (fmm-exactcomplex).real/exactcomplex.real

# plot spatial distribution of errors
fig, ax = plt.subplots(figsize=(9,7))
x, y = (lambda x: (x.real, x.imag))(gridcomplex.get_all_coords())
data = abs(fmm_errs)
sc = plt.scatter(x, y, c=data, vmin = min(data), vmax=max(data), cmap = 'cividis_r', s=3)
title = f'FMM spatial distribution of error;   $N$={n}, $p$={p}, $m={m}$'
plt.title(title)
plt.colorbar(sc, label='(abs.) fractional error')
plt.show()
if input("save plot? (y/n) ") == 'y':
    fig.savefig(title + '.jpg', dpi=300)
