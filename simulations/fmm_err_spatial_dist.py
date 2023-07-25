# FMM - spatial distribution of error
# This simulation gives a visualisation of the spatial distribution of the errors on the particles
print(
"""
In the event of ImportError:
1. Type below in the command prompt, from the directory that contains the N-Body-Barnes-Hut-FMM package:
        > python -m N-Body-Barnes-Hut-FMM.simulations.fmm_err_spatial_dist
   Note: -m tells Python to load fmm_err_spatial_dist.py as a module (instead of a top-level script
   which relative imports like ..classes won't work)
--------------------
In the event of ModuleNotFoundError:
1. CHANGE "classes" to "..classes" in helperfunctions/fmm_functions.py
2. Perform step 1 of above "In the event of ImportError".
3. After running, CHANGE  "..classes" back to "classes" for the module to work in Jupyter notebook.


""")
import numpy as np
import time
import matplotlib.pyplot as plt

from ..classes import GridComplex
from ..helperfunctions import construct_tree_fmm, fmm_calc_phi, grid_direct_sum_complex

p = 5
n = 5000 # n=5000 takes around 4 mins to run
ptcmax = 10
lvls = int(np.ceil(np.emath.logn(4, n/ptcmax)))

very_start = time.time()
print(f"------ p = {p}, n = {n}, lvls = {lvls}, ptcmax = {ptcmax} ------")
print("Timer elapsed reset.")
np.random.seed(4)
gridcomplex = GridComplex(size=128)
all_coords = [i for i in range(n)]
all_q = np.random.random(len(all_coords))*10+50
all_particles = gridcomplex.create_particles(len(all_coords), all_coords=None, all_q=all_q)
print('Grid and particles initialised.')

tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, ptcmax, p)
if crowded:
    while crowded:
        lvls+=1
        tic = time.perf_counter()
        tree, idx_helpers, crowded = construct_tree_fmm(lvls, gridcomplex, ptcmax, p)
        toc = time.perf_counter()
    print(f'lvls readjusted to {lvls}.')
print('Tree constructed.')

innertimes = fmm_calc_phi(tree, idx_helpers, lvls, p)
fmm = gridcomplex.get_all_phi()

grid_direct_sum_complex(gridcomplex)
print(f'Directly calculated.')
print('Time elapsed:', time.time() - very_start)

exactcomplex = np.array(gridcomplex.get_all_phi())
fmm_errs = (fmm-exactcomplex).real/exactcomplex.real

fig, ax = plt.subplots(figsize=(9,7))
x, y = (lambda x: (x.real, x.imag))(gridcomplex.get_all_coords())

data = abs(fmm_errs)
sc = plt.scatter(x, y, c=data, vmin = min(data), vmax=max(data), cmap = 'cividis_r', s=3)
plt.colorbar(sc, label='FMM fractional error')
plt.show()