# FMM steps
""" Uses FMM core calculations, and combines all of the steps in the final 
fmm_calc_phi function. fmm_calc_phi also keeps track of the time used by 
each of the steps."""
import time

from .complex_functions import direct_sum_complex, direct_source_target_complex
from .fmm_functions import direct_neighbours, interaction_list
from .fmm_core_calculations import S2M_coeff, M2M_translation, M2L_coeff, sum_local, L2L_translation

def fmm_step_S2M(tree, lvls, p):
    """Finds source representation.
    Information flow: source particles -> leaf child box
    """
    for child in tree[lvls]:
        child.S2M_coeffs = S2M_coeff(child.particles, child, p)

def fmm_step_M2M(tree, lvls):
    """Transfers source representation. 
    Information flow: child -> parent
    """
    for lvl in range(lvls-1, 1, -1):
        for par in tree[lvl]:
            for child in par.children:
                par.S2M_coeffs = par.S2M_coeffs + M2M_translation(child.S2M_coeffs, child, par)

def fmm_step_M2L(box, tree, idx_helpers):
    """Evaluates target representation from (far) source representation. 
    Information flow: far source (i.e. interaction list) -> target
    """
    for far_source_box in interaction_list(box, tree, idx_helpers):
        box.M2L_coeffs = box.M2L_coeffs + M2L_coeff(far_source_box.S2M_coeffs, far_source_box, box)

def fmm_step_L2L(box):
    """Transfers target representation. 
    Information flow: parent -> child
    """
    for child in box.children:
        child.M2L_coeffs = child.M2L_coeffs + L2L_translation(box.M2L_coeffs, box, child)

def fmm_step_L2P(leaf_box):
    """Uses target representation to evaluate the potential.
    Information flow: leaf child box -> target particles
    """
    target_coords = [ptc.coords for ptc in leaf_box.particles]
    phis = sum_local(leaf_box.M2L_coeffs, leaf_box, target_coords)
    for phi, ptc in zip(phis, leaf_box.particles):
        ptc.phi += phi

def fmm_step_P2P(leaf_box, tree, idx_helpers):
    """Evaluates potential directly from nearby sources.
    Information flow: 
    neighbouring boxes + leaf child box sources -> leaf child box targets
    """
    sources = []
    for neigh_leaf in direct_neighbours(leaf_box, tree, idx_helpers, inclself=0):
        sources.extend(neigh_leaf.particles)
    direct_source_target_complex(sources, leaf_box.particles)
    direct_sum_complex(leaf_box.particles)

def fmm_calc_phi(tree, idx_helpers, lvls, p):
    """Combines all the fmm_steps.
    The resulting potentials are stored in the phi attribute of the particles.
    1. S2M: Finds source representation.
        (source particles -> leaf child box)
    2. M2M: Transfers source representation. 
        (child -> parent)
    3. M2L: Evaluates target representation from (far) source representation. 
        (far source -> target)
    4. L2L: Transfers target representation. 
        (parent -> child)
    5. L2P: Uses target representation to evaluate the potential. 
        (leaf child box -> target particles)
    6. P2P: Evaluates potential directly from nearby sources.
        (neighbouring boxes + leaf child box sources -> leaf child box targets)

    Returns
    -------
    a dictionary of times elapsed for each of the steps.
    """
    print()
    print('start of fmm calc...')
    fmm_start = time.time()

    keys = ['S2M', 'M2M', 'M2L', 'L2L', 'L2P', 'P2P']
    times = {key:[] for key in keys}

    #1
    tic = time.perf_counter()
    fmm_step_S2M(tree, lvls, p)
    toc = time.perf_counter()
    times[keys[0]].append(toc-tic)
    # print(f'{keys[0]} ed.')
    # print('Time elapsed:', time.time() - fmm_start)

    #2
    tic = time.perf_counter()
    fmm_step_M2M(tree, lvls)
    toc = time.perf_counter()
    times[keys[1]].append(toc-tic)
    # print(f'{keys[1]} ed.')
    # print('Time elapsed:', time.time() - fmm_start)

    time_M2L = 0
    time_L2L = 0
    for lvl in range(2, lvls+1):
        for box in tree[lvl]:
            #3
            tic = time.perf_counter()   
            fmm_step_M2L(box, tree, idx_helpers)
            toc = time.perf_counter()
            time_M2L += toc-tic

            if lvl >= lvls:
                continue

            #4
            tic = time.perf_counter()   
            fmm_step_L2L(box)
            toc = time.perf_counter()
            time_L2L += toc-tic
    
    times[keys[2]].append(time_M2L)
    # print(f'{keys[2]} ed.')
    times[keys[3]].append(time_L2L)
    # print(f'{keys[3]} ed.')
    # print('Time elapsed:', time.time() - fmm_start)
    
    time_L2P = 0
    time_P2P = 0
    for leaf_box in tree[-1]:
        #5
        tic = time.perf_counter()   
        fmm_step_L2P(leaf_box)
        toc = time.perf_counter()
        time_L2P += toc-tic

        #6
        tic = time.perf_counter()   
        fmm_step_P2P(leaf_box, tree, idx_helpers)
        toc = time.perf_counter()
        time_P2P += toc-tic
    
    times[keys[4]].append(time_L2P)
    # print(f'{keys[4]} ed.')
    times[keys[5]].append(time_P2P)
    # print(f'{keys[5]} ed.')
    print('Time elapsed within fmm:', time.time() - fmm_start)
    print('end of fmm calc')
    print()
    return times