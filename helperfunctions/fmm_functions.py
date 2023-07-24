# FMM (helper) functions
import numpy as np

from classes import BoxComplex

def four_fractal(lvl):
    """Returns array of integer in a specific format to help in the indexing of FMM boxes.
    """
    if lvl>0:
        a0 = np.array([[0,1],[2,3]])
        i_4 = np.ones((2,2))
        a = np.array([[0,1],[2,3]])
        for n in range(1, lvl):
            N = int(2**n)
            i_4n = np.ones((N,N)) * 4**n
            a = np.kron(a0,i_4n) + np.kron(i_4,a)
        return a.T.astype('i').tolist()
    return None

def bin_coords_to_ij(items, lvl):
    grid = items[0].grid
    all_coords = [(item.coords.real, item.coords.imag) for item in items]
    all_coords_x, all_coords_y = [coord for coord in zip(*all_coords)]
    bin_is = np.digitize(all_coords_x, np.arange(0, grid.size, grid.size/(2**lvl)))-1
    bin_js = np.digitize(all_coords_y, np.arange(0, grid.size, grid.size/(2**lvl)))-1
    return bin_is, bin_js

def construct_tree_fmm(lvls, grid, ptcmax, p):
    tree = [[BoxComplex(grid.size/2+1j*grid.size/2, size=grid.size, ptcmax=ptcmax, grid=grid,p=p)]]
    crowded = False
    crowd = {key:[] for key in ('coords', 'nptcs')}
    # empty tree
    if lvls > 0:
        for lvl in range(1, lvls+1):
            next_lvl = []
            for i in range(len(tree[lvl-1])):
                next_lvl.extend(tree[lvl-1][i].quadrant_split())
            tree.append(next_lvl)
    # insert particles into leaf level
    idx_helpers = [four_fractal(lvl) for lvl in range(lvls+1)]
    bin_is, bin_js = bin_coords_to_ij(grid.particles, lvls)
    for ptc, bin_i, bin_j in zip(grid.particles, bin_is, bin_js):
        tree[lvls][idx_helpers[lvls][bin_i][bin_j]].particles.append(ptc)
    for leaf_box in tree[lvls]:
        if len(leaf_box.particles) >= ptcmax:
            crowd['coords'].append(leaf_box.coords)
            crowd['nptcs'].append(len(leaf_box.particles))
    if crowd['nptcs']:
        print(f"Leaf box(es) centered at {crowd['coords']} too crowded, it has {crowd['nptcs']} particles. Try increasing 'lvls'.")
        crowded = True

    return tree, idx_helpers, crowded

def direct_neighbours(box, tree, idx_helpers, inclself=True):
    """Returns the direct neighbours of a given box."""
    lvl = int(np.log2(box.grid.size/box.size))
    same_lvl_boxes = tree[lvl]
    idx_helper = idx_helpers[lvl]
    max_ij = int(2**lvl)-1
    box_i, box_j = bin_coords_to_ij([box], lvl)

    dir_neighs = []
    irange = range(int(max(0, box_i-1)), int(min(box_i+2, max_ij+1)))
    jrange = range(int(max(0, box_j-1)), int(min(box_j+2, max_ij+1)))
    for i in irange:
        for j in jrange:
            dir_neighs.append(same_lvl_boxes[idx_helper[i][j]])
    if inclself:
        return dir_neighs
    else:
        dir_neighs.remove(box)
        return dir_neighs

def interaction_list(box, tree, idx_helpers):
    """Returns the interaction list of a given box."""
    lvl = int(np.log2(box.grid.size/box.size))
    same_lvl_boxes = tree[lvl]
    idx_helper = idx_helpers[lvl]
    max_ij = int(2**lvl)-1
    box_i, box_j = bin_coords_to_ij([box], lvl)
    i_odd, j_odd = (box_i%2 != 0), (box_j%2 != 0)
    i_low, j_low = box_i-2-i_odd, box_j-2-j_odd
    i_upp, j_upp = i_low+6, j_low+6
    irange = range(int(max(0, i_low)), int(min(i_upp, max_ij+1)))
    jrange = range(int(max(0, j_low)), int(min(j_upp, max_ij+1)))

    interaction_list = []
    for i in irange:
        for j in jrange:
            interaction_list.append(same_lvl_boxes[idx_helper[i][j]])

    dir_neighs = direct_neighbours(box, tree, idx_helpers, inclself=1)
    for dn in dir_neighs:
        interaction_list.remove(dn)
    
    return interaction_list