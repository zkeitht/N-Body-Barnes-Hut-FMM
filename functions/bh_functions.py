# BH functions
import numpy as np

from .real_functions import direct_source_target

def bh_tree_insert_particle(particle, box, counter=0):
    """Inserts a particle into a box recursively into a tree structure.
    """
    box.add_com(particle)
    # box.print_step(particle, box, add_com=1)

    if len(box.particles) < box.m: # if there is vacancy
        box.particles.append(particle)
        # box.print_step(particle, box, add_ptc=1)
    elif box.children: # children exists (and box full)
        # print(f'box {id(box)} full, children exist.')
        child_box = box.get_child_quadrant(particle)
        bh_tree_insert_particle(particle, child_box, counter=counter)
    else: # no children (and box full)
        # print(f'box {id(box)} full, creating children.')
        particles = box.particles + [particle]
        box.quadrant_split()
        child_boxes = [box.get_child_quadrant(particle) for particle in particles]
        for particle, child_box in zip(particles, child_boxes):
            bh_tree_insert_particle(particle, child_box, counter=counter)

def bh_create_tree(grid, particles):
    """Constructs the tree structure required for the BH method."""
    rootbox = grid.rootbox
    assert len(rootbox.children)==0, \
            f"Root box already has a children. Start with a new tree."
    for p in particles:
        # print('New particle')
        bh_tree_insert_particle(p, rootbox)
        # print()

def bh_calc_at_particle(particle, box, theta=0.5):
    """Calculates the potential induced by a box evaluated at a particle's location 
    using the BH allgorithm.
    """
    s = box.size
    r = particle.distance(box.com_R)
    if r == 0: # the particle itself
        return
    if s/r < theta: # box far
        # print('box far')
        # particle.phi -= box.com_M/r # old potential
        particle.phi += box.com_M*np.log(r)
        # print(f'Added com contribution from box {id(box)}.')
    elif box.children: # box near, children exist
        # print('box near')
        for child_box in box.children:
            if child_box.particles: # only evaluate non-empty
                bh_calc_at_particle(particle, child_box, theta)
    else: # box near, no children
        # print('box near no child')
        sources = [p for p in box.particles if p is not particle]
        target = [particle]
        direct_source_target(sources, targets = target)
        # print(f'Added contributions from sources {[id(s) for s in sources]} in box {id(box)}.')

def bh_calc_phi(grid, theta=0.5):
    """Iterates through the whole grid to calculate the potential of every particle"""
    rootbox = grid.rootbox
    grid.clear_all_phi()
    for particle in grid.particles:
        bh_calc_at_particle(particle, rootbox, theta)

def find_bh_tree_depth(box, gridsize):
    """Finds maximum depth of a BH tree"""
    depth = 0
    if box.children:
        for child in box.children:
            depth = max(depth, find_bh_tree_depth(child, gridsize))
        return depth + 1
    else:
        return 0