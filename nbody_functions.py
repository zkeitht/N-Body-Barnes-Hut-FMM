# nbody_functions

def direct_sum(particles, grid):
    """Calculates the potential for each particle using direct summation method.

    Potential := Σ-(q/r), where q is the charge of the source particles and 
    r is the distance between the particle and the source particles. 
    The potential of each of the particles is stored in the phi attribute of 
    the particles.

    This can be used for all particles in existance, or just for a subset 
    of particles.
    """
    grid.clear_all_phi()
    for i, target in enumerate(particles):
        for source in (particles[:i] + particles[i+1:]):
            r = target.distance(source)
            target.phi -= (source.q)/r


def direct_source_target(sources, targets):
    """Calculates the potential induced by a group of SOURCE particles, 
    evaluated at each of the TARGET particles, using direct summation method.

    Potential := Σ-(q/r), where q is the charge of the SOURCE particles and 
    r is the distance between the target and the source particles. 
    The potential evaluated at each of the TARGET particles is stored in the 
    phi attribute of the TARGET particles.
    Interactions within the source group and within the target group are ignored.
    Arguments:
        sources: list of source particles
        targets: list of target particles
    """
    for target in targets:
        for source in sources:
            r = target.distance(source)
            target.phi -= (source.q)/r


def tree_insert_particle(particle, box, add_com = True, counter=0):
    """Inserts a particle into a box recursively into a tree structure.
    """
    # counter+=1
    # if counter == 20:
    #     print('Counter = 20!')
    #     return False
    if add_com:
        box.add_com(particle)
        # box.print_step(particle, box, add_com=1)

    if len(box.particles) < box.pmax: # if there is vacancy
        box.particles.append(particle)
        # box.print_step(particle, box, add_ptc=1)
    elif box.children: # children exists (and box full)
        # print(f'box {id(box)} full, children exist.')
        child_box = box.get_child_quadrant(particle)
        tree_insert_particle(particle, child_box, add_com=True, counter=counter)
    else: # no children (and box full)
        # print(f'box {id(box)} full, creating children.')
        particles = box.particles + [particle]
        box.quadrant_split()
        child_boxes = [box.get_child_quadrant(particle) for particle in particles]
        for particle, child_box in zip(particles, child_boxes):
            tree_insert_particle(particle, child_box, add_com=True, counter=counter)


def create_tree(root_box, particles, add_com = False):
    assert len(root_box.children)==0, \
            f"Root box already has a children. Start with a new tree."
    for p in particles:
        # print('New particle')
        tree_insert_particle(p, root_box, add_com)
        # print()


def calc_at_particle(particle, box, theta=0.5, mode=0):
    """
    mode:
        0 - fixed no. of particle per box
        1 - traditional Barnes-Hut - only max 1 particle per box
    """
    x, y = particle.coords
    if mode == 0:
        s = box.size
        r = particle.distance(other_coords = box.com_R)
        if r == 0:
            # print(id(box), id(particle), 'self')
            return
        if s/r < theta: # box far
            particle.phi -= box.com_M/r
            # print(f'Added com contribution from box {id(box)}.')
        elif box.children: # box near, children exist
            for child_box in box.children:
                if child_box.particles: # only evaluate non-empty
                    calc_at_particle(particle, child_box)
        else: # box near, no children
            sources = [p for p in box.particles if p is not particle]
            target = [particle]
            direct_source_target(sources, targets = target)
            # print(f'Added contributions from sources {[id(s) for s in sources]}.')

    elif mode == 1:
        if box.children:
            s = box.size
            r = particle.distance(other_coords = box.com_R)
            if s/r < theta: # box far
                particle.phi -= box.com_M/r
                print(f'Added com contribution from box {id(box)}.')
            else: # box near, go to child level
                for box_child in box.children:
                    calc_at_particle(particle, box_child)
        else: # box may be near or far tho, only contain <=pmax particles
            # mode 1: traditional BH:
            sources = [p for p in box.particles if p is not particle] # only one source in traditional BH
            target = [particle] # only one target
            direct_source_target(sources, targets=target)
            print(f'Added contributions from sources {[id(s) for s in sources]}.')
    

def barnes_hut_calc_phi(root_box, grid):
    grid.clear_all_phi()
    for particle in grid.particles:
        calc_at_particle(particle, root_box, mode = 1)


def replace_str(text, find, replace):
    for f, r in zip(find, replace):
        text =  text.replace(f, r)
    return text