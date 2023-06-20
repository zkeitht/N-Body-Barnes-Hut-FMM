# nbody_objects
import numpy as np
import matplotlib.pyplot as plt

class Grid():
    """Creates grid that acts as a "container" of the particles and a platform to 
    display the particle distribution. 
    Note that the grid can only contain particles having coordinates > 0. 
    Otherwise, shift all the particles until they are all > 0.
    """
    def __init__(self, size=10):
        self.particles = []
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.size = size
        plt.close()

    def add_particle(self, particle):
        """Adds particle to grid automatically upon particle initialisation. 
        (see __init__ of Particle Class)
        Makes sure that there is no negative-valued coordinate and that
        the coordinates are within the (square) size of the grid.
        """
        if all(c>0 for c in particle.coords): 
            self.particles.append(particle)
            max_coord = max([c for c in particle.coords])
            if max_coord > self.size:
                self.size = max_coord
                print(f"Grid size adjusted to {max_coord}")
        else:
            raise Exception("Grid can only take particles with positive-valued"
            "coordinates. Try shifting the origin of the particle(s).")
    
    def create_particles(self, N, all_coords=None, all_q=None):
        if all_q is None:
            all_q = np.ones(N)
            print('All charges initialised to 1.')

        if all_coords is None:
            particles = [Particle(grid = self, q=q) for q in all_q]
        else:
            particles = [Particle(grid = self, q=q, coords=coords) for q, coords in zip(all_q, all_coords)]
        return particles

    def draw(self, particles=True, square_level=None):
        if particles:
            self.ax.clear()
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)      
            self.ax.set_xlabel(r'$x$')
            self.ax.set_ylabel(r'$y$')

            all_x = [particle.coords[0] for particle in self.particles]
            all_y = [particle.coords[1] for particle in self.particles]
            all_colours = [particle.colour for particle in self.particles]
            self.ax.scatter(all_x, all_y, color = all_colours)
            # plt.show(self.fig)
            return self.fig
        
        if square_level is not None:
            l=square_level
            self.ax.clear()
            size = self.size
            for i in range(1,4**l):
                self.ax.plot([0,size], [size/(2**l)*i]*2, color = 'w')
                self.ax.plot([size/(2**l)*i]*2, [0,size], color = 'w')
            self.ax.set_xlim(0, size)
            self.ax.set_ylim(0, size)      
            self.ax.set_xticks([])      
            self.ax.set_yticks([])      
            return self.fig

    def direct_sum(self):
        """
        Calculates the potential for each particle using direct summation method.
        Potential := Σ-(q/r), where q is the charge of the source particles and 
        r is the distance between the particle and the source particles. 
        The potential of each of the particles is stored in the phi attribute of 
        the particles.
        """
        self.clear_all_phi()
        for i, target in enumerate(self.particles):
            for source in (self.particles[:i] + self.particles[i+1:]):
                r = target.distance(source)
                target.phi -= (source.q)/r
    
    def direct_source_target(self, sources, targets):
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
        self.clear_all_phi()
        for target in targets:
            for source in sources:
                r = target.distance(source)
                target.phi -= (source.q)/r

    def get_all_coords(self):
        return np.array([p.coords for p in self.particles])
    
    def get_all_phi(self):
        return [p.phi for p in self.particles]

    def clear_all_phi(self):
        for p in self.particles:
            p.phi = 0
    
    def S2M_source_coeffs(self, sources, center, weird=1):
        """Takes all particles in a cell (as sources), and add the SOURCE coefficients
        of the multipole expansions terms (centered at the cell) sum over all source particles. 
        Intermediate calculation to S2M."""
        delta = np.array(center) - np.array([source.coords for source in sources])
        dx, dy = delta.T # e.g. dx contains source-center distance of every source particle
        source_q = np.array([source.q for source in sources])
        # the code reference having an extra factor of 1/2 for the last term (weird = 1/2)
        # this is just to show that I did check everything in the reference and build them myself
        source_coeffs = [np.ones(len(sources)), dx, dy, (dx**2)/2, (dy**2)/2, dx*dy*weird]
        # source_coeff = np.matmul(np.diag(source_q), source_coeff)
        source_coeffs = source_q * source_coeffs
        return np.sum(source_coeffs,axis=1)

    def S2M_potential(self, sources, targets, center, weird=1):
        """Calculate the potential induced by a group of *source* particles, evaluated at each of the *target* particles
        Uses the intermediate S2M_source_sum and combines it with the TARGET coefficients."""
        S2M_source_coeffs = self.S2M_source_coeffs(sources, center, weird)
        delta = np.array([target.coords for target in targets]) - np.array(center)
        dx,dy = delta.T
        r = np.sqrt(dx**2 + dy**2)
        r3 = r**3
        r5 = r**5
        target_coeffs = [-1/r, dx/r3, dy/r3, 1/r3-(dx**2)/r5, 1/r3-(dy**2)/r5, -3*dx*dy/r5]
        potential = np.dot(S2M_source_coeffs, target_coeffs)
        return potential


class Particle():
    """Creates a particle with a "charge" (e.g. mass, electric charge) within a given grid.
    Its initial coordinates is randomised within the grid.
    grid: the grid to which the particle belongs
    q: charge of the particle
    coords: spatial coordinates of the particle
    phi: potential, initialised to 0
    """
    def __init__(self, grid, q, coords=None, colour="c"):
        self.grid = grid
        self.q = q
        self.phi = 0 # Initialise potential to 0
        # initialise with randomised coordinates if they are not provided.
        if coords is not None:
            assert len(coords) == 2, "the length of the coordinates should be 2."
            self.coords = np.array(coords)
        else:
            size = self.grid.size
            rand_x = size * np.random.random()
            rand_y = size * np.random.random()
            self.coords = np.array((rand_x, rand_y))
        self.grid.add_particle(self) # The particle is automatically added to the grid upon creation.
        self.colour = colour
        
    
    def distance(self, others=None, other_coords=None):
        """Calculates distance between the particle and another object.
        The other object can either be another object with a 'coords' attribute
        or an array of coords.
        """
        assert None in (others, other_coords), f"Input ONLY ONE of 'other' or 'other_coords'. "
        if others:
            other_coords = others.coords
        return np.linalg.norm(self.coords-other_coords)


class Box():
    """Elements that constitute a tree.

    Arguments
    ---------
    coords: center coordinates of the box.
    com: ΣmR/Σm (summed over all particles) m can be charge or mass
    pmax: maximum number of particles allowed in the box.
    """
    def __init__(self, coords, size, pmax=1, parent=None, grid=None, S2M_source_coeffs=np.zeros(6)):
        self.coords = np.array(coords)
        self.size = size
        self.pmax = pmax
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
        self.children = []
        self.grid = grid
        if grid is None:
            self.grid = self.parent.grid
        self.S2M_source_coeffs = S2M_source_coeffs
        self.particles = []
        self.com_M = 0
        self.com_R = np.array(self.coords)

    def quadrant_split(self):
        """Creates quadrant (child) boxes centered at parent box.
         Q2 | Q3
        ----+----
         Q0 | Q1
        """
        assert len(self.children)==0, f"Current box already has children."
        x, y = self.coords
        size = self.size/2
        q_i = np.arange(4)
        q_xs = (q_i%2 * size) - size/2 + x 
        q_ys = (q_i//2 * size) - size/2 + y
        q_children = [Box((qx, qy), size, pmax=self.pmax, parent=self) for qx, qy in zip(q_xs, q_ys)]
        # print(q_xs, q_ys, q_children)
        return q_children
    
    def get_child_quadrant(self, particle):
        """ Locates the child quadrant a particle belongs to and returns it.
        quadrant_i: 0 or 1 or 2 or 3
         Q2 | Q3
        ----+----
         Q0 | Q1
        """
        assert self.children, f"Current box has no children. Use quadrant_split() to obtain children first."
        x, y = particle.coords
        x_c, y_c = self.coords
        quadrant_i = [[0,2],[1,3]][int(x > x_c)][int(y > y_c)] # 0|1|2|3
        return self.children[quadrant_i]

    def add_com(self, particle):
        # self.particles.append(particle) 
        # - I guess there is no point in adding the child itself as 
        # a) the com will be added
        # b) individual child are only needed at the external box level
        M, R = self.com_M, self.com_R
        m, r = particle.q, particle.coords
        self.com_R = (M*R + m*r)/(M+m)
        self.com_M += m
    
    def print_step(self, particle, box, add_com=0, add_ptc=0):
        if add_com:
            print(f'COM contribution of particle {id(particle)} added to box {id(box)}.')
        elif add_ptc:
            print(f'Particle {id(particle)} added to box {id(box)}.')
