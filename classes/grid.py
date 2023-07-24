import numpy as np
import matplotlib.pyplot as plt

# Grid, GridComplex, Point, and Particle classes
# the GridComplex class inherits the Grid class
# the Particle class inherits the Point class
from .box import Box
# from .point import Point, Particle
# from .point import Particle


class Grid():
    """Creates grid that acts as a "container" of the particles and a platform to 
    display the particle distribution. 
    Note that the grid can only contain particles having coordinates > 0. 
    Otherwise, shift all the particles until they are all > 0.
    """
    def __init__(self, size=16):
        self.size = size
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.particles = []
        self.points = []
        self.rootbox = Box(coords=(size/2, size/2), size=size, ptcmax=1, grid=self)
        plt.close()

    def add_point(self, point):
        """Adds point/particle to grid automatically upon point/
        particle initialisation. 
        (see __init__ of Point Class)
        Makes sure that there is no negative-valued coordinate and that
        the coordinates are within the (square) size of the grid.
        """
        if type(point)==Particle:
            self.particles.append(point)
            if point.coords is not None:
                assert all(c>0 for c in point.coords), \
                    """Grid can only take particles with positive-valued
                    coordinates. Try shifting the origin of the particle(s)."""
                max_coord = max([c for c in point.coords])
                if max_coord > self.size:
                    self.size = max_coord
                    print(f"Grid size adjusted to {max_coord}")
        elif type(point)==Point:
            self.points.append(point)

    def create_particles(self, N, all_coords=None, all_q=None):
        if all_q is None:
            all_q = np.ones(N)*10
            print('All charges initialised to 10.')

        if all_coords is None:
            particles = [Particle(grid=self, q=q) for q in all_q]
        else:
            particles = [Particle(grid=self, coords=coords, q=q) for q, coords in zip(all_q, all_coords)]
        
        return particles

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)      
        self.ax.set_xlabel(r'$x$')
        self.ax.set_ylabel(r'$y$')

        all_x = [particle.coords[0] for particle in self.particles]
        all_y = [particle.coords[1] for particle in self.particles]
        all_colours = [particle.colour for particle in self.particles]
        self.ax.scatter(all_x, all_y, color = all_colours)
        return self.fig

    def draw_squares(self, lvls, save=0):
        """Draws square grid at given level.
        Call grid.draw() first to include particles.
        """
        size = self.size
        fig, ax = self.fig, self.ax
        lw = 2
        for lvl in range(lvls+1):
            for i in range(1,4**lvl):
                ax.plot([0,size], [size/(2**lvl)*i]*2, color = 'k', linewidth=lw)
                ax.plot([size/(2**lvl)*i]*2, [0,size], color = 'k', linewidth=lw)
            lw *= 0.6
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)      
        # ax.set_xticks([])      
        # ax.set_yticks([])
        if save:
            name = input('') + f' Grid level {lvls}' + '.jpg'
            fig.savefig(name, dpi=500)
        return fig
    
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
    

class GridComplex(Grid):
    """Creates complex grid that acts as a "container" of the particles and a platform to 
    display the particle distribution.
    """
    def __init__(self, size=16):
        super().__init__(size)

    def add_point(self, point):
        """Adds point/particle to grid automatically upon point/
        particle initialisation. 
        (see __init__ of Point Class)
        
        Makes sure that there is no negative-valued coordinate 
        (x>0, y>0) for x+iy and that the coordinates are within the 
        (square) size of the grid.
        """
        if type(point)==Particle:
            self.particles.append(point)
        elif type(point)==Point:
            self.points.append(point)
        if point.coords is not None:
            r, i = point.coords.real, point.coords.imag
            if r<self.size and i<self.size:
                pass
            else:
                print(f'{type(point).__name__} {id(point)} out of grid')   
    
    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)      
        self.ax.set_xlabel(r'$x$')
        self.ax.set_ylabel(r'$y$')
        for ps in [self.particles, self.points]:
            if ps:
                r_i_c = [(p.coords.real, p.coords.imag, p.colour) for p in ps]
                all_x, all_y, all_colours = zip(*r_i_c)
                self.ax.scatter(all_x, all_y, color = all_colours)
        return self.fig
    

class Point():
    """Point object.

    Attributes
    ----------
    grid: the grid to which the point belongs
    q: charge of the particle
    coords: spatial coordinates of the particle
    phi: potential, initialised to 0
    """
    def __init__(self, grid, coords, colour = 'r'):
        self.grid = grid
        self.coords = coords
        self.grid.add_point(self) # The point is automatically added to the grid upon creation.
        self.colour = colour

    def distance(self, target_coords):
        """Returns (real) distance between the point and a target.
        The target can either be another object with a coords
        attribute or an array of coords.
        """    
        if hasattr(target_coords, 'coords'):
            target_coords = target_coords.coords
        r = target_coords-self.coords
        return np.linalg.norm(r)
    
    def coords_to_real(self):
        return self.coords.real, self.coords.imag


class Particle(Point):
    """Creates a particle with a "charge" (e.g. mass, electric charge) within a given grid.
    Its initial coordinates is randomised within the grid.
    grid: the grid to which the particle belongs
    q: charge of the particle
    coords: spatial coordinates of the particle
    phi: potential, initialised to 0
    """
    def __init__(self, grid, coords=None, q=10, colour="c"):
        super().__init__(grid, coords, colour)
        self.q = q
        self.phi = 0 # Initialise potential to 0
        # initialise with randomised coordinates if they are not provided.
        if coords is None:
            size = self.grid.size
            rand_x = size * np.random.random()
            rand_y = size * np.random.random()
            if type(self.grid) == GridComplex:
                self.coords = rand_x + 1j*rand_y
            else:
                self.coords = np.array((rand_x, rand_y))
        elif type(self.grid) == GridComplex:
            self.coords = coords
        else:
            try:
                _ = iter(coords)
                self.coords = np.array(coords)
            except TypeError:
                print('coords given not iterable')