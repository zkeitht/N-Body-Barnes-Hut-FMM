import numpy as np
import matplotlib.pyplot as plt
import time
import random

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

class Grid():
    """ Creates grid that acts as a "container" of the particles
    """
    def __init__(self, size = 10):
        self.particles = []
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.size = size
        # plt.close()

    def add_particle(self, particle):
        self.particles.append(particle)
    
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
        plt.show()
    
    def direct_sum(self):
        """Calculate the potential for each particle using direct summation method.
        Potential := Σ-(q/r), where q is the charge of the source particles and r is the distance between the particle and the source particles. 
        The potential of each of the particles is stored in the phi attribute of the particles.
        """
        self.clear_all_phi()
        for i, target in enumerate(self.particles):
            for source in (self.particles[:i] + self.particles[i+1:]):
                r = target.distance(source)
                target.phi -= (source.q)/r
    
    def get_all_coords(self):
        return [p.coords for p in self.particles]
    
    def get_all_phi(self):
        return [p.phi for p in self.particles]

    def clear_all_phi(self):
        for p in self.particles:
            p.phi = 0

class Particle():
    """ Creates a particle with a "charge" (e.g. mass, electric charge) within a given grid.
    Its initial coordinates is randomised within the grid.
    grid: the grid to which the particle belongs
    q: charge of the particle
    coords: spatial coordinates of the particle
    phi: potential, initialised to 0
    """
    def __init__(self, grid, q, coords = (), colour = "c"):
        self.grid = grid
        self.grid.add_particle(self) # The particle is automatically added to the grid upon creation.
        self.q = q
        self.phi = 0 # Initialise potential to 0
        # initialise with randomised coordinates if they are not provided.
        if coords:
            assert len(coords) == 2, "the length of the coordinates should be 2."
            self.coords = np.array(coords)
        else:
            size = self.grid.size
            rand_x = size * random.random()
            rand_y = size * random.random()
        self.coords = np.array((rand_x, rand_y))
        self.colour = colour
        
    
    def distance(self, others):
        """Calculates distance between the particle and another fixed point"""
        return np.linalg.norm(self.coords-others.coords)
    
theGrid = Grid()
# Fixing random state for reproducibility
random.seed(4)
particles = [Particle(theGrid, i) for i in range(1,6)]
theGrid.draw()
# plt.show()
anotherp = Particle(theGrid,3)
theGrid.draw()
# plt.show()
