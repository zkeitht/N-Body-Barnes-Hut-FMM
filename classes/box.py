import numpy as np

class Box():
    """Elements that constitute a tree.

    Arguments
    ---------
    coords: center coordinates of the box.
    com: ΣmR/Σm (summed over all particles) m can be charge or mass
    m: maximum number of particles allowed in the box.
    """
    def __init__(self, coords, size=16, m=1, parent=None, grid=None):
        self.coords = np.array(coords)
        self.size = size
        self.m = m
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
        self.children = []
        self.grid = grid
        if grid is None:
            self.grid = self.parent.grid
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
        q_children = [Box((qx, qy), size, m=self.m, parent=self) for qx, qy in zip(q_xs, q_ys)]
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


class BoxComplex():
    """Elements that constitute a tree.

    Arguments
    ---------
    coords: center coordinates of the box.
    """
    def __init__(self, coords, size=16, m=10, parent=None, grid=None, p=6):
        self.coords = coords
        self.size = size
        self.m = m
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
        self.children = []
        self.grid = grid
        if grid is None and self.parent is not None:
            self.grid = self.parent.grid
        self.p = p
        self.S2M_coeffs = np.zeros(p+1)
        self.M2L_coeffs = np.zeros(p+1)
        self.particles = []

    def quadrant_split(self):
        """Creates quadrant (child) boxes centered at parent box.
         Q2 | Q3
        ----+----
         Q0 | Q1
        """
        assert len(self.children)==0, f"Current box already has children."
        x, y = self.coords.real, self.coords.imag
        size = self.size/2
        q_i = np.arange(4)
        q_xs = (q_i%2 * size) - size/2 + x 
        q_ys = (q_i//2 * size) - size/2 + y
        q_children = [BoxComplex(qx +1j*qy, size, m=self.m, parent=self, p=self.p) for qx, qy in zip(q_xs, q_ys)]
        return q_children
    
    def coords_to_real(self):
        return self.coords.real, self.coords.imag