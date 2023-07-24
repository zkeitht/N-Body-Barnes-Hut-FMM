# FMM core calculations
import math
import numpy as np

def sum_4_7(a_k, source_centre, target_coords):
    """
    for Broadcasting purposes:
    k-varying terms: [None, :] (power of multipole)
    i-varying terms: [:, None] (particle index)
    sum: 
        axis = 0: sum over particles
        axis = 1: sum over powers
    a_0: number
    a_k: array
    sum returns potential evaluated at target coordinates
    """
    
    p = len(a_k)-1
    target_coords = np.array(target_coords)
    k_k = np.arange(1,p+1) #[None, :]
    z_t = target_coords-source_centre.coords
    power_terms = (1/(z_t)[:, None] ** k_k)
    return a_k[0]*np.log(z_t) + np.matmul(power_terms, a_k[1:])

def S2M_coeff(particles, source_centre, p=6):
    """S2M coefficients (4.7) i.e. a_k given centre
    for Broadcasting purposes:
    k-varying terms: [None, :] (power of multipole)
    i-varying terms: [:, None] (particle index)
    sum: 
        axis = 0: sum over particles
        axis = 1: sum over powers
    a_0: number
    a_k: array
    """
    if particles:
        q_i = np.array([particle.q for particle in particles])[:, None]
        z_i = np.array([(particle.coords - source_centre.coords) for particle in particles])[:, None]
        k_k = np.arange(1,p+1) #[None, :]
        
        a_0 = sum(q_i)[0]
        a_k = np.sum(-q_i*(z_i**k_k)/k_k, axis = 0)
        a_k = np.insert(a_k, 0, a_0)
        return a_k
    return np.zeros(p+1)

def M2M_translation(a_k, source_centre_old, source_centre_new):
    """M2M translation (4.15) (old source centre to new source centre)
    a_0, a_k --> a_0, b_l
    a_0: number
    a_k, b_l: array
    """
    # a_k_all = np.insert(a_k, 0, a_0)
    z_0 = source_centre_old.coords - source_centre_new.coords
    b_l = np.array([sum([a_k[k] * z_0**(l-k) * math.comb(l-1, k-1) 
                         for k in range(1, l+1)]) 
                         for l in range(1, len(a_k))])
    l_l = np.arange(1, len(a_k))
    b_l = b_l - a_k[0]*(z_0**l_l)/l_l
    b_l = np.insert(b_l, 0, a_k[0])
    return b_l

def M2L_coeff(a_k, source_centre, target_centre):
    """takes "S2M coefficients", returns b_l (M2L coefficients) (4.18) 
    not a_k from source"""
    p = len(a_k)-1
    # a_k_all = np.insert(a_k, 0, a_0)
    z_0 = (source_centre.coords - target_centre.coords)
    k_k = np.arange(1,p+1)
    b_0 = a_k[0] * np.log(-z_0) + np.sum(a_k[1:] * (-1/z_0)**k_k)
    b_l = np.array(
        [- a_k[0]/l/(z_0**l)
         + (sum([a_k[k] * (-1/z_0)**k * math.comb(l+k-1, k-1) for k in range(1, p+1)])/z_0**l)
         for l in range(1, p+1)]
        )
    b_l = np.insert(b_l, 0, b_0, axis=0)
    return b_l

def sum_local(b_l, target_centre, target_coords):
    """(4.17) Although sum_4_7 provides the same multi-target evaluation, it requires
    evaluation of the a_0, a_k coeffs for each of the targets.
    sum_4_17 only uses the target info at the end of the"""
    p = len(b_l)-1
    target_coords = np.array(target_coords)
    l_l = np.arange(0,p+1) #[None, :]
    z_t = target_coords - target_centre.coords
    power_terms = (z_t)[:, None] ** l_l
    return np.matmul(power_terms, b_l)

def L2L_translation(a_k, target_centre_old, target_centre_new):
    """L2L translation (4.22) (old target centre to new target centre)
    a_k (old) --> b_k (new)
    a_k: array
    this is an exact expression hence shifting to new origin wouldn't affect the results
    """
    p = len(a_k)-1
    b_k = np.copy(a_k)
    z_0 = target_centre_old.coords - target_centre_new.coords
    for j in range(0, p):
        for k in range(p-j-1, p):
            b_k[k] = b_k[k] - z_0*b_k[k+1]
    return b_k