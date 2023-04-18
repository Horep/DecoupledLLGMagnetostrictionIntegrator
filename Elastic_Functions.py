from ngsolve import *
from random import random
import math
import numpy as np
import General_Functions as genfunc

mu = 1
lam = 1


def strain(u):
    '''
    Returns the total strain from (grad(u) + grad(u)^T) / 2.
    '''
    return Sym(grad(u))


def strain_el(m, u):
    return strain(u) - strain_m(m, u)


def strain_m(m, u):
    return None


def stress(strain):
    '''
    Returns the stress associated with (the isotropic) Hooke's law from a given strain.
    '''
    return 2*mu*strain + lam*Trace(strain)*Id(3)


def give_random_displacement(disp_grid_func):
    '''
    Returns a random normalised magnetisation grid function.

    Parameters:
        disp_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        disp_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with randomised nodal values in [-1,1]^3.   
    '''
    num_points = genfunc.get_num_nodes(disp_grid_func)
    disp_grid_func.vec.FV().NumPy()[:] = (2*np.random.rand(3*num_points, 1) - 1).flatten()
    return disp_grid_func