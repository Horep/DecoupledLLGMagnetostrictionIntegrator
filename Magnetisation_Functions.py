from ngsolve import *
from random import random
import math
import numpy as np

def give_random_magnetisation(mag_grid_func):
    '''
    Returns a random normalised magnetisation grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with randomised nodal values and length 1 at each node.   
    '''
    num_points = get_num_nodes(mag_grid_func)
    for i in range(num_points):
        a, b, c = 2*random()-1, 2*random()-1, 2*random()-1
        size = math.sqrt(a*a + b*b + c*c)
        try:
            a, b, c = a/size, b/size, c/size
        except ZeroDivisionError:  # it is extremely unlikely, but possible, to have a=b=c=0. If this happens, use (1,0,0)
            a, b, c = 1, 0, 0
        mag_grid_func.vec[3*i] = a
        mag_grid_func.vec[3*i + 1] = b
        mag_grid_func.vec[3*i + 2] = c
    
    return mag_grid_func


def nodal_projection(mag_grid_func):
    '''
    Returns a grid function with all nodal values projected onto the unit sphere.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with length 1 at each node.   
    '''
    num_points = get_num_nodes(mag_grid_func)
    for i in range(num_points):
        a = mag_grid_func.vec[3*i]
        b = mag_grid_func.vec[3*i + 1]
        c = mag_grid_func.vec[3*i + 2]
        size = math.sqrt(a*a + b*b + c*c)

        mag_grid_func.vec[3*i] = a/size
        mag_grid_func.vec[3*i + 1] = b/size
        mag_grid_func.vec[3*i + 2] = c/size
    
    return mag_grid_func


def build_tangent_plane_matrix_transpose(mag_grid_func):
    num_points = get_num_nodes(mag_grid_func)
    B_transpose = Matrix(3*num_points, num_points)
    for j in range(num_points):
        B_transpose[3*j, j] = mag_grid_func.vec[3*j]
        B_transpose[3*j+1, j] = mag_grid_func.vec[3*j+1]
        B_transpose[3*j+2, j] = mag_grid_func.vec[3*j+2]
    return B_transpose


def get_num_nodes(mag_grid_func):
    '''
    Returns the number of nodes within the grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        number of nodes (Int): Number of nodes in grid function.   
    '''
    assert len(mag_grid_func.vec) % 3 == 0, "The vector data is not a multiple of three. Wrong dimension?"
    return len(mag_grid_func.vec) // 3


def give_magnetisation_update(A, B_T, F):
    A = A.NumPy()
    B_T = B_T.NumPy()
    F = F.NumPy()
    print(F.ndim)
