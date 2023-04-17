from ngsolve import *
from random import random
import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


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
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    for i in range(num_points):
        a = mag_gfux.vec[i]
        b = mag_gfuy.vec[i]
        c = mag_gfuz.vec[i]
        size = math.sqrt(a*a + b*b + c*c)

        mag_gfux.vec[i] = a/size
        mag_gfuy.vec[i] = b/size
        mag_gfuz.vec[i] = c/size
    mag_grid_func.components = mag_gfux, mag_gfuy, mag_gfuz
    return mag_grid_func


def build_tangent_plane_matrix_transpose(mag_grid_func):
    '''
    Returns the tangent plane transpose B^T used in the saddle point formulation for the tangent plane update.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.
    
    Returns:
        B_T (ngsolve.bla.MatrixD): 3NxN tangent plane matrix.
    '''
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    num_points = get_num_nodes(mag_grid_func)
    B_transpose = Matrix(3*num_points, num_points)
    for j in range(num_points):
        B_transpose[3*j, j] = mag_gfux.vec[j]
        B_transpose[3*j+1, j] = mag_gfuy.vec[j]
        B_transpose[3*j+2, j] = mag_gfuz.vec[j]
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
    '''
    Returns the tangent plane update v^(i) to the magnetisation such that m^(i+1) = m^(i) + v^(i).

    Parameters:
        A   (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix from the variational formulation.
        B_T (ngsolve.bla.MatrixD): The 3NxN transpose of the tangent plane matrix.
        F   (ngsolve.comp.LinearForm): The 3Nx1 force vector from the variational formulation.
    Returns:
        vlam (numpy.ndarray): The set of components to use for the update.
    '''
    rows,cols,vals = A.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols))).todense()
    B_T = B_T.NumPy()
    F = F.vec.FV().NumPy()[:]
    assert len(F) % 3 == 0, "The force vector is not a multiple of three, very bad."
    N = len(F) // 3
    stiffness_block = np.block([
    [A,                              B_T],
    [np.transpose(B_T), np.zeros((N, N))]
    ])
    force_block = np.concatenate((F, np.zeros(N)), axis=0)
    vlam = np.linalg.solve(stiffness_block, force_block)

    return vlam[0:3*N]
