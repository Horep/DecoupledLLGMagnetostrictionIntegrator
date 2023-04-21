from ngsolve import *
from random import random
import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import General_Functions as genfunc


def give_random_magnetisation(mag_grid_func):
    """
    Returns a random normalised magnetisation grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with randomised nodal values in [-1,1]^3 and length 1 at each node.
    """
    num_points = genfunc.get_num_nodes(mag_grid_func)
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    for i in range(num_points):
        a, b, c = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
        size = math.sqrt(a * a + b * b + c * c)
        try:
            a, b, c = a / size, b / size, c / size
        except (
            ZeroDivisionError
        ):  # it is extremely unlikely, but possible, to have a=b=c=0. If this happens, use (1,0,0)
            a, b, c = 1, 0, 0
        mag_gfux.vec[i] = a
        mag_gfuy.vec[i] = b
        mag_gfuz.vec[i] = c
    return mag_grid_func


def nodal_projection(mag_grid_func):
    """
    Returns a grid function with all nodal values projected onto the unit sphere. Every node z will satisfy |m(z)|=1.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with length 1 at each node.
    """
    num_points = genfunc.get_num_nodes(mag_grid_func)
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    for i in range(num_points):
        a = mag_gfux.vec[i]
        b = mag_gfuy.vec[i]
        c = mag_gfuz.vec[i]
        size = math.sqrt(a * a + b * b + c * c)

        mag_gfux.vec[i] = a / size
        mag_gfuy.vec[i] = b / size
        mag_gfuz.vec[i] = c / size
    return mag_grid_func


def build_tangent_plane_matrix(mag_grid_func):
    """
    Returns the tangent plane matrix used in the saddle point formulation for the tangent plane update.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        B (numpy.ndarray): Nx3N tangent plane matrix.
    """
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components

    #  Cast the components of mag_grid_func to flat vector numpy arrays, and then assemble B as a block matrix from diagonal matrices of m1,m2,m3.
    m1 = mag_gfux.vec.FV().NumPy()[:]
    m2 = mag_gfuy.vec.FV().NumPy()[:]
    m3 = mag_gfuz.vec.FV().NumPy()[:]
    B = np.block([[np.diag(m1, k=0), np.diag(m2, k=0), np.diag(m3, k=0)]])
    return B


def give_magnetisation_update(A, B, F):
    """
    Returns the tangent plane update v^(i) to the magnetisation such that m^(i+1) = m^(i) + v^(i).

    Parameters:
        A (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix from the variational formulation.
        B (ngsolve.bla.MatrixD): The 3NxN transpose of the tangent plane matrix.
        F (ngsolve.comp.LinearForm): The 3Nx1 assembled force vector from the variational formulation.

    Returns:
        vlam (numpy.ndarray): The set of components to use for the update.
    """
    rows, cols, vals = A.mat.COO()
    A = sp.csr_matrix((vals, (rows, cols))).todense()
    A = np.array(A)
    F = F.vec.FV().NumPy()[:]
    assert len(F) % 3 == 0, "The force vector is not a multiple of three, very bad."
    N = len(F) // 3
    stiffness_block = np.block([[A, np.transpose(B)], [B, np.zeros((N, N))]])
    force_block = np.concatenate((F, np.zeros(N)), axis=0)
    vlam = np.linalg.solve(stiffness_block, force_block)
    v = vlam[0 : 3 * N]
    residual = np.linalg.norm(
        B.dot(v), 1
    )  # in theory, the update should satisfy |Bv| = 0.
    if residual > 1e-10:
        print(
            "WARNING: ||Bv|| = {residual} > 1e-12. Tangent plane matrix B or update v may not be correctly calculated."
        )

    return v


def build_strain_m(fes_eps_m, mag_grid_func):
    """
    Builds a matrix of the form
        m1*m1-1/3 m1*m2     m1*m3\n
        m2*m1     m2*m2-1/3 m2*m3\n
        m3*m1     m3*m2     m3*m3-1/3
    from an input magnetisation of the form (m1,m2,m3)
    """
    numpoints = genfunc.get_num_nodes(mag_grid_func)
    m1, m2, m3 = mag_grid_func.components
    mymatrix = GridFunction(fes_eps_m)
    M11, M12, M13,\
    M21, M22, M23,\
    M31, M32, M33 = mymatrix.components
    # this is a bad implementation, should be broadcast using numpy arrays, and use symmetry of the matrix. I have avoided this as it makes the code less readable
    # the symmetry can be implemented in the finite element space fes_eps_m directly with the flag symmetry=True

    for i in range(numpoints):
        M11.vec[i] = m1.vec[i] * m1.vec[i] - 1 / 3
        M22.vec[i] = m2.vec[i] * m2.vec[i] - 1 / 3
        M33.vec[i] = m3.vec[i] * m3.vec[i] - 1 / 3
        M12.vec[i] = m1.vec[i] * m2.vec[i]
        M13.vec[i] = m1.vec[i] * m3.vec[i]
        M23.vec[i] = m2.vec[i] * m3.vec[i]
        M21.vec[i] = M12.vec[i]
        M31.vec[i] = M13.vec[i]
        M32.vec[i] = M23.vec[i]

    return mymatrix


def update_magnetisation(mag_grid_func, v):
    """
    Updates a magnetisation vector with the new values.
    Parameters:
        mag_grid_func (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix from the variational formulation.
        v                         (numpy.ndarray): The 3NxN transpose of the tangent plane matrix.

    Returns:
        mag_grid_func (numpy.ndarray): The set of components to use for the update.

    """

    N = genfunc.get_num_nodes(mag_grid_func)
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    mag_gfux.vec.FV().NumPy()[:] += v[0:N]
    mag_gfuy.vec.FV().NumPy()[:] += v[N : 2 * N]
    mag_gfuz.vec.FV().NumPy()[:] += v[2 * N : 3 * N]

    return mag_grid_func
