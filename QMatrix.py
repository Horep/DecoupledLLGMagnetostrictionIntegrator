from ngsolve import *
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def qmatrix(mag_grid_func: GridFunction):
    m1, m2, m3 = mag_grid_func.components
    m1, m2, m3 = m1.vec.FV().NumPy(), m2.vec.FV().NumPy(), m3.vec.FV().NumPy()
    column_m = np.column_stack((m1, m2, m3))

    out_array = q_basis(column_m)
    return out_array


def q_basis(m: np.ndarray):
    """
    Given a magnetisation node m = [[m1(z_1), m2(z_1), m3(z_1)],
                                    [m1(z_2), m2(z_2), m3(z_2)],
                                    ...,
                                    [m1(z_N), m2(z_N), m3(z_N)]], 
    returns a suitable basis based upon the smallest in magnitude of m1, m2, m3
    """
    N = len(m)
    index = np.argmin(np.abs(m), axis=1)  # yields which case, 0,1,2 should be used

    u,v,w = m[0]
    L, M = basis_choice(u, v, w, index[0])
    out_array= np.array(L)
    out_array = np.vstack((out_array, M))
    for i in range(1, N):  # miss out first as we already do this above to initialise arrays
        u, v, w = m[i]
        L, M = basis_choice(u, v, w, index[i])
        out_array = np.vstack((out_array, L))
        out_array = np.vstack((out_array, M))

    return out_array


def basis_choice(u: float, v: float, w: float, index: int):
    """
    We use notation from Ramage 2013 during this section. Given an input magnetisation vector n=[u,v,w]^T, return two vectors L,M such that (n,L,M) is an orthonormal basis.
    """
    if index == 0:
        square = v*v + w*w
        my_norm = np.sqrt(square)
        L = np.array([0, -w, v])/my_norm
        M = np.array([square, -u*v, -u*w])/my_norm
        return L, M
    elif index == 1:
        square = u*u + w*w
        my_norm = np.sqrt(square)
        L = np.array([w, 0, -u])/my_norm
        M = np.array([-v*u, square, -v*w])/my_norm
        return L, M
    elif index == 2:
        square = u*u + v*v
        my_norm = np.sqrt(square)
        L = np.array([-v, u, 0])/my_norm
        M = np.array([-w*u, -w*v, square])/my_norm
        return L, M


def Q_block(my_vec, N):
    cols = np.repeat(np.arange(N+1), 2)  # looks like [0,0,1,1,...,N,N]
    rows = np.arange(2*N+1)  # looks like [0,1,2,3,...,2N]
    
    return None