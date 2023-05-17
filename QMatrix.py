from ngsolve import *
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def QMatrix(mag_grid_func: GridFunction):
    m1, m2, m3 = mag_grid_func.components
    m1, m2, m3 = m1.vec.FV().NumPy(), m2.vec.FV().NumPy(), m3.vec.FV().NumPy()
    m1 ,m2, m3 = m1.T, m2.T, m3.T
    return None


def get_Q_basis(m: np.ndarray):
    """
    Given a magnetisation node m = [[m1(z_1), m2(z_1), m3(z_1)],
                                    [m1(z_2), m2(z_2), m3(z_2)],
                                    ...,
                                    [m1(z_N), m2(z_N), m3(z_N)]], 
    returns a suitable basis based upon the smallest in magnitude of m1, m2, m3
    """
    index = np.argmin(np.abs(m))


def reorder_magnetisation(m1, m2, m3):
    return 