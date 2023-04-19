from ngsolve import *
from random import random
import math
import numpy as np


def get_num_nodes(grid_func):
    '''
    Returns the number of nodes within the grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        number of nodes (Int): Number of nodes in grid function.   
    '''
    assert len(grid_func.vec) % 3 == 0, "The vector data is not a multiple of three. Wrong dimension?"
    return len(grid_func.vec) // 3
