from ngsolve import *
from random import random
import math
import numpy as np


def get_num_nodes(grid_func: GridFunction) -> int:
    """
    Returns the number of nodes within the grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        number of nodes (int): Number of nodes in grid function.
    """
    assert (
        len(grid_func.vec) % 3 == 0
    ), "The vector data is not a multiple of three. Wrong dimension?"
    return len(grid_func.vec) // 3


def ceiling_division(a: float, b: float) -> int:
    """
    Returns the closest integer greater than a/b

    Parameters:
        a (float): The numerator.
        b (float): The denominator.

    Returns:
        ceil(a/b) (int): Number of nodes in grid function.

    Examples:
        ceiling_division(1.0, 2.0) = 1
        ceiling_division(5.0, 4.0) = 2
    """
    return int(-(a // -b))


def export_to_vtk_file(
    displacement: GridFunction,
    magnetisation: GridFunction,
    mesh: Mesh,
    export: bool = False,
):
    if export is False:
        return None
    vtk = VTKOutput(
        ma=mesh,
        coefs=[displacement, magnetisation],
        names=["displacement", "magnetisation"],
        filename="the_result",
    )
    vtk.Do()
