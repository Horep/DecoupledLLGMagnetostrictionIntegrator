from ngsolve import *
from random import random
import math
import numpy as np
from datetime import timedelta
import scipy


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
    index: int = 0,
    save_step: int = 1,
):
    """
    Exports the magnetisation and displacement to a VTU file.
    A VTK file is a legacy version of this format.

    Parameters:
        displacement (ngsolve.comp.GridFunction): The displacement to be exported.
        magnetisation (ngsolve.comp.GridFunction): The magnetisation to be exported.
        mesh (ngsolve.comp.Mesh): The mesh to be exported.
        export (bool): Truth value as to whether or not the export should actually happen.
        index (int): The index, used to denote which timestep the file should be labelled with, e.g. "the_result0.vtu, the_result1.vtu".
        save_step (int): The output will only be saved at multiples of this index. Use 1 for every step, 2 for every other step, etc.
    Returns:
        None
    """
    if export is False:
        return None
    elif index % save_step == 0:
        vtk = VTKOutput(
            ma=mesh,
            coefs=[displacement, magnetisation],
            names=["displacement", "magnetisation"],
            filename=f"the_result{index}",
        )
        vtk.Do()


def calculate_KAPPA(
    density: float, exchange_length: float, gyromagnetic_ratio: float, mu_0: float
) -> float:
    """
    Computes a parameter to describe the relative strength of the elastic and magnetic contributions to the fields and energy.
    Parameters:
        density (float): Density (kg/m^3).
        exchange_length (float): Exchange length (m).
        gyromagnetic_ratio (float): rad /(s T).
        mu_0 (float): Permeability of free space (N / A^2).

    Returns:
        KAPPA (float): The elastic/magnetic relative strength parameter (dimensionless).
    """
    return density * exchange_length**2 * gyromagnetic_ratio**2 * mu_0


def stress_density_factor(KAPPA, mu_0, M_s):
    """
    Given a traction G in SI units (kg/s^2 m), divide G by the output of this function to get nondimensional g = G/output
    """
    return KAPPA * mu_0 * M_s**2


def force_density_factor(exchange_length, KAPPA, mu_0, M_s):
    """
    Given a force density F in SI units (kg/s^2 m^2), divide F by the output of this function to get nondimensional f = F/output
    """
    return stress_density_factor(KAPPA, mu_0, M_s) / exchange_length


def force_density_grav(grav_accel, density, exchange_length, KAPPA, mu_0, M_s):
    """
    Returns the gravitational force density from rho*g/(force_density_factor). For downward force, input grav_accel as negative.
    """
    return (
        density * grav_accel / force_density_factor(exchange_length, KAPPA, mu_0, M_s)
    )


def lame_parameters(E, v, KAPPA, mu_0, M_s):
    """
    Returns mu,lambda from the Young's modulus E and Poisson's ratio v
    """
    mu = E / (2 * (1 + v))
    lam = E * v / (1 + v)
    my_fac = stress_density_factor(KAPPA, mu_0, M_s)
    return mu / my_fac, lam / my_fac


def nondimensional_time(gyromagnetic, mu_0, M_s):
    """
    Returns the factor to be multiplied by time in SI units, yielding a non-dimensional time.
    """
    return gyromagnetic * mu_0 * M_s


def est_time_remaining(num_steps: int, index: int, time_dif: int):
    """
    Terrible estimate of time remaining. Multiplies last time taken by remaining steps.
    """
    return timedelta(seconds=(num_steps - index) * time_dif)


def diagonal_sparse_inv(diagonal):
    """
    Computes the inverse of a diagonal matrix, and returns a sparse representation of the inverse.
    Should be quicker than the standard inverse, but should not be used on anything that is not a diagonal matrix.
    We do not check if it is a diagonal matrix in the interest of speed.
    """
    return scipy.sparse.diags(1 / diagonal.diagonal())
