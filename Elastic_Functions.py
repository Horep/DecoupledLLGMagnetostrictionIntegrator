from ngsolve import *
from random import random
import math
import numpy as np
import itertools
import General_Functions as genfunc
import Magnetisation_Functions as magfunc

mu = 1
lam = 1


def strain(u):
    """
    Returns the total strain from (Grad(u) + Grad(u)^T) / 2.
    """
    return Sym(Grad(u))


def strain_el(m, u):
    """
    Returns the elastic strain, the difference of the total strain and magnetostrain.
    """
    return strain(u) - magfunc.build_strain_m(m, u, 1.0)


def stress(strain):
    """
    Returns the stress associated with (the isotropic) Hooke's law from a given strain.
    """
    return 2 * mu * strain + lam * Trace(strain) * Id(3)


def give_random_displacement(disp_grid_func):
    """
    Returns a displacement grid function with random entries in [-0.1, 0.1].

    Parameters:
        disp_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        disp_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with randomised nodal values in [-0.1, 0.1]^3.
    """
    num_points = genfunc.get_num_nodes(disp_grid_func)
    disp_grid_func.vec.FV().NumPy()[:] = np.random.uniform(
        -0.1, 0.1, 3 * num_points
    ).flatten()  # 3N uniformly distributed points in [-0.1, 0.1]
    return disp_grid_func


def update_displacement(
    fes_disp, disp_gfu, disp_gfu_prev, fes_eps_m, mag_gfu, f_body, g_surface, K
):
    """

    Updates a displacement vector with the new values.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): VectorH1 finite element space.
        disp_gfu (ngsolve.comp.GridFunction): The current displacement.
        disp_gfu_prev (ngsolve.comp.GridFunction): The displacement from the previous iteration.
        f_body (ngsolve.fem.CoefficientFunction): The body force.
        g_surface (ngsolve.fem.CoefficientFunction): The traction.

    Returns:
        disp_gfu (ngsolve.comp.GridFunction): The new updated displacement at the next time step.
    """
    # Test functions
    u = fes_disp.TrialFunction()
    psi = fes_disp.TestFunction()
    # Building the linear system for the displacement
    a_disp = BilinearForm(fes_disp)
    a_disp += InnerProduct(u, psi) * dx  # <u^(i+1), ψ>
    a_disp += (
        K * K * InnerProduct(stress(strain(u)), strain(psi)) * dx
    )  # k^2<Cε(u), ε(ψ)>

    f_disp = LinearForm(fes_disp)
    f_disp += (
        InnerProduct(
            stress(
                magfunc.build_strain_m(fes_eps_m, magfunc.nodal_projection(mag_gfu))
            ),
            strain(psi),
        )
        * dx
    )  # <Cε_m(Π m),ε(ψ)>
    f_disp += (
        InnerProduct(disp_gfu - disp_gfu_prev, psi) * dx
    )  # k<d_t u^i, ψ> = <u^i - u^(i-1), ψ>
    f_disp += InnerProduct(disp_gfu, psi) * dx  # <u^i, ψ>
    f_disp += InnerProduct(f_body, psi) * dx  # k^2 <f, ψ>
    f_disp += InnerProduct(g_surface, psi) * ds  # k^2 _/‾ g·ψ ds

    disp_gfu.vec.data = (
        a_disp.mat.Inverse(fes_disp.FreeDofs(), inverse="sparsecholesky") * f_disp.vec
    )
    return disp_gfu


def FIRST_RUN_update_displacement(
    fes_disp, disp_gfu, vel_gfu, fes_eps_m, mag_gfu, f_body, g_surface, K
):
    """
    >Uses the initial velocity condition instead of a difference quotient.<
    Updates a displacement vector with the new values.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): VectorH1 finite element space.
        disp_gfu (ngsolve.comp.GridFunction): A VectorH1 grid function at the i=0 time step.
        vel_gfu (ngsolve.comp.GridFunction): A VectorH1 grid function that models the initial velocity.
        f_body (ngsolve.fem.CoefficientFunction): The body force.
        g_surface (ngsolve.fem.CoefficientFunction): The traction.

    Returns:
        disp_gfu (ngsolve.comp.GridFunction): The new updated displacement at the i=1 time step.
    """
    # Test functions
    u = fes_disp.TrialFunction()
    psi = fes_disp.TestFunction()
    # Building the linear system for the displacement
    a_disp = BilinearForm(fes_disp)
    a_disp += InnerProduct(u, psi) * dx  # <u^(i+1), ψ>
    a_disp += (
        K * K * InnerProduct(stress(strain(u)), strain(psi)) * dx
    )  # k^2<Cε(u), ε(ψ)>

    f_disp = LinearForm(fes_disp)
    f_disp += (
        InnerProduct(
            stress(
                magfunc.build_strain_m(fes_eps_m, magfunc.nodal_projection(mag_gfu))
            ),
            strain(psi),
        )
        * dx
    )  # <Cε_m(Π m),ε(ψ)>
    f_disp += K * InnerProduct(vel_gfu, psi) * dx  # k<d_t u^i, ψ>
    f_disp += InnerProduct(disp_gfu, psi) * dx  # <u^i, ψ>
    f_disp += InnerProduct(f_body, psi) * dx  # k^2 <f, ψ>
    f_disp += InnerProduct(g_surface, psi) * ds  # k^2 _/‾ g·ψ ds

    disp_gfu.vec.data = (
        a_disp.mat.Inverse(fes_disp.FreeDofs(), inverse="sparsecholesky") * f_disp.vec
    )
    return disp_gfu


def elastic_energy(mesh, disp_gfu, mag_gfu, f_body, g_surface, KAPPA) -> float:
    """
    >Uses the initial velocity condition instead of a difference quotient.<
    Updates a displacement vector with the new values.

    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.
        disp_gfu (ngsolve.comp.GridFunction): Displacement VectorH1 grid function.
        f_body (ngsolve.fem.CoefficientFunction): The body force.
        g_surface (ngsolve.fem.CoefficientFunction): The traction.
        KAPPA (float): Relative strength of magnetic to elastic contributions.

    Returns:
        KAPPA*energy (float): Elastic energy.
    """
    mystrain = strain_el(mag_gfu, disp_gfu)
    vol_integrand = 0.5 * InnerProduct(
        stress(mystrain), mystrain
    )  # -InnerProduct(f_body, disp_gfu)  # 1/2 <Cε_el(m,u), ε_el(m,u)> - <f,u>
    # surf_integrand = -InnerProduct(g_surface, disp_gfu)  # -<g,u>_BND
    energy = Integrate(vol_integrand, mesh, VOL)
    # energy += Integrate(surf_integrand, mesh, BND)
    return KAPPA * energy


def initial_kinetic_energy(mesh, vel_gfu, KAPPA) -> float:
    """
    Returns the kinetic energy KAPPA/2 _/‾ ||u_t||^2 dx.
    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.
        vel_gfu (ngsolve.comp.GridFunction): Initial velocity VectorH1 grid function.
        KAPPA (float): Relative strength of magnetic to elastic contributions.

    Returns:
        Kinetic energy (float): The initial kinetic energy.
    """
    integrand = InnerProduct(vel_gfu, vel_gfu)
    return KAPPA * 0.5 * Integrate(integrand, mesh, VOL)


def kinetic_energy(mesh, disp_gfu, disp_gfu_prev, KAPPA, K) -> float:
    """
    Returns the kinetic energy KAPPA/(2*K*K) _/‾ ||u^i - u^(i-1)||^2 dx.
    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.
        vel_gfu (ngsolve.comp.GridFunction): Initial velocity VectorH1 grid function.
        KAPPA (float): Relative strength of magnetic to elastic contributions.
        K (float): Timestep.

    Returns:
        Kinetic energy (float): The kinetic energy.
    """
    disp_diff = disp_gfu - disp_gfu_prev
    integrand = InnerProduct(disp_diff, disp_diff)
    return KAPPA * 0.5 * Integrate(integrand, mesh, VOL) / (K * K)


def Voigt_6x6_to_full_3x3x3x3(C):
    """
    Convert from the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.
    Parameters
    ----------
    C : array_like
        6x6 stiffness matrix (Voigt notation).
    Returns
    -------
    C : array_like
        3x3x3x3 stiffness matrix.
    Taken from https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py
    """

    C = np.asarray(C)
    C_out = np.zeros((3, 3, 3, 3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out


def full_3x3_to_Voigt_6_index(i, j):
    """
    Taken from https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py
    """
    if i == j:
        return i
    return 6 - i - j
