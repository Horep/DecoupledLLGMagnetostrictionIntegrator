from ngsolve import *
from random import random
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import General_Functions as genfunc
import Elastic_Functions as elfunc
import QMatrix
import time


def give_random_magnetisation(mag_grid_func: GridFunction) -> GridFunction:
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


def give_uniform_magnetisation(mag_grid_func: GridFunction, direction) -> GridFunction:
    """
    Returns a random normalised magnetisation grid function.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A uniform VectorH1 grid function with value in [-1,1]^3 and length 1 at each node.
    """
    num_points = genfunc.get_num_nodes(mag_grid_func)
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    # a, b, c = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
    a, b, c = direction[0], direction[1], direction[2]
    size = math.sqrt(a * a + b * b + c * c)
    try:
        a, b, c = a / size, b / size, c / size
    except (
        ZeroDivisionError
    ):  # it is extremely unlikely, but possible, to have a=b=c=0. If this happens, use (1,0,0)
        a, b, c = 1, 0, 0
    print(f"a={a},b={b},c={c}, size={math.sqrt(a * a + b * b + c * c)}")
    for i in range(num_points):
        mag_gfux.vec[i] = a
        mag_gfuy.vec[i] = b
        mag_gfuz.vec[i] = c
    return mag_grid_func


def nodal_projection(mag_gfu: GridFunction, fes_mag) -> GridFunction:
    """
    Returns a grid function with all nodal values projected onto the unit sphere. Every node z will satisfy |m(z)|=1.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function with length 1 at each node.
    """
    projected_func = GridFunction(fes_mag)
    num_points = genfunc.get_num_nodes(mag_gfu)
    mag_gfux, mag_gfuy, mag_gfuz = mag_gfu.components
    projected_funcx, projected_funcy, projected_funcz = projected_func.components
    for i in range(num_points):
        a = mag_gfux.vec[i]
        b = mag_gfuy.vec[i]
        c = mag_gfuz.vec[i]
        size = math.sqrt(a * a + b * b + c * c)

        projected_funcx.vec[i] = a / size
        projected_funcy.vec[i] = b / size
        projected_funcz.vec[i] = c / size
    return projected_func


def build_tangent_plane_matrix(
    mag_grid_func: GridFunction,
) -> scipy.sparse.csr.csr_matrix:
    """
    Returns the tangent plane matrix used in the saddle point formulation for the tangent plane update.

    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): A VectorH1 grid function.

    Returns:
        B (scipy.sparse.csr.csr_matrix): Sparse Nx3N tangent plane matrix.
    """
    mag_gfux, mag_gfuy, mag_gfuz = mag_grid_func.components
    N = genfunc.get_num_nodes(mag_grid_func)
    #  Cast the components of mag_grid_func to flat vector numpy arrays,
    #  and then assemble B as a block matrix from diagonal matrices of m1,m2,m3.
    m1 = scipy.sparse.spdiags(mag_gfux.vec.FV().NumPy()[:], diags=0, m=N, n=N)
    m2 = scipy.sparse.spdiags(mag_gfuy.vec.FV().NumPy()[:], diags=0, m=N, n=N)
    m3 = scipy.sparse.spdiags(mag_gfuz.vec.FV().NumPy()[:], diags=0, m=N, n=N)
    B = scipy.sparse.bmat([[m1, m2, m3]], format="csr")
    return B


def give_magnetisation_update(
    A: BilinearForm,
    B: scipy.sparse.csr.csr_matrix,
    F: LinearForm,
    A_FIXED: BilinearForm,
) -> np.ndarray:
    """
    Returns the tangent plane update v^(i) to the magnetisation such that m^(i+1) = m^(i) + v^(i).

    Parameters:
        A (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix from the variational formulation.
        B (ngsolve.bla.MatrixD): The Nx3N tangent plane matrix.
        F (ngsolve.comp.LinearForm): The 3Nx1 assembled force vector from the variational formulation.

    Returns:
        vlam (numpy.ndarray): The set of components to use for the update.
    """
    #  Convert to dense numpy matrix for A, and convert F to a numpy array.
    #  Converting to dense is bad for performance.
    rows, cols, vals = A.mat.COO()
    A = scipy.sparse.csr_matrix((vals, (rows, cols)))
    rows, cols, vals = A_FIXED.mat.COO()
    A_FIXED = scipy.sparse.csr_matrix((vals, (rows, cols)))
    A += A_FIXED
    F = F.vec.FV().NumPy()[:]
    assert len(F) % 3 == 0, "The force vector is not a multiple of three, very bad."
    N = len(F) // 3
    #  Make block stiffness matrix and block force vector and then solve.
    #  Throw away last N terms as these are the lagrange multipliers enforcing the tangent plane.
    stiffness_block = scipy.sparse.bmat([[A, B.transpose()], [B, None]], format="csr")
    force_block = np.concatenate((F, np.zeros(N)), axis=0)
    time2 = time.time()
    M2 = scipy.sparse.linalg.spilu(
        stiffness_block
    )  # spilu preconditioner used in GMRES algorithm below.
    M = scipy.sparse.linalg.LinearOperator((4 * N, 4 * N), M2.solve)
    vlam, myinfo = scipy.sparse.linalg.gmres(
        stiffness_block, force_block, tol=1e-8, M=M
    )
    time3 = time.time()
    v = np.asarray(vlam)[0 : 3 * N]
    residual = np.linalg.norm(
        B.dot(v), np.inf
    )  # in theory, the update should satisfy |Bv| = 0.
    print(f"GMRES completed in {time3-time2}, info={myinfo}, residual = {residual}")
    if residual > 1e-8:
        print(
            f"WARNING: |Bv| = {residual} > 1e-8. Tangent plane matrix B or update v may not be correctly calculated."
        )
    return v


def build_strain_m(mag_grid_func: GridFunction, lambda_m: float) -> CoefficientFunction:
    """
    Builds a Coefficient function matrix of the form
        m1*m1-1/3 m1*m2     m1*m3\n
        m2*m1     m2*m2-1/3 m2*m3\n
        m3*m1     m3*m2     m3*m3-1/3
    from an input magnetisation of the form (m1,m2,m3) scaled by 3/2 lambda_m.

    Parameters:
        fes_eps_m (ngsolve.comp.MatrixValued): A matrix valued FE space for magnetisation.
        mag_grid_func (ngsolve.comp.GridFunction): Input magnetisation grid function.
        lambda100 (float): The saturation magnetostrictive strain.
    Returns:
        mymatrix (ngsolve.fem.CoefficientFunction): The magnetostrain matrix.
    """
    mymatrix = OuterProduct(mag_grid_func, mag_grid_func) - 1 / 3 * Id(3)
    mymatrix = 3 * lambda_m / 2 * mymatrix
    return mymatrix


def build_fixed_mag(
    fes_mag: VectorH1, ALPHA: float, THETA: float, K: float
) -> BilinearForm:
    """
    Computes the fixed matrix sum of the mass and stiffness matrix for the magnetisation.
    """
    massLumping = IntegrationRule(
        points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        weights=[1 / 24, 1 / 24, 1 / 24, 1 / 24],
    )  # use mass lumping approach for integration
    with TaskManager():
        v = fes_mag.TrialFunction()
        phi = fes_mag.TestFunction()
        M_mag_fixed = BilinearForm(fes_mag)
        L_mag_fixed = BilinearForm(fes_mag)
        M_mag_fixed += SymbolicBFI(
            ALPHA * InnerProduct(v, phi), intrule=massLumping
        )  # α<v,Φ>
        L_mag_fixed += SymbolicBFI(
            THETA * K * InnerProduct(Grad(v), Grad(phi)), intrule=massLumping
        )  # θk<∇v,∇Φ>
        M_mag_fixed.Assemble()
        L_mag_fixed.Assemble()
    return M_mag_fixed, L_mag_fixed


def build_magnetic_lin_system(
    fes_mag: VectorH1,
    mag_gfu: GridFunction,
    ALPHA: float,
    THETA: float,
    K: float,
    KAPPA: float,
    disp_gfu: GridFunction,
    mu: float,
    lam: float,
    lambda_m: float,
    zeeman: CoefficientFunction,
):
    """
    Builds the variational formulation in terms of H1 components.
    Does not account for unit length constraint other than use of tangent plane scheme.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): VectorH1 finite element space.
        mag_gfu (ngsolve.comp.GridFunction): A VectorH1 grid function.
        fes_eps_m (ngsolve.comp.MatrixValued): Matrix finite element space.
        ALPHA (float): Dissipative constant in LLG equation.
        THETA (float): Implicitness parameter.
        K (float): Time step.
        KAPPA (float): Relative strength of magnetic vs. elastic contributions.

    Returns:
        a_mag (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix.
        f_mag (ngsolve.comp.LinearForm): The 3Nx1 assembled force vector.
    """
    # Test functions
    v = fes_mag.TrialFunction()
    phi = fes_mag.TestFunction()
    proj_mag = nodal_projection(mag_gfu, fes_mag)
    magnetostrain = elfunc.magnetostriction_field(
        elfunc.strain(disp_gfu), proj_mag, mu, lam, lambda_m
    )
    # Building the linear system for the magnetisation
    with TaskManager():
        a_mag = BilinearForm(fes_mag)
        a_mag += InnerProduct(Cross(mag_gfu, v), phi) * dx  # <m×v,Φ>
        a_mag.Assemble()
        f_mag = LinearForm(fes_mag)
        f_mag += -InnerProduct(Grad(mag_gfu), Grad(phi)) * dx  # -<∇m,∇Φ>
        f_mag += KAPPA * InnerProduct(magnetostrain, phi) * dx  # <h_m , Φ>
        f_mag += InnerProduct(zeeman, phi) * dx  # <h_external , Φ>
        f_mag.Assemble()
    return a_mag, f_mag


def update_magnetisation(
    fes_mag: VectorH1,
    mag_gfu: GridFunction,
    ALPHA: float,
    THETA: float,
    K: float,
    KAPPA: float,
    disp_gfu: GridFunction,
    mu: float,
    lam: float,
    lambda_m: float,
    zeeman: GridFunction,
    M_mag_fixed: BilinearForm,
    L_mag_fixed: BilinearForm,
    v0: np.ndarray,
):
    """
    Updates a magnetisation vector with the new values.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): VectorH1 finite element space.
        mag_gfu (ngsolve.comp.GridFunction): A VectorH1 grid function.
        fes_eps_m (ngsolve.comp.MatrixValued): Matrix finite element space.
        ALPHA (float): Dissipative constant in LLG equation.
        THETA (float): Implicitness parameter.
        K (float): Time step.
        KAPPA (float): Relative strength of magnetic vs. elastic contributions.
        mu (float): The first lame constant.
        lam (float): The second lame constant
        lambda_m (float): The magnetostrain constant
        zeeman (GridFunction): The external field gridfunction
        M_mag_fixed (BilinearForm): The fixed mass matrix. Should use mass lumping, and be constant.
        L_mag_fixed (BilinearForm): The fixed skew matrix. Should use mass lumping, and be constant.
        v0 (np.ndarray): The starting point for the GMRES tangent plane update. Should either use previous step, or zeros.

    Returns:
        mag_grid_func (ngsolve.comp.GridFunction): The new updated magnetisation at the next time step.
        v (np.ndarray): The tangent plane update. Should be used as v0 in the next iteration.
    """
    a_mag, f_mag = build_magnetic_lin_system(
        fes_mag, mag_gfu, ALPHA, THETA, K, KAPPA, disp_gfu, mu, lam, lambda_m, zeeman
    )
    B = build_tangent_plane_matrix(mag_gfu)
    Q = QMatrix.qmatrix(mag_gfu, fes_mag)
    v = QMatrix.give_q_magnetisation_update(
        a_mag, B, f_mag, M_mag_fixed, L_mag_fixed, Q, v0
    )
    print(
        f"biggest update = {K*np.amax(v)}"
    )  # gives an idea of how large the updates are. Big is bad.
    N = fes_mag.ndof
    mag_gfux, mag_gfuy, mag_gfuz = mag_gfu.components
    mag_gfux.vec.FV().NumPy()[:] += K * v[0:N]
    mag_gfuy.vec.FV().NumPy()[:] += K * v[N : 2 * N]
    mag_gfuz.vec.FV().NumPy()[:] += K * v[2 * N : 3 * N]
    return mag_gfu, v


def magnetic_energy(mag_gfu: GridFunction, mesh: Mesh, f_zeeman) -> float:
    """
    Returns 1/2 _/‾||∇m||^2 dx, integrated over the mesh.
    Parameters:
        mag_gfu (ngsolve.comp.GridFunction): A VectorH1 grid function.
        mesh (ngsolve.comp.Mesh): The mesh on which the solution is defined.

    Returns:
        magnetic_energy (float): The magnetic energy 1/2 _/‾||∇m||^2 dx.
    """
    integrand = 0.5 * InnerProduct(Grad(mag_gfu), Grad(mag_gfu)) - InnerProduct(
        f_zeeman, mag_gfu
    )
    return Integrate(integrand, mesh, VOL)


def projected_magnetic_energy(mag_gfu: GridFunction, mesh: Mesh, fes_mag) -> float:
    """
    Returns 1/2 _/‾||∇proj(m)||^2 dx, integrated over the mesh.
    Useful for checking how much energy is contributed from the direction vs. magnitude.
    """
    my_projection = nodal_projection(mag_gfu, fes_mag)
    return 0.5 * Integrate(
        InnerProduct(Grad(my_projection), Grad(my_projection)), mesh, VOL
    )


def nodal_norm(mag_grid_func: GridFunction) -> float:
    """
    Returns the maximum magnitude of the nodal points.
    """
    mag_x, mag_y, mag_z = mag_grid_func.components
    node_x, node_y, node_z = (
        mag_x.vec.FV().NumPy()[:],
        mag_y.vec.FV().NumPy()[:],
        mag_z.vec.FV().NumPy()[:],
    )
    magnitudes = np.sqrt(node_x**2 + node_y**2 + node_z**2)
    return np.amax(magnitudes)
