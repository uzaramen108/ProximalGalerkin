"""

Obstacle problem based on experiment 4 in [2].

The FEniCSx code solve this problem is based on [3]:

SPXD License: MIT License

Original license file [../../licenses/LICENSE.surowiec](../../licenses/LICENSE.surowiec)
is included in the repository.

[1] Keith, B. and Surowiec, T.M., Proximal Galerkin: A Structure-Preserving Finite Element Method
for Pointwise Bound Constraints. Found Comput Math (2024). https://doi.org/10.1007/s10208-024-09681-8
[2] Keith, B., Surowiec, T. M., & Dokken, J. S. (2023). Examples for the Proximal Galerkin Method
    (Version 0.1.0) [Computer software]. https://github.com/thomas-surowiec/proximal-galerkin-examples
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import basix
import numpy as np
import pandas as pd
import ufl
from dolfinx import default_scalar_type, fem, io, mesh
from dolfinx.fem.petsc import NonlinearProblem
from ufl import Measure, conditional, exp, grad, inner, lt, SpatialCoordinate


def rank_print(string: str, comm: MPI.Comm, rank: int = 0):
    """Helper function to print on a single rank

    :param string: String to print
    :param comm: The MPI communicator
    :param rank: Rank to print on, defaults to 0
    """
    if comm.rank == rank:
        print(string)


def allreduce_scalar(form: fem.Form, op: MPI.Op = MPI.SUM) -> np.floating:
    """Assemble a scalar form over all processes and perform a global reduction

    :param form: Scalar form
    :param op: MPI reduction operation
    """
    comm = form.mesh.comm
    return comm.allreduce(fem.assemble_scalar(form), op=op)


def solve_problem(
    filename: Path,
    polynomial_order: int,
    maximum_number_of_outer_loop_iterations: int,
    alpha_scheme: str,
    alpha_max: float,
    tol_exit: float,
):
    """ """

    # Create mesh
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    # Define FE subspaces
    P = basix.ufl.element("Lagrange", msh.basix_cell(), polynomial_order)
    mixed_element = basix.ufl.mixed_element([P, P])
    V = fem.functionspace(msh, mixed_element)

    # Define functions and parameters
    alpha = fem.Constant(msh, default_scalar_type(1))
    #f = fem.Constant(msh, 0.0)

    # ── 포물선(Parabolic) 형태의 외력 정의 ──
    x_coords = SpatialCoordinate(msh)
    r2 = x_coords[0]**2 + x_coords[1]**2  # 중심으로부터 거리의 제곱 (x^2 + y^2)
    
    R_mesh = 1.0   # 짐작되는 메시의 반지름 (사용하는 메시 크기에 맞게 수정)
    f_max = -5.0   # 중심부(r=0)에서 가해질 최대 하중 (원하는 세기로 조절)
    
    # 공식: f(x,y) = f_max * (1 - r^2 / R^2)
    # 중심에선 f_max, 테두리(R_mesh)에선 0이 됩니다.
    f = f_max * (1.0 - r2 / (R_mesh**2))
    # 여기까지 수정.

    # Define BCs
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    V0, _ = V.sub(0).collapse()
    dofs = fem.locate_dofs_topological((V.sub(0), V0), entity_dim=1, entities=facets)

    u_bc = fem.Function(V0)
    u_bc.x.array[:] = 0.0
    bcs = fem.dirichletbc(value=u_bc, dofs=dofs, V=V.sub(0))

    # Define solution variables
    sol = fem.Function(V)
    sol_k = fem.Function(V)

    u, psi = ufl.split(sol)
    u_k, psi_k = ufl.split(sol_k)

    def phi_set(x):
        # 1. 배경은 막이 닿지 않도록 아주 깊은 곳(-10.0)으로 설정
        phi_vals = np.full_like(x[0], -10.0) 
        
        # ─── 파라미터 설정 (입맛에 맞게 조절하세요) ───
        R = 0.1        # 원관(파이프)의 반지름 (굵기)
        L = 0.4       # 파이프 간의 간격 (격자의 촘촘함)
        Z_top = -0.2   # 파이프 가장 윗면의 높이 (장애물 최대 높이)
        Z_center = Z_top - R  # 파이프 중심축의 높이
        # ──────────────────────────────────────────────
        
        # 2. 각 좌표에서 가장 가까운 파이프 중심선까지의 수직 거리 계산
        # np.round(x / L) * L 은 가장 가까운 중심축(0, ±0.4, ±0.8...)의 좌표를 찾습니다.
        dx = np.abs(x[0] - np.round(x[0] / L) * L) # y축 방향으로 뻗은 파이프들과의 거리
        dy = np.abs(x[1] - np.round(x[1] / L) * L) # x축 방향으로 뻗은 파이프들과의 거리
        
        # 3. 파이프가 존재하는 영역 마스킹 (거리가 반지름 R 이하인 곳)
        mask_y_pipe = dx <= R
        mask_x_pipe = dy <= R
        
        # 4. 각 파이프의 둥근 곡면 높이 계산 (원방정식: z = z_center + sqrt(R^2 - d^2))
        h_y_pipe = np.full_like(x[0], -10.0)
        h_y_pipe[mask_y_pipe] = Z_center + np.sqrt(R**2 - dx[mask_y_pipe]**2)
        
        h_x_pipe = np.full_like(x[0], -10.0)
        h_x_pipe[mask_x_pipe] = Z_center + np.sqrt(R**2 - dy[mask_x_pipe]**2)
        
        # 5. 가로 파이프와 세로 파이프 중 더 높은 값을 취함 (파이프들이 교차하는 형상 구현)
        phi_vals = np.maximum(phi_vals, h_y_pipe)
        phi_vals = np.maximum(phi_vals, h_x_pipe)
        
        return phi_vals

    quadrature_degree = 6
    Qe = basix.ufl.quadrature_element(msh.topology.cell_name(), degree=quadrature_degree)
    Vq = fem.functionspace(msh, Qe)
    # Lower bound for the obstacle
    phi = fem.Function(Vq, name="phi")
    phi.interpolate(phi_set)

    # Define non-linear residual
    (v, w) = ufl.TestFunctions(V)
    dx = Measure("dx", domain=msh, metadata={"quadrature_degree": quadrature_degree})
    F = (
        alpha * inner(grad(u), grad(v)) * dx
        + psi * v * dx
        + u * w * dx
        - exp(psi) * w * dx
        - phi * w * dx
        - alpha * f * v * dx
        - psi_k * v * dx
    )
    J = ufl.derivative(F, sol)

    # Setup non-linear problem
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "ksp_monitor": None,
        "snes_monitor": None,
        "snes_error_if_not_converged": True,
        "snes_linesearch_type": "none",
        "snes_rtol": 1e-6,
        "snes_max_it": 100,
    }
    problem = NonlinearProblem(
        F, u=sol, bcs=[bcs], J=J, petsc_options=petsc_options, petsc_options_prefix="obstacle_"
    )

    # observables
    energy_form = fem.form(0.5 * inner(grad(u), grad(u)) * dx - f * u * dx)
    complementarity_form = fem.form((psi_k - psi) / alpha * u * dx)
    feasibility_form = fem.form(conditional(lt(u, 0), -u, fem.Constant(msh, 0.0)) * dx)
    dual_feasibility_form = fem.form(
        conditional(lt(psi_k, psi), (psi - psi_k) / alpha, fem.Constant(msh, 0.0)) * dx
    )
    H1increment_form = fem.form(inner(grad(u - u_k), grad(u - u_k)) * dx + (u - u_k) ** 2 * dx)
    L2increment_form = fem.form((exp(psi) - exp(psi_k)) ** 2 * dx)

    # Proximal point outer loop
    n = 0
    increment_k = 0.0
    sol.x.array[:] = 0.0
    sol_k.x.array[:] = sol.x.array[:]
    alpha_k = 1
    step_size_rule = alpha_scheme
    C = 1.0
    r = 1.5
    q = 1.5

    energies = []
    complementarities = []
    feasibilities = []
    dual_feasibilities = []
    Newton_steps = []
    step_sizes = []
    primal_increments = []
    latent_increments = []
    for k in range(maximum_number_of_outer_loop_iterations):
        # Update step size
        if step_size_rule == "constant":
            alpha.value = C
        elif step_size_rule == "double_exponential":
            try:
                alpha.value = max(C * r ** (q**k) - alpha_k, C)
            except OverflowError:
                pass
            alpha_k = alpha.value
            alpha.value = min(alpha.value, alpha_max)
        else:
            step_size_rule == "geometric"
            alpha.value = C * r**k
        rank_print(f"OUTER LOOP {k + 1} alpha: {alpha.value}", msh.comm)

        # Solve problem
        problem.solve()
        converged_reason = problem.solver.getConvergedReason()
        n = problem.solver.getIterationNumber()
        rank_print(f"Newton steps: {n}   Converged: {converged_reason}", msh.comm)

        # Check outer loop convergence
        energy = allreduce_scalar(energy_form)
        complementarity = np.abs(allreduce_scalar(complementarity_form))
        feasibility = allreduce_scalar(feasibility_form)
        dual_feasibility = allreduce_scalar(dual_feasibility_form)
        increment = np.sqrt(allreduce_scalar(H1increment_form))
        latent_increment = np.sqrt(allreduce_scalar(L2increment_form))

        tol_pp = increment

        if increment_k > 0.0:
            rank_print(
                f"Increment size: {increment}" + f"   Ratio: {increment / increment_k}", msh.comm
            )
        else:
            rank_print(f"Increment size: {increment}", msh.comm)
        rank_print("", msh.comm)

        energies.append(energy)
        complementarities.append(complementarity)
        feasibilities.append(feasibility)
        dual_feasibilities.append(dual_feasibility)
        Newton_steps.append(n)
        step_sizes.append(np.copy(alpha.value))
        primal_increments.append(increment)
        latent_increments.append(latent_increment)

        if tol_pp < tol_exit:
            break

        # Update sol_k with sol_new
        sol_k.x.array[:] = sol.x.array[:]
        increment_k = increment

    # # Save data
    cwd = Path.cwd()
    output_dir = cwd / "output"
    output_dir.mkdir(exist_ok=True)

    # Create output space for bubble function
    V_primal, primal_to_mixed = V.sub(0).collapse()

    num_primal_dofs = V_primal.dofmap.index_map.size_global

    phi_out_space = fem.functionspace(msh, basix.ufl.element("Lagrange", msh.basix_cell(), 6))
    phi_out = fem.Function(phi_out_space, name="phi")
    phi_out.interpolate(phi_set)
    with io.VTXWriter(msh.comm, output_dir / "phi.bp", [phi_out]) as bp:
        bp.write(0.0)
    if MPI.COMM_WORLD.rank == 0:
        df = pd.DataFrame()
        df["Energy"] = energies
        df["Complementarity"] = complementarities
        df["Feasibility"] = feasibilities
        df["Dual Feasibility"] = dual_feasibilities
        df["Newton steps"] = Newton_steps
        df["Step sizes"] = step_sizes
        df["Primal increments"] = primal_increments
        df["Latent increments"] = latent_increments
        df["Polynomial order"] = np.full(k + 1, polynomial_order)
        df["dofs"] = np.full(k + 1, num_primal_dofs)
        df["Step size rule"] = [step_size_rule] * (k + 1)
        filename = f"./example_polyorder{polynomial_order}_{num_primal_dofs}.csv"
        print(f"Saving data to: {str(output_dir / filename)}")
        df.to_csv(output_dir / filename, index=False)
        rank_print(df, msh.comm)

    if k == maximum_number_of_outer_loop_iterations - 1:
        rank_print("Maximum number of outer loop iterations reached", msh.comm)
    return sol, sum(Newton_steps)


# -------------------------------------------------------
if __name__ == "__main__":
    desc = "Run examples from paper"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file-path",
        "-f",
        dest="filename",
        type=Path,
        required=True,
        help="Name of input file",
    )
    parser.add_argument(
        "--polynomial_order",
        "-p",
        dest="polynomial_order",
        type=int,
        default=1,
        choices=[1, 2],
        help="Polynomial order of primal space",
    )
    parser.add_argument(
        "--alpha-scheme",
        dest="alpha_scheme",
        type=str,
        default="constant",
        choices=["constant", "double_exponential", "geometric"],
        help="Step size rule",
    )
    parser.add_argument(
        "--max-iter",
        "-i",
        dest="maximum_number_of_outer_loop_iterations",
        type=int,
        default=100,
        help="Maximum number of outer loop iterations",
    )
    parser.add_argument(
        "--alpha-max",
        "-a",
        dest="alpha_max",
        type=float,
        default=1e5,
        help="Maximum alpha",
    )
    parser.add_argument(
        "--tol",
        "-t",
        dest="tol_exit",
        type=float,
        default=1e-6,
        help="Tolerance for exiting Newton iteration",
    )
    args = parser.parse_args()
    solve_problem(
        args.filename,
        args.polynomial_order,
        args.maximum_number_of_outer_loop_iterations,
        args.alpha_scheme,
        args.alpha_max,
        args.tol_exit,
    )