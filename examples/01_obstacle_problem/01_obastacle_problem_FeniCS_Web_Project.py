from pathlib import Path

from mpi4py import MPI

import dolfinx.io
import gmsh
import packaging.version

__all__ = ["generate_disk"]


def generate_disk(filename: Path, res: float, order: int = 1, refinement_level: int = 1):
    """Generate a disk around the origin with radius 1 and resolution `res`.

    Args:
        filename: Name of the file to save the mesh to.
        res: Resolution of the mesh.
        order: Order of the mesh elements.
        refinement_level: Number of gmsh refinements
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(order)
        for _ in range(refinement_level):
            gmsh.model.mesh.refine()
            gmsh.model.mesh.setOrder(order)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    model = dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
    msh = model[0]
    gmsh.finalize()
    out_name = filename.with_stem(f"{filename.stem}_{refinement_level}").with_suffix(".xdmf")
    filename.parent.mkdir(exist_ok=True, parents=True)
    with dolfinx.io.XDMFFile(mesh_comm, out_name, "w") as xdmf:
        xdmf.write_mesh(msh)


if __name__ == "__main__":
    for i in range(4):
        generate_disk(Path("meshes/disk.xdmf"), res=0.1, order=2, refinement_level=i)

"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import scipy.sparse
import ufl

from lvpp.optimization import galahad_solver, ipopt_solver

parser = argparse.ArgumentParser(
    description="""Solve the obstacle problem on a general mesh using a spatially varying
      phi using Galahad or IPOPT""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--path",
    "-P",
    dest="infile",
    type=Path,
    default=Path("meshes/disk_0.xdmf"),
    help="Path to infile",
)
parser.add_argument("--ipopt", action="store_true", default=False, help="Use Ipopt")
parser.add_argument("--galahad", action="store_true", default=False, help="Use Galahad")
parser.add_argument("--max-iter", type=int, default=200, help="Maximum number of iterations")
parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
parser.add_argument(
    "--hessian", dest="use_hessian", action="store_true", default=False, help="Use exact hessian"
)
parser.add_argument(
    "--output", "-o", dest="outdir", type=Path, default=Path("results"), help="Output directory"
)


def setup_problem(
    filename: Path,
):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(Vh)
    v = ufl.TestFunction(Vh)

    mass = ufl.inner(u, v) * ufl.dx
    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(Vh, tdim - 1, boundary_facets)

    # Get dofs to deactivate
    bcs = [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, Vh)]

    def psi(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        r0 = 0.5
        beta = 0.9
        b = r0 * beta
        tmp = np.sqrt(r0**2 - b**2)
        B = tmp + b * b / tmp
        C = -b / tmp
        cond_true = B + r * C
        cond_false = np.sqrt(np.clip(r0**2 - r**2, 0, None))
        true_indices = np.flatnonzero(r > b)
        cond_false[true_indices] = cond_true[true_indices]
        return cond_false

    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)

    f = dolfinx.fem.Function(Vh)
    f.x.array[:] = 0.0
    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(mass))

    return S.to_scipy(), M.to_scipy(), f, (lower_bound, upper_bound)


class ObstacleProblem:
    total_iteration_count: int

    def __init__(self, S, M, f):
        S.eliminate_zeros()
        self._S = S
        self._M = M
        self._Mf = M @ f
        self._f = f
        tri_S = scipy.sparse.tril(self._S)
        self._sparsity = tri_S.nonzero()
        self._H_data = tri_S.data

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return 0.5 * x.T @ (self._S @ x) - self._f.T @ (self._M @ x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""

        return self._S @ x - self._Mf

    def pure_hessian(self, x):
        return self._H_data

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * self.pure_hessian(x)

    def hessianstructure(self):
        return self._sparsity

    def intermediate(self, *args):
        """Ipopt callback function"""
        self.total_iteration_count = args[1]


if __name__ == "__main__":
    args = parser.parse_args()

    S_, M_, f_, bounds_ = setup_problem(args.infile)
    V = f_.function_space
    bounds = tuple(b.x.array for b in bounds_)
    # Restrict all matrices and vectors to interior dofs
    problem = ObstacleProblem(S_.copy(), M_.copy(), f_.x.array)
    outdir = args.outdir
    if args.galahad:
        x_g = dolfinx.fem.Function(V, name="galahad")
        x_g.x.array[:] = 0.0
        init_galahad = x_g.x.array.copy()
        x_galahad, iterations = galahad_solver(
            problem,
            init_galahad,
            bounds,
            max_iter=args.max_iter,
            use_hessian=args.use_hessian,
            tol=args.tol,
        )
        x_g.x.array[:] = x_galahad
        with dolfinx.io.VTXWriter(V.mesh.comm, outdir / "galahad_obstacle.bp", [x_g]) as bp:
            bp.write(0.0)

    if args.ipopt:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        init_ipopt = x_i.x.array.copy()
        x_ipopt = ipopt_solver(
            problem,
            init_ipopt,
            bounds,
            max_iter=args.max_iter,
            tol=args.tol,
            activate_hessian=args.use_hessian,
        )

        x_i.x.array[:] = x_ipopt

        # Output on geometry space
        mesh = x_i.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, outdir / "ipopt_obstacle.bp", [x_i_out]) as bp:
            bp.write(0.0)

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
from ufl import Measure, conditional, exp, grad, inner, lt


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
    f = fem.Constant(msh, 0.0)
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
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        r0 = 0.5
        beta = 0.9
        b = r0 * beta
        tmp = np.sqrt(r0**2 - b**2)
        B = tmp + b * b / tmp
        C = -b / tmp
        cond_true = B + r * C
        cond_false = np.sqrt(np.clip(r0**2 - r**2, 0, None))
        true_indices = np.flatnonzero(r > b)
        cond_false[true_indices] = cond_true[true_indices]
        return cond_false

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
        default=Path("meshes/disk_0.xdmf"),
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