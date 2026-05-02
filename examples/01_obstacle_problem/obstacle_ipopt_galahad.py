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
    required=True,
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

    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)

    # ── ▼▼▼ 외력 f 설정 부분 수정 ▼▼▼ ──
    def f_set(x):
        r2 = x[0]**2 + x[1]**2
        R_mesh = 1.0
        f_max = -5.0  # 이전 비교군과 동일한 최대 하중
        
        # 포물선 프로파일 계산
        f_vals = f_max * (1.0 - r2 / (R_mesh**2))
        
        # 안전장치: 가장자리에서 값이 양수가 되어 위로 당기는 현상 차단
        return np.minimum(f_vals, 0.0)

    f = dolfinx.fem.Function(Vh)
    f.interpolate(f_set)
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
