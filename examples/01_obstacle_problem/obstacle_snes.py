"""
Solve obstacle problem with SNES
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT

The SNES solver is based on https://github.com/Wells-Group/asimov-contact
and is distributed under the MIT License.
The license file can be found under [../../licenses/LICENSE.asimov](../../licenses/LICENSE.asimov)
"""

import argparse
import typing
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import ufl

parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square.",
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


def snes_solve(
    filename: Path,
    snes_options: typing.Optional[dict] = None,
):
    snes_options = {} if snes_options is None else snes_options

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

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

    # Lower bound for the obstacle
    phi = dolfinx.fem.Function(V)
    phi.interpolate(phi_set)

    uh = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    # 수정된 코드: 포물선(Parabolic) 형태의 외력 정의
    x_coords = ufl.SpatialCoordinate(mesh)
    r2 = x_coords[0]**2 + x_coords[1]**2  # 중심으로부터 거리의 제곱 (x^2 + y^2)
    
    R_mesh = 1.0   # 사용하는 메시(원판)의 반지름
    f_max = -5.0   # 중심부에서 아래로 누르는 최대 하중 (이전 코드와 동일하게 설정)
    
    # 중심(r=0)에선 f_max, 테두리(R_mesh)에선 0이 되는 포물선 분포
    f_expression = f_max * (1.0 - r2 / (R_mesh**2))
    
    # 안전장치: 메시 밖에서 하중이 반대로 작용하는 것을 방지
    f = ufl.conditional(ufl.gt(f_expression, 0.0), 0.0, f_expression)
    F = (ufl.inner(ufl.grad(uh), ufl.grad(v)) - ufl.inner(f, v)) * ufl.dx

    # bc_expr = dolfinx.fem.Expression(u_ex, V.element.interpolation_points())
    u_bc = dolfinx.fem.Function(V)
    u_bc.x.array[:] = 0.0
    # u_bc.interpolate(bc_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcs = [dolfinx.fem.dirichletbc(u_bc, boundary_dofs)]

    u_max = dolfinx.fem.Function(V)
    u_max.x.array[:] = PETSc.INFINITY

    # Create nonlinear problem
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, uh, bcs=bcs, petsc_options=snes_options, petsc_options_prefix="snes_"
    )
    problem.solver.setVariableBounds(phi.x.petsc_vec, u_max.x.petsc_vec)
    problem.solve()

    num_iterations = problem.solver.getIterationNumber()

    mesh = uh.function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u_out = dolfinx.fem.Function(V_out, name="llvp")
    u_out.interpolate(uh)

    return u_out, num_iterations


if __name__ == "__main__":
    args = parser.parse_args()
    snes_solve(
        args.infile,
        snes_options={
            "snes_type": "vinewtonssls",
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_max_it": 1000,
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "snes_stol": 1e-8,
        },
    )
