import gmsh
import math
import sys
from pathlib import Path
import meshio

def generate_pipe_net_mesh():
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("PipeNet")

    # ─── 파라미터 설정 ───
    R = 0.05        # 파이프 반지름 (굵기)
    L = 0.4         # 파이프 간격
    R_domain = 1.0  # 전체 디스크(도메인)의 반지름
    
    # ▼▼▼ Z 좌표 설정 추가 ▼▼▼
    Z_top = -0.2          # 장애물의 최대 높이 (입력)
    Z_center = Z_top - R  # 파이프 중심축의 Z 좌표 계산
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    lines = []
    coord = 0.0
    while coord <= R_domain:
        lines.append(coord)
        if coord > 0:
            lines.append(-coord)
        coord += L

    cylinder_tags = []

    # 1. X축 방향 파이프 생성 (z 좌표를 Z_center로 수정)
    for y in lines:
        if abs(y) >= R_domain: continue
        x_length = math.sqrt(R_domain**2 - y**2)
        # addCylinder(x, y, z, dx, dy, dz, radius)
        tag = gmsh.model.occ.addCylinder(-x_length, y, Z_center, 2*x_length, 0.0, 0.0, R)
        cylinder_tags.append((3, tag))

    # 2. Y축 방향 파이프 생성 (z 좌표를 Z_center로 수정)
    for x in lines:
        if abs(x) >= R_domain: continue
        y_length = math.sqrt(R_domain**2 - x**2)
        tag = gmsh.model.occ.addCylinder(x, -y_length, Z_center, 0.0, 2*y_length, 0.0, R)
        cylinder_tags.append((3, tag))

    # 3. 파이프 합치기
    out, out_map = gmsh.model.occ.fuse([cylinder_tags[0]], cylinder_tags[1:])
    gmsh.model.occ.synchronize()

    # 메쉬 해상도 설정
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.005)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

    print("Generating 3D mesh for the pipe net...")
    gmsh.model.mesh.generate(3)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    temp_msh_file = str(output_dir / "temp_pipe_net.msh")
    final_xdmf_file = str(output_dir / "pipe_net_3d.xdmf")
    
    gmsh.write(temp_msh_file)
    print("Temporary .msh file generated.")
    gmsh.finalize()

    print(f"Converting to XDMF format: {final_xdmf_file} ...")
    mesh_data = meshio.read(temp_msh_file)
    
    tetra_mesh = meshio.Mesh(
        points=mesh_data.points,
        cells=[("tetra", mesh_data.get_cells_type("tetra"))]
    )
    meshio.write(final_xdmf_file, tetra_mesh)
    print("Conversion complete! Z-coordinate successfully applied.")

if __name__ == "__main__":
    generate_pipe_net_mesh()