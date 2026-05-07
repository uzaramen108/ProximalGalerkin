#!/bin/bash

echo "========================================"
echo "🚀 1단계: 도커 컨테이너 시작"
echo "========================================"
docker start proximal-custom

echo ""
echo "========================================"
echo "🐳 2단계: 도커 내부 작업 실행"
echo "========================================"
docker exec proximal-custom //bin/bash -c "
export HDF5_USE_FILE_LOCKING=FALSE
cd /home/jovyan/shared/examples/01_obstacle_problem

echo '>> FEniCSx 버전 확인:'
python3 -c 'import dolfinx; print(dolfinx.__version__)'

echo ''
echo '>> 메쉬 생성 중 (generate_mesh_gmsh.py)...'
python3 generate_mesh_gmsh.py

echo ''
echo '>> [1/3] Coarse 시뮬레이션 실행 중...'
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse 2>&1 | tee output/simulation_coarse_log.txt

echo ''
echo '>> [2/3] Medium 시뮬레이션 실행 중...'
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium 2>&1 | tee output/simulation_medium_log.txt

echo ''
echo '>> [3/3] Fine 시뮬레이션 실행 중...'
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine 2>&1 | tee output/simulation_fine_log.txt

echo ''
echo '>> 파이프 그물망 3D 메쉬 생성 중 (generate_pipe_net.py)...'
python3 generate_pipe_net.py
"

echo ""
echo "========================================"
echo "📂 3단계: 결과 파일을 다운로드 폴더로 이동"
echo "========================================"
mv examples/01_obstacle_problem/coarse /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/medium /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/fine /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/output /c/Users/parkg/Downloads/

echo ""
echo "🎉 모든 시뮬레이션 및 파일 이동이 완벽하게 끝났습니다! 다운로드 폴더를 확인하세요."
