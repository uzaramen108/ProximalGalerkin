bash 1.

도커 시작

docker start proximal-custom

docker exec -it proximal-custom //bin/bash

export HDF5_USE_FILE_LOCKING=FALSE
cd shared/examples/01_obstacle_problem

여기까지 했으면 이제 fenics 제대로 설치되었는지 확인 필요

python -c "import dolfinx; print(dolfinx.__version__)"

0.10.0 나오면 다음 명령어 실행

python3 generate_mesh_gmsh.py

셋 중 1개 또는 2~3개 실행

python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse 2>&1 | tee output/simulation_coarse_log.txt
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium 2>&1 | tee output/simulation_medium_log.txt
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine 2>&1 | tee output/simulation_fine_log.txt

그물 메시 생성
pip install h5py
python3 generate_pipe_net.py


bash 2.

생성된 파일 다운로드로 이동하는 코드.
mv examples/01_obstacle_problem/coarse /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/medium /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/fine /c/Users/parkg/Downloads/
mv examples/01_obstacle_problem/output /c/Users/parkg/Downloads/