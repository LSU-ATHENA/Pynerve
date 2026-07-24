#!/bin/bash
#SBATCH --job-name=pynerve-tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/pradip0/Comp/Pynerve/test_output_%j.log
#SBATCH --error=/work/pradip0/Comp/Pynerve/test_error_%j.log

set -e

echo "=== Starting Pynerve HPC test run ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load cuda/12.1.1 cmake gcc python/3.9.7-anaconda

echo "CUDA: $(which nvcc)"
echo "CMake: $(which cmake)"
echo "GCC: $(which g++)"

ENV_DIR=/work/pradip0/Comp/pynerve_env
if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.11 -y
    source activate "$ENV_DIR"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install pybind11 numpy pytest pytest-timeout pytest-xdist pytest-benchmark coverage ninja
else
    source activate "$ENV_DIR"
fi

python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

ROOT=/work/pradip0/Comp/Pynerve/Pynerve
cd $ROOT

echo "=== Building C++ extensions ==="
rm -rf build-python
mkdir build-python
cd build-python

cmake ../python \
    -DPython_EXECUTABLE=$(which python) \
    -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())') \
    -DENABLE_CUDA=ON \
    -DENABLE_PYTORCH=ON \
    -DNERVE_GPU_BASE_ARCHS=80 \
    -DNERVE_SIMD=scalar \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    -GNinja

ninja -j$SLURM_CPUS_PER_TASK

cp python/*.so $ROOT/python/
echo "=== Build complete ==="

cd $ROOT
export PYTHONPATH=python

echo "=== Running tests ==="
python -m pytest tests/python/test_torch_*.py \
    tests/python/test_error_codes.py tests/python/test_types.py \
    tests/python/test_exceptions_core.py tests/python/test_validation_exceptions.py \
    tests/python/test_fallback_classes.py tests/python/test_cache_engine.py \
    tests/python/test_cache_smart_memo.py tests/python/test_formats_files.py \
    tests/python/test_formats_interop.py tests/python/test_formats_auto.py \
    tests/python/test_diagnostics_coverage.py tests/python/test_utils_coverage.py \
    tests/python/test_merge.py tests/python/test_image_utils.py \
    tests/python/test_scalars_edge.py tests/python/test_geometric_edge.py \
    -q --tb=line 2>&1 | tee /work/pradip0/Comp/Pynerve/test_results.log

echo "=== Done ==="
tail -5 /work/pradip0/Comp/Pynerve/test_results.log
