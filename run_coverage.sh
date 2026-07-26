#!/bin/bash
#SBATCH --job-name=pynerve-cov
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/pradip0/Comp/Pynerve/cov_output_%j.log
#SBATCH --error=/work/pradip0/Comp/Pynerve/cov_error_%j.log

set -e

echo "=== Starting Pynerve coverage run ==="
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
    pip install pybind11 numpy scipy scikit-learn numba hypothesis matplotlib pytest pytest-timeout pytest-xdist pytest-benchmark coverage pytest-cov ninja triton psutil ripser gudhi dionysus
    conda install -c conda-forge eigen nlohmann_json -y
else
    source activate "$ENV_DIR"
fi

# Ensure Eigen3 and missing packages are available (may have been added after initial env creation)
conda list eigen 2>/dev/null | grep -q eigen || conda install -c conda-forge eigen -y 2>/dev/null || true
conda list nlohmann_json 2>/dev/null | grep -q nlohmann_json || conda install -c conda-forge nlohmann_json -y 2>/dev/null || true
# Re-install core deps silently; skip triton (leave the pre-installed version alone)
pip install -q scipy scikit-learn numba hypothesis matplotlib pytest-cov 2>/dev/null || true
# psutil is required for diagnostics tests — fail if it can't install
pip install -q psutil
# Optional benchmark comparison packages (may fail on some systems)
pip install -q ripser gudhi dionysus 2>/dev/null || echo "Warning: some benchmark deps could not be installed"

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
    -DNERVE_ENABLE_CUDA_COMPONENTS=ON \
    -DENABLE_PYTORCH=ON \
    -DNERVE_GPU_BASE_ARCHS=80 \
    -DENABLE_EIGEN=ON \
    -DNERVE_SIMD=scalar \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$CONDA_PREFIX" \
    -GNinja

ninja -j$SLURM_CPUS_PER_TASK

cp python/*.so $ROOT/python/ 2>/dev/null || true
echo "=== Build complete ==="

# Diagnostic: check which .so files were copied and whether the core extension loads
echo "=== .so files in python/ ==="
ls -la $ROOT/python/*.so 2>&1
echo "=== Core extension import test ==="
cd $ROOT
export PYTHONPATH=python
python -c "
try:
    import pynerve_internal
    print('pynerve_internal: OK', getattr(pynerve_internal, '__file__', '?'))
except Exception as e:
    print('pynerve_internal: FAILED', repr(e))
try:
    import pynerve
    print('pynerve._core:', getattr(pynerve, '_core', None))
except Exception as e:
    print('pynerve import: FAILED', repr(e))
" 2>&1 || true

cd $ROOT
export PYTHONPATH=python

echo "=== Running full test suite with coverage ==="
# Use pytest-cov so individual test crashes don't lose coverage data.
# --continue-on-collection-errors prevents missing-dep aborts.
rm -f .coverage .coverage.*
python -m pytest tests/python/ \
    -q --tb=line -p no:warnings \
    --timeout=120 \
    --continue-on-collection-errors \
    -n 4 --dist=loadscope \
    --ignore=tests/python/test_memory_spectral_sheaf.py \
    --ignore=tests/python/test_memory_pool.py \
    --ignore=tests/python/test_triton_gpu_kernels.py \
    --cov=python/pynerve --cov-branch --cov-report= \
    2>&1 | tee /work/pradip0/Comp/Pynerve/coverage_test_results.log
# Run triton GPU kernel tests serially (needs exclusive GPU access)
python -m pytest tests/python/test_triton_gpu_kernels.py \
    -q --tb=line -p no:warnings \
    --timeout=120 \
    --cov=python/pynerve --cov-branch --cov-append --cov-report= \
    2>&1 | tee -a /work/pradip0/Comp/Pynerve/coverage_test_results.log

echo "=== Generating coverage report ==="
python -m coverage report --show-missing --skip-covered 2>&1 | tee /work/pradip0/Comp/Pynerve/coverage_report.log || true

# Also generate a JSON summary for easy parsing
python -m coverage json -o /work/pradip0/Comp/Pynerve/coverage.json 2>&1 || true

echo "=== Extracting summary line ==="
grep "^TOTAL" /work/pradip0/Comp/Pynerve/coverage_report.log || \
    tail -5 /work/pradip0/Comp/Pynerve/coverage_report.log

echo "=== Top 30 least-covered modules ==="
python -m coverage report --skip-covered 2>&1 | head -35

echo "=== Done ==="
