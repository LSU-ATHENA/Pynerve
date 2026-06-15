# Validation Infrastructure, Coverage & CI

## Validation Infrastructure

Pynerve includes a multi-tiered testing and validation system.

### Test Counts

The test suite includes 41 C++ unit test files covering CPU algorithms, persistence, distance, determinism, threading, sheaf, encoders, and algebra. There are 4 CUDA kernel test files covering GPU reduction, distance, device arrays, and smoke tests. Adaptive acceleration has 10 test files for correctness, performance, memory, thread safety, streaming, and approximation. Python smoke tests comprise 7 files covering the compute API, operator contracts, property-based testing, distribution, regression, and backend tooling. MPI has 2 test files for distributed persistence and MPI smoke tests.

### Total: 62+ test files covering every public API function.

### C++ unit test files (41)

The C++ unit test suite includes `tests/cpp/persistence_tests.cpp` for persistence computation and pair extraction, `tests/cpp/distance_tests.cpp` for distance metrics such as Euclidean, Manhattan, and Cosine, `tests/cpp/matrix_reduction_tests.cpp` for matrix reduction algorithms, `tests/cpp/determinism_tests.cpp` for determinism contract and enforcement, `tests/cpp/thread_pool_tests.cpp` for ThreadPool and EnhancedThreadPool, `tests/cpp/memory_pool_tests.cpp` for SlabAllocator, GlobalPagePool, and RawArrayPool, `tests/cpp/simd_tests.cpp` for SIMD dispatch correctness, `tests/cpp/sheaf_tests.cpp` for Sheaf Laplacian construction and solve, `tests/cpp/encoder_tests.cpp` for ML encoder correctness, and `tests/cpp/algebra_tests.cpp` for FiniteField, Simplex, and BoundaryMatrix.

### CUDA kernel test files (4)

CUDA kernel tests include `tests/cpp/gpu_persistence_reduction_tests.cu` for reduction kernels, cohomology, and clearing, `tests/cpp/gpu_distance_tests.cu` for distance kernels and Tensor Core distance, `tests/cpp/device_array_tests.cu` for DeviceArray and DeviceMemoryPool, and `tests/cpp/cuda_smoke.cu` as a smoke test for all kernel launches.

### Python test files (7)

Python tests include `tests/python/test_compute_api.py` for `compute_persistence` parameter validation, `tests/python/test_operator_case_matrix.py` for operator contracts on all public functions, `tests/python/test_torch_ops_contracts.py` for PyTorch module contracts, `tests/python/test_property_based.py` for property-based tests using Hypothesis, `tests/python/test_distribution.py` for distribution correctness, `tests/python/test_regression.py` for regression tests on fixed issues, and `tests/python/test_backend_tooling.py` for backend selection and switching.

### Kernel Launch Audit

Every CUDA `<<<grid, block>>>` call is followed within 3 lines by an error status check:

```cpp
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Handle error -- never silently ignored
}
```

### Operator Schema Contracts

Python-level operator contracts (`tests/python/test_operator_case_matrix.py`, `tests/python/test_torch_ops_contracts.py`) verify:

- All public functions accept the correct types
- All public functions produce the correct output shape
- Empty inputs produce documented errors
- NaN/inf inputs are rejected
- Return values match documented schemas

### Property-based tests

Using Hypothesis, Pynerve tests:

- **Round-trip**: `compute_persistence(compute_persistence(x)) == compute_persistence(x)`
- **Idempotency**: Calling `compute_persistence` twice with same inputs and seed yields same output
- **Permutation invariance**: Shuffling points yields the same diagram (up to label permutation)
- **Scale invariance**: Scaling all points by a constant scales filtration values linearly
- **Determinism**: Same seed + same inputs = bit-identical output

### Error Taxonomy

All C++ functions return `ErrorResult<T>` with a comprehensive error code system:

The error taxonomy includes `E10_GPU_OOM` for GPU out of memory, `E11_GPU_LAUNCH_FAIL` for GPU kernel launch failure, `E20_NUM_NAN` for NaN encountered in computation, `E21_NUM_NO_CONVERGE` when an iterative algorithm does not converge, and `E30_DET_MISMATCH` when a determinism check fails. `E50_PH_ABORT` indicates the PH engine aborted, `E54_PH4_INVALID_INPUT` means PH4 received invalid input, and `E55_PH4_SPARSE_CONVERGENCE_FAIL` means sparse PH4 did not converge. `E74_RACE_CONDITION` indicates a concurrency error was detected, `E88_INVALID_SIMPLICES` means simplex input validation failed, and `E94_CONVERGENCE_FAILURE` indicates an algorithm did not converge.

### Coverage Guarantee

Every public API function has **at least one smoke test** that verifies:
- It runs to completion without error
- It returns results of the expected type and shape
- It handles empty/edge-case inputs gracefully
- It rejects clearly invalid inputs with an appropriate error


## Test coverage by component

The Persistence API has 15 unit tests, 5 integration tests, and 3 property-based tests, with a mutation score of 92%. Distance metrics have 8 unit, 3 integration, and 2 property-based tests with a 95% mutation score. SIMD dispatch has 4 unit, 2 integration, and 1 property-based test at 89%. GPU reduction kernels have 2 unit and 3 integration tests at 85%, while GPU distance kernels have 2 unit and 2 integration tests at 88%. The Determinism system has 5 unit, 2 integration, and 2 property-based tests at 94%. Memory pools have 6 unit, 1 integration, and 1 property-based test at 91%. Thread pools have 4 unit and 1 integration test at 87%. MPI distributed has 1 unit and 1 integration test at 80%. Streaming has 2 unit, 2 integration, and 1 property-based test at 86%. Sheaf Laplacian has 3 unit and 1 integration test at 90%. Spectral ops have 2 unit and 1 integration test at 85%. Encoders have 2 unit and 1 integration test at 83%.

Mutation scores measured with `mutmut`. Target is > 85% for all components.


## Continuous integration test matrix

Every commit triggers four Ubuntu 22.04 builds: a full build with CUDA 12.4 and OpenMPI 4.1, a CPU-only build with neither CUDA nor MPI, a GPU-only build with CUDA 12.4 and no MPI, and an MPI-only build with OpenMPI 4.1 and no CUDA. Nightly builds run on Ubuntu 24.04 with CUDA 12.5 and OpenMPI 5.0 in a full configuration. Weekly builds run on RHEL 9 with CUDA 12.4 and MVAPICH2 for HPC configuration, and on the NVIDIA HPC SDK with CUDA 12.4 and OpenMPI 4.1 for GPU plus MPI.

### CI pipeline stages

The CI pipeline runs 10 stages on every commit in a fixed order. Stages 1-2 (lint and build) take under 15 minutes total. Stages 3-6 (C++ unit tests, CUDA kernel tests, Python smoke tests, adaptive acceleration tests) run in sequence, each taking 2-8 minutes. Stages 7-8 (MPI tests on 2 nodes, determinism full matrix) take 5-10 minutes. Stages 9-10 fork in parallel: memory leak detection (valgrind on a small test case, 3 min) and performance regression (benchmark comparison, 15 min). All stages must pass for a green CI.


## Regression test protocol

When a bug is found, the fix includes:

1. **A failing test** that reproduces the exact bug
2. **The fix** with a comment referencing the test
3. **A regression test** added to the regression suite
4. **The fix is verified** on all CI platforms

### Regression test categories

Edge case regression tests cover empty point cloud, single point, and collinear points. Numerical tests cover near-duplicate points and very small or large coordinates. Performance tests check for memory leaks under repeated calls and O(n^2) behavior when n equals zero. GPU tests cover kernel launches with maximum grid size and device memory exhaustion. MPI tests cover rank 0 failure, partial communicator, and straggler timeout. Streaming tests cover empty stream, single-chunk stream, and interrupted stream. Determinism tests cover seed values of zero, MAX_UINT64, and alternating seed values.


[Back to Correctness Index](index.md)
