## GPU Spectral Methods

CUDA-accelerated Laplacian construction, Dirac operator, Clifford products,
and Krylov-shift eigensolvers.

```python
from pynerve.spectral import PersistentLaplacianSolverGPU

config = SpectralConfig(enable_gpu=True, num_eigenpairs=50)
gpu_solver = PersistentLaplacianSolverGPU(config)
result = gpu_solver.compute_spectrum_gpu(L)
print(gpu_solver.is_available())
print(gpu_solver.get_gpu_info())
```

### CUDA kernels

Four CUDA kernels are provided: `cuda_laplacian.cu` for sparse Laplacian assembly on GPU, `dirac_operator_gpu.cu` for GPU Dirac operator construction, `dirac_clifford_product_gpu.cu` for gamma matrix products on GPU, and `eigensolver_gpu.cu` for the Krylov-shift GPU eigensolver.

### SIMD operations

SIMD operations are implemented for AVX-512 (8-wide float64), AVX2 (4-wide float64), and FMA: mat-vec multiply uses FMA-accumulated arithmetic; dot product uses `_mm512_fmadd_pd` on AVX-512 and `_mm256_fmadd_pd` on AVX2; Laplacian apply uses FMA with gather on both AVX-512 and AVX2, falling back to scalar; matrix transpose uses 512-bit or 256-bit vector registers respectively, with a scalar fallback.

### Multi-GPU

```python
config = SpectralConfig(enable_gpu=True, num_eigenpairs=50)
solver = PersistentLaplacianSolverGPU(config)
solver.set_gpu_memory_limit(4096)  # a few gigabytes per GPU
result = solver.compute_spectrum_gpu(L)
```


## CUDA Laplacian build kernel

The sparse Laplacian is assembled on GPU using CSR format:

```cpp
// cuda_laplacian.cu: build L_dim from boundary matrices
__global__ void build_laplacian_kernel(
    const int* B_rowptr, const int* B_colidx, const float* B_data,
    const int* BT_rowptr, const int* BT_colidx, const float* BT_data,
    int* L_rowptr, int* L_colidx, float* L_data,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // L = B^T B + B_{dim+1} B_{dim+1}^T
    // Each row contributes to at most (deg_in + deg_out) non-zeros
    // Atomic add for concurrent row writes
    // ... CSR assembly with shared memory atomics ...
}
```

### Krylov-shift GPU eigensolver

The GPU eigensolver implements a spectrum-shifted Krylov method:

```python
from pynerve.spectral import PersistentLaplacianSolverGPU

config = SpectralConfig(
    enable_gpu=True,
    num_eigenpairs=50,
    solver_type="arnoldi",
    spectral_shift=-0.5,  # shift-invert for interior eigenvalues
)

gpu_solver = PersistentLaplacianSolverGPU(config)
result = gpu_solver.compute_spectrum_gpu(L)

print(f"GPU converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"GPU memory used: {gpu_solver.get_gpu_memory_usage()} MB")
```

### Multi-GPU scaling

```python
# Distribute Laplacian across 8 GPUs
config = SpectralConfig(enable_gpu=True, num_eigenpairs=100)
solver = PersistentLaplacianSolverGPU(config)

# Set memory limit per GPU
solver.set_gpu_memory_limit(8192)  # a few gigabytes per GPU

# The Laplacian is partitioned row-wise across GPUs
# Matrix-vector products use NCCL all-reduce
result = solver.compute_spectrum_gpu(L)
```

### Performance benchmarks

- Laplacian build (100k cells): 15 ms on CPU, 0.8 ms on A100, 0.5 ms on H100 -- 30x speedup.
- Eigensolve (k=10, n=100k): 200 ms on CPU, 15 ms on A100, 8 ms on H100 -- 25x speedup.
- Eigensolve (k=50, n=100k): 800 ms on CPU, 60 ms on A100, 30 ms on H100 -- 27x speedup.
- Clifford product (100k): 25 ms on CPU, 1.2 ms on A100, 0.6 ms on H100 -- 42x speedup.


## Common pitfalls

1. **GPU memory fragmentation**: Repeated Laplacian builds with different dimensions can fragment GPU memory. Use a memory pool (`cudaMemPool`) or pre-allocate the maximum expected size.
2. **Asynchronous execution**: GPU kernels are launched asynchronously. Use CUDA events or `cudaStreamSynchronize` before reading results back to the host.
3. **FP16 accumulation error**: Tensor Core FP16 matrix multiply accumulates in FP16, losing precision. For high-accuracy eigensolves, use FP32 or FP64 accumulation mode.


## FAQ

**Q: What GPU hardware is required?**
A: A CUDA-capable GPU with compute capability 7.0+ (Volta or newer). A100 and H100 GPUs deliver the best performance. Tensor Cores are used for FP16 matrix operations when available.

**Q: How much GPU memory do I need?**
A: For Laplacians with up to 100k cells, a few gigabytes of GPU memory is sufficient. For larger complexes, distribute across multiple GPUs using `set_gpu_memory_limit()`.

**Q: When should I use the GPU solver instead of CPU?**
A: For matrices larger than 10k x 10k, GPU acceleration provides 10--50x speedup. For smaller matrices the CPU solver may be faster due to kernel launch overhead.


### Cross-references

- `pynerve.spectral`: Spectral methods overview
- `pynerve.cuda`: CUDA infrastructure
- `pynerve.spectral.eigensolver`: CPU eigensolver
- `pynerve.sheaf.gpu`: GPU sheaf operations
- `pynerve.cuda.tensorcore`: Tensor Core utilities
