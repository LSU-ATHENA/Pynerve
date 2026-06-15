## GPU Sheaf Operations

CUDA-accelerated sheaf Laplacian construction, Tensor Core support for
FP16/BF16/FP8, and multi-GPU distributed stalk computation.

```python
from pynerve.sheaf.gpu import benchmark_sheaf_gpu

bm = benchmark_sheaf_gpu(num_stalks=10000, stalk_dim=64)
# bm.cpu_time_ms, bm.gpu_time_ms, bm.speedup
```

### Tensor Core accelerated

```python
from pynerve.sheaf.gpu.tensorcore import TensorCoreSheafLaplacian

tc = TensorCoreSheafLaplacian(num_gpus=1)
tc.initialize()
tc.compute(stalk_matrix, restriction_matrix, output, num_stalks, num_restrictions)
# WMMA 16x16x16 tiles (FP16), auto FP8 on Blackwell TMA
print(tc.get_last_compute_time_ms())
print(tc.get_effective_tflops())
```

### SIMD stalk operations

Auto-dispatches to best SIMD path via CPUID at call time.

```python
from pynerve.sheaf import SIMDStalkOperations

SIMDStalkOperations.add_stalks(a, b, result)        # AVX-512: 8-wide f64
SIMDStalkOperations.scale_stalk(stalk, scalar, result)
dot = SIMDStalkOperations.dot_product(a, b)          # FMA-accumulated
SIMDStalkOperations.normalize_stalk(stalk)
```

- **Stalk add**: 8-wide f64 FMA (AVX-512), 4-wide f64 FMA (AVX2), 2-wide f64 (SSE4.1)
- **Stalk scale**: 8-wide mul (AVX-512), 4-wide mul (AVX2), scalar (SSE4.1)
- **Dot product**: FMA + reduce (AVX-512 and AVX2), scalar (SSE4.1)
- **Normalize**: sqrt + div (AVX-512 and AVX2), scalar (SSE4.1)

### Multi-GPU

Distributed stalk computation across GPUs via the `SheafEngine`.

```python
config = pynerve.sheaf.SheafConfig(
    num_stalks=100000,
    stalk_dimension=64,
    use_multi_gpu=True,
)
engine = pynerve.sheaf.SheafEngine(config)
engine.build_sheaf(stalk_positions, stalk_dimensions)
engine.compute_cohomology(cocycle)
```

GPU Tensor Core cost: O(n * d^2 / 16) via WMMA 16x16 tiled.


## Tensor Core sheaf Laplacian

The Tensor Core kernel uses WMMA (Warp Matrix Multiply-Accumulate) for FP16 matrix operations:

```cpp
// WMMA 16x16x16 tile for sheaf Laplacian assembly
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                       half, nvcuda::wmma::row_major> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                       half, nvcuda::wmma::col_major> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                       float> c_frag;

// Each tile computes L_ij += F_ik * F_jk^T for restriction maps F
load_matrix_sync(a_frag, restriction_map, 16);
load_matrix_sync(b_frag, restriction_map_transpose, 16);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(laplacian_block, c_frag, 16);
```

### Memory layout optimization

```cpp
// Interleaved stalk storage for coalesced access
struct InterleavedStalks {
    half* data;  // [num_stalks, stalk_dim] interleaved
    int stride;  // padded stride (multiple of 16 for Tensor Core)
};
```

### Multi-GPU scaling

```python
from pynerve.sheaf.gpu import MultiGPUSheafEngine

engine = MultiGPUSheafEngine(num_gpus=4)
config = pynerve.sheaf.SheafConfig(
    num_stalks=100000,
    stalk_dimension=64,
    use_multi_gpu=True,
)

# Each GPU handles a partition of the stalks
# Restriction maps at partition boundaries are communicated via NCCL
result = engine.build_sheaf_laplacian(config)
```

### CUDA kernel launch configuration

- **Sheaf Laplacian build**: grid = ceil(num_edges * d^2 / 256), block = 256, shared memory = single-digit kilobytes
- **Restriction apply**: grid = ceil(num_stalks / 128), block = 128, shared memory = single-digit kilobytes
- **Cohomology compute**: grid = ceil(num_cells / 64), block = 64, shared memory = tens of kilobytes
- **Stalk add (SIMD)**: CPU only, no shared memory


## Performance benchmarks

```python
from pynerve.sheaf.gpu import benchmark_sheaf_gpu

for num_stalks in [1000, 10000, 100000]:
    bm = benchmark_sheaf_gpu(num_stalks=num_stalks, stalk_dim=64)
    print(f"Stalks={num_stalks}: CPU={bm.cpu_time_ms:.1f}ms, "
          f"GPU={bm.gpu_time_ms:.1f}ms, speedup={bm.speedup:.1f}x")
```

Expected speedups on H100 vs 96-core AMD EPYC:
- 10k stalks, d=8: 5-10x
- 100k stalks, d=16: 10-20x
- 1M stalks, d=8: 20-40x (memory-bound, GPU HBM bandwidth)


## Common pitfalls

1. **Tensor Core alignment**: Stalk dimensions must be multiples of 16 for FP16 WMMA tiles. Pad stalks with zeros if d is not a multiple of 16.
2. **Multi-GPU boundary communication**: Restriction maps at partition boundaries require NCCL all-gather. For small sheaves (<10k stalks), single-GPU is faster.
3. **FP8 underflow**: FP8 (e4m3) has limited dynamic range. After each Laplacian assembly step, scale the accumulator to avoid underflow.


## FAQ

**Q: When should I use multi-GPU vs single GPU?**
A: Single GPU is faster for small sheaves (fewer than 10k stalks) because NCCL boundary communication overhead outweighs parallelization benefits. Multi-GPU scales well for large sheaves (100k+ stalks) where each GPU handles a partition of the stalks and only boundary restriction maps are communicated.

**Q: What precision should I use for Tensor Core operations?**
A: FP16 is recommended for general use with good accuracy and 2x throughput vs FP32. FP8 (e4m3) offers 4x throughput but requires careful scaling to avoid underflow due to limited dynamic range. BF16 is useful when exponent range matters more than precision.

**Q: How do I handle stalk dimensions that are not multiples of 16 on Tensor Cores?**
A: Pad stalks with zeros to the next multiple of 16. The WMMA 16x16x16 tile requires dimensions divisible by 16 for FP16 operations. The padded zeros contribute correctly to the Laplacian assembly since they do not affect restriction map computations.


### Cross-references

- `pynerve.sheaf`: Sheaf module overview
- `pynerve.cuda`: CUDA infrastructure
- `pynerve.spectral.gpu`: GPU spectral methods
- `pynerve.cuda.tensorcore`: Tensor Core utilities
