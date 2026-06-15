# Determinism

## Compiler flags

GPU persistence is built with these `nvcc` flags to ensure deterministic floating-point arithmetic:

```cmake
target_compile_options(nerve_core PRIVATE
    --fmad=false        # disable fused multiply-add
    --prec-div=true     # precise division
    --prec-sqrt=true    # precise square root
    --ftz=false         # preserve subnormals
)
```

## Fixed-tree reductions

Reduction ordering is fixed at compile time. The warp-shuffle reduction uses a deterministic butterfly pattern, and cross-warp accumulation uses a shared-memory tree with a fixed reduction order. No atomic operations are used, so results are bitwise identical across runs and GPU architectures.

### Fixed-tree reduction code structure

```cpp
// Conceptual structure of the deterministic GPU reduction
template <int kBlockSize>
__device__ double warpReduceSum(double val) {
    // Deterministic butterfly: fixed XOR pattern
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}

template <int kBlockSize>
__device__ double blockReduceSum(double val) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;

    // Hierarchical tree: fixed reduction order
    if constexpr (kBlockSize > 512) {
        if (tid < 512) shared[tid] = val + shared[tid + 512];
        __syncthreads();
    }
    if constexpr (kBlockSize > 256) {
        if (tid < 256) shared[tid] = val + shared[tid + 256];
        __syncthreads();
    }
    if constexpr (kBlockSize > 128) {
        if (tid < 128) shared[tid] = val + shared[tid + 128];
        __syncthreads();
    }
    if (tid < 64) shared[tid] = val + shared[tid + 64];
    __syncthreads();
    // Warp shuffle for final reduction
    double sum = shared[tid];
    sum = warpReduceSum<kBlockSize>(sum);
    return sum;
}
```

## RFA (Reproducible Floating-Point Accumulation) -- opt-in

For **cross-GPU reproducibility** (same computation producing identical results across different GPU counts or topologies), enable RFA:

```python
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    error_tolerance=1e-12,
)
```

RFA adds 20-30% overhead. It works by:
1. Assigning each column a global index
2. Accumulating reductions in a fixed global order using deterministic tree reduction
3. Broadcasting partial sums across ranks with sorted merge

Without RFA, results are deterministic for a fixed GPU topology but may differ when the GPU count changes.


<- [Back to GPU Acceleration index](index.md)
