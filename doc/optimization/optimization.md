# Optimization

Memory-efficient compact summaries of persistence diagrams, streaming PH
optimization, GPU-accelerated SGD/Adam optimizers, and SIMD gradient clipping.
Reduces persistence diagram size while preserving topological features.

```cpp
#include <nerve/optimization/component_optimizations.hpp>

using namespace nerve::optimization;

// Compact summary: reduce point cloud to 128-byte topological summary
AcceleratedCompactSummaries::SummaryConfig scfg;
scfg.summary_size = 128;
scfg.enable_avx512 = true;

AcceleratedCompactSummaries summarizer(scfg);
CallContract contract{/*time_budget_ms=*/100.0, /*strict=*/true, "summary"};

auto summary = summarizer.computeSummary(points, contract);
// summary.betti_numbers, summary.top_lifetimes,
// summary.persistence_entropy, summary.laplacian_top4
```


## Components

The optimization module includes four components: **[compact_summary.md](compact_summary.md)** provides AcceleratedCompactSummaries for 128-byte topological features; **[streaming_ph.md](streaming_ph.md)** implements memory-efficient streaming persistence with error bounds; **[gpu.md](gpu.md)** covers GPU SGD/Adam optimizer steps and gradient clipping; and **[simd.md](simd.md)** covers AVX-512 gradient clipping and L2 norm.


## Hardware optimizations

Utilities in `src/include/nerve/optimization/hardware_optimizations.hpp`:

```cpp
namespace nerve::optimization {

// Prefetching
enum class PrefetchLevel { L1, L2, L3, RAM };
template <PrefetchLevel LEVEL>
void prefetch(const void* ptr);
template <PrefetchLevel LEVEL>
void prefetchWrite(const void* ptr);

// Alignment
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t SIMD_ALIGNMENT = 64;
#define ALIGN_CACHE_LINE alignas(64)
#define ALIGN_SIMD alignas(64)

// Branchless operations
double branchlessMax(double a, double b);
double branchlessMin(double a, double b);
double branchlessAbs(double x);

// Cache control
void streamStore(double* dst, double value);
void cacheFlush(const void* ptr);
void memoryFence();

// CPU feature detection
struct CpuFeatures {
    bool has_sse2, has_avx, has_avx2, has_avx512f;
    bool has_fma, has_bmi2, has_popcnt, has_lzcnt;
    static CpuFeatures detect();
};

// NUMA utilities
void* numaAllocOnNode(size_t size, int node);
void* numaAllocInterleaved(size_t size);
int getCurrentNumaNode();

}
```


## Complexity

CompactSummary complexity is O(n * d + d^2 + k^3) where n is the number of points, d is the dimension, and k is the number of top features. StreamingPH (witness) is O(n * L + L^3) per window where L is the number of landmarks (much smaller than n). StreamingPH (sketch) is O(n * k + k^3) per window where k is the sketch dimension. GPU SGD step and GPU Adam step are both O(n) in a single CUDA kernel each. SIMD clip gradients is O(n/8) using AVX-512.



## Practical guidance

### When to use each optimization

Compact summaries are best for real-time monitoring and streaming, producing a fixed 128-byte output per window. Streaming PH handles out-of-core persistence with bounded memory for infinite streams. GPU optimizer accelerates large model training with a 10-50x speedup for SGD and Adam. SIMD clipping provides 8x speedup over scalar for gradient norm control.

### Common pitfalls

1. **Compact summary accuracy**: The 128-byte summary captures only the top 8 lifetimes and 4 Laplacian eigenvalues. For detailed analysis, use the full persistence diagram instead.
2. **Streaming PH drift**: Error accumulates over long streams. The `ErrorTracker` provides confidence bounds. Reset the stream periodically if error exceeds the budget.
3. **GPU optimizer memory**: The Adam step stores first and second moment estimates (2x model size). For very large models, use SGD to halve memory.
4. **SIMD alignment**: `simdClipGradients` requires 64-byte alignment for maximum throughput. Use `ALIGN_SIMD` on gradient buffers.

### Performance tuning for CompactSummary

```python
from pynerve.optimization import AcceleratedCompactSummaries

# Fastest configuration (real-time)
fast_cfg = SummaryConfig(
    summary_size=128,
    enable_avx512=True,
    use_per_thread_allocators=True,
    precomputeHeavyReductions=True,
)

# Memory-efficient configuration
mem_cfg = SummaryConfig(
    summary_size=128,
    enable_avx512=False,
    use_per_thread_allocators=False,
    thread_allocator_size=256 * 1024,  # 256 KB per thread
)

# Profile
from pynerve.optimization import benchmark_summary
bm = benchmark_summary(num_points=10000, config=fast_cfg)
print(f"Summary time: {bm.mean_time_ms:.2f}ms")
```

### Hardware optimization macros

```cpp
// Use cache line alignment for hot data structures
ALIGN_CACHE_LINE double gradient_buffer[4096];

// Prefetch next batch while processing current
prefetch<PrefetchLevel::L1>(next_batch);

// Branchless min/max for gradient clipping
double clipped = branchlessMax(branchlessMin(grad, max_norm), -max_norm);

// Non-temporal store to avoid cache pollution
streamStore(output, value);
```

### NUMA-aware allocation

```cpp
// On multi-socket systems, allocate memory close to the CPU
int node = getCurrentNumaNode();
double* local_memory = (double*)numaAllocOnNode(size, node);

// For interleaved allocation across all nodes
double* interleaved = (double*)numaAllocInterleaved(size);
```


## Integration with pipeline

```python
from pynerve.optimization import AcceleratedCompactSummaries
from pynerve.optimization import GPUOptimizer

# Summarize stream windows, optimize with GPU
summarizer = AcceleratedCompactSummaries(cfg)
optimizer = GPUOptimizer(algorithm="adam", learning_rate=1e-3)

for window in sliding_windows:
    summary = summarizer.computeSummary(points_in_window, contract)
    features = summary_to_features(summary)
    loss = model(features)
    loss.backward()
    optimizer.step(model.params, model.grads)
```


## FAQ

**Q: How does error tracking work in StreamingPH?**
A: The `ErrorTracker` compares the approximate result with occasional exact computations. It maintains a confidence interval on the error and signals when the error exceeds the budget, triggering a recomputation or coarsening.

**Q: Can I use CompactSummary with non-VR filtrations?**
A: Yes. The summary extracts topological features from any persistence diagram. The `precomputeHeavyReductions` flag computes PCA-like bases that work for arbitrary filtrations, not just VR.

**Q: What precision do the GPU optimizers use?**
A: The default is FP32 gradients and parameters. Enable `enable_mixed_precision=True` to use FP16 for gradients (halves memory, minor accuracy loss). The `AcceleratedGpuPrimitives::GPUConfig` controls precision per operation.


### Cross-references

- `pynerve.memory`: Memory pools used by optimizers
- `pynerve.core`: Thread pool, SIMD operations
- `pynerve.cuda`: GPU optimizer kernels
- `pynerve.ml`: ML pipeline using optimized summaries
- `pynerve.specialized`: Zigzag persistence for streaming data
