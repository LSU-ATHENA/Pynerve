# Bandwidth-bound vs compute-bound operations

Different operations have different bottleneck characteristics. The distance matrix for n of 10,000 or fewer is memory bandwidth bound by DRAM read of point data, with GPU providing 2 to 4 times benefit via Tensor Cores. For n over 10,000 it becomes compute bound by FP arithmetic, with GPU providing 4 to 10 times benefit. Sparse column reduction is memory bandwidth bound by CSR index traversal with 2 to 3 times GPU benefit, while dense column reduction is compute bound by XOR and pivot search with 3 to 5 times GPU benefit. Filtration sort is memory bandwidth bound by key-value moves with 2 times GPU benefit. Persistence image is compute bound by Gaussian evaluation with 5 to 10 times GPU benefit. Bottleneck distance is compute bound by the matching algorithm with 3 to 5 times GPU benefit.

Identifying whether your workload is bandwidth-bound or compute-bound helps choose between CPU and GPU:

### Floating-point operation throughput

Floating-point operation throughput varies dramatically between CPU and GPU. FP32 FMA achieves roughly 100 GFLOPS on a single AVX-512 core, up to a few TFLOPS across 32 cores, and thousands of TFLOPS on an H100 Tensor Core. FP64 FMA achieves roughly 50 GFLOPS on a single core, over 1 TFLOPS across 32 cores, and tens of TFLOPS on H100. FP16 and FP8 FMA are only available on GPU, reaching thousands of TFLOPS.

### Cache blocking strategy

Pynerve uses cache-aware blocking for distance matrix computation:

```cpp
// Block size tuned for L2 cache
constexpr size_t L2_BLOCK = 64;  // points per block

for (size_t i0 = 0; i0 < n; i0 += L2_BLOCK) {
    for (size_t j0 = i0; j0 < n; j0 += L2_BLOCK) {
        // Block (i0, j0) fits in L2 cache
        compute_distance_block(i0, j0, L2_BLOCK);
    }
}
```

This reduces L2 cache misses by 40% for n > 10,000 compared to naive triple-loop distance computation.

### Real-world throughput for persistence

At n of 1,000, CPU throughput is hundreds of thousands of points per second while GPU throughput is hundreds of thousands, for a ratio of roughly 0.2 times as GPU is transfer bound. At n of 10,000, CPU throughput is hundreds of thousands and GPU throughput is hundreds of thousands, for a 2.5 times GPU advantage. At n of 100,000, CPU throughput is tens of thousands while GPU throughput is hundreds of thousands, for a 10 times advantage. At n of 1,000,000 in sparse mode, CPU throughput is tens of thousands and GPU throughput is hundreds of thousands, for a 13 times advantage.

[Back to index](index.md)
