# SIMD dispatch

Pynerve detects CPU capabilities at runtime and dispatches to the optimal SIMD backend. AVX-512 requires Intel Skylake-X or AMD Zen 4 and delivers 8 to 16 times speedup versus scalar code. AVX2 requires Intel Haswell or AMD Excavator and delivers 4 to 8 times speedup. SSE4.1 requires Intel Penryn or AMD Bulldozer and delivers 2 to 3 times speedup. A scalar fallback works on any x86-64 CPU at 1x speed. Runtime dispatch is implemented via a CPUID-based selector with no compile-time flags required, so a single binary runs optimally on any x86-64 CPU.

SIMD is used throughout the pipeline. Pairwise distance operates at 256 to 512-bit width with FMA, reduce, and sqrt operations. Filtration sort uses 128 to 256-bit wide vectorized key-value sort. Boundary matrix operations use 256-bit bulk XOR for Z2 coefficients. Diagram vectorization uses 256 to 512-bit grid evaluation and convolution. Euclidean distance for float32 uses 512-bit (16-wide on AVX-512) or 256-bit (8-wide on AVX2) with FMA instructions, while float64 uses 512-bit (8-wide) or 256-bit (4-wide). Dot product uses FMA accumulate at 512 or 256-bit. Spectral ops and sheaf Laplacian use 256 to 512-bit FMA with gather and scatter. Persistence image uses 512-bit FMA for weighted Gaussian.

Each operation maps to specific intrinsic sets. Euclidean float32 uses `_mm512_fmadd_ps`, `_mm512_reduce_add_ps`, and `_mm512_sqrt_ps` on AVX-512, `_mm256_fmadd_ps`, `_mm256_hadd_ps`, and `_mm256_sqrt_ps` on AVX2, and `_mm_dp_ps` with `_mm_sqrt_ps` on SSE4.1. Euclidean float64 uses corresponding packed double variants on AVX-512 and AVX2, with scalar fallback on SSE4.1. Manhattan float32 uses absolute value and reduce instructions on AVX-512 and AVX2, scalar on SSE4.1. Cosine uses FMA plus divide. Filtration sort uses AVX-512 bitonic with permute or AVX2 bitonic sorting networks. Column XOR uses 512, 256, or 128-bit XOR intrinsics. Spectral operations combine FMA with gather instructions on AVX-512 and AVX2. Image vectorization uses FMA and exponent intrinsics on AVX-512, with FMA only on AVX2.

```python
# SIMD dispatch is automatic -- no user configuration needed.
# Pynerve probes CPUID on first use and selects the fastest path.
result = pynerve.compute_persistence(points, max_dim=2)
```

[Back to index](index.md)
