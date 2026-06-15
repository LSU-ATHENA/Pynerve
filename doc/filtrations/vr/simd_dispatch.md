# Distance computation optimization: SIMD dispatch

Distance computation is the dominant cost for $n > 10^4$. Pynerve uses a
**runtime SIMD dispatch** system that probes CPU capabilities at startup
and selects the fastest available kernel:

### Dispatch table

The dispatch system selects the fastest available kernel based on CPU capabilities. AVX-512 (BW/CD) delivers an 8 to 12 times speedup over scalar on Ice Lake, Skylake-X, and Sapphire Rapids microarchitectures. AVX-512 (F/CD) achieves 7 to 10 times speedup on Knights Landing and Cascade Lake. AVX2 with FMA provides 4 to 6 times speedup on Haswell, Broadwell, and Zen 1/2/3/4. AVX offers 2 to 3 times speedup on Sandy Bridge, Ivy Bridge, and Zen+. SSE4.1 delivers 2 to 2.5 times speedup on Nehalem, Westmere, and Bulldozer. SSE2 serves as the fallback with baseline performance on all x86-64 CPUs.

### Dispatch logic

```c
// Pseudocode for the dispatch mechanism
simd_kernel select_distance_kernel() {
    if (cpu_has_avx512())   return &avx512_euclidean;
    if (cpu_has_avx2_fma()) return &avx2_fma_euclidean;
    if (cpu_has_avx())      return &avx_euclidean;
    if (cpu_has_sse41())    return &sse41_euclidean;
    return &sse2_euclidean;
}
```

The dispatch is entirely automatic -- no compile flags, no manual tuning.
Each kernel processes 4-16 distance computations per instruction via
SIMD vectorization.

### Blocked distance computation

For $n > 5000$, distances are computed in blocks of $B \times B$ where
$B$ is chosen so that two $B \times d$ tiles fit in L2 cache:

```
for i in range(0, n, B):
    for j in range(i, n, B):
        for ii in range(i, min(i+B, n)):
            for jj in range(j, min(j+B, n)):
                d[ii][jj] = simd_euclidean(points[ii], points[jj])
```

Block sizes depend on the SIMD width and cache geometry:
- AVX-512: $B = 256$ (2 * 256 * 4 bytes * d fits in 1-megabytes L2)
- AVX2: $B = 128$
- SSE: $B = 64$


<- [Vietoris-Rips Overview](index.md)
