# SIMD Dispatch

### CPUFeatureDetector

Runtime CPUID-based detection of SIMD capabilities. Queries the CPU once at first access and caches results.

```cpp
namespace nerve::cpu::simd;

bool has_avx512 = CPUFeatureDetector::hasAVX512F();   // AVX-512 Foundation
bool has_avx512vl = CPUFeatureDetector::hasAVX512VL(); // AVX-512 Vector Length
bool has_avx512bw = CPUFeatureDetector::hasAVX512BW(); // AVX-512 Byte/Word
bool has_avx2 = CPUFeatureDetector::hasAVX2();
bool has_fma = CPUFeatureDetector::hasFMA();
int width = CPUFeatureDetector::getMaxSIMDWidth();     // 512, 256, or 128
```

### Dispatch Pattern

SIMD operations use a two-level dispatch:

1. **Build-time**: `NERVE_SIMD` (`auto`, `avx2`, `avx512`, `scalar`) controls compile flags
2. **Runtime**: `CPUFeatureDetector` selects the best path at each call site

```cpp
// Typical dispatch idiom
if (cpu::CPUFeatureDetector::instance().hasAVX512F()) {
    // AVX-512 path (512-bit vectors, 8 doubles / 16 floats)
} else if (cpu::CPUFeatureDetector::instance().hasAVX2()) {
    // AVX2 path (256-bit vectors, 4 doubles / 8 floats)
} else {
    // SSE4.1 or scalar fallback
}
```

### SIMD-Optimized Operations

SIMD acceleration is available across three instruction set levels. For Euclidean distance on float32, AVX-512 uses `_mm512_fmadd_ps` with 16-wide vectors, AVX2 uses `_mm256_fmadd_ps` with 8-wide vectors, and SSE4.1 uses `_mm_dp_ps` with 4-wide vectors. For float64 Euclidean distance, AVX-512 uses `_mm512_fmadd_pd` with 8-wide vectors, AVX2 uses `_mm256_fmadd_pd` with 4-wide vectors, and SSE4.1 falls back to scalar. Dot product follows the same pattern with `_mm512_fmadd_ps/pd` on AVX-512, `_mm256_fmadd_ps/pd` on AVX2, and scalar on SSE4.1. Vectorized square root uses `_mm512_sqrt_ps/pd` on AVX-512, `_mm256_sqrt_ps/pd` on AVX2, and `_mm_sqrt_ps/pd` on SSE4.1. Matrix reduction operates on 512-bit columns with AVX-512, 256-bit columns with AVX2, and scalar with SSE4.1. Filtration uses 512-bit sorting networks on AVX-512, 256-bit on AVX2, and scalar on SSE4.1. Spectral operations use FMA plus gather on both AVX-512 and AVX2, and scalar on SSE4.1. The Sheaf Laplacian uses FMA plus scatter on AVX-512 and AVX2, and scalar on SSE4.1. Persistence vectorization uses `_mm512_fmadd_pd` on AVX-512, `_mm256_fmadd_pd` on AVX2, and scalar on SSE4.1.

### Dispatch Control

```bash
# Force AVX-512 at build time
cmake -DNERVE_SIMD=avx512 ..

# Force scalar fallback (for testing)
cmake -DNERVE_SIMD=scalar ..

# Auto-detect (default)
cmake -DNERVE_SIMD=auto ..
```


[Back to Architecture Index](index.md)
