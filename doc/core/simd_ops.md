## SIMD ops

`src/core/core_simd_ops.cpp` provides runtime-dispatched SIMD primitives:

`simd_memcpy` uses `_mm512_storeu`/`_mm512_loadu` on AVX-512, `_mm256_storeu`/`_mm256_loadu` on AVX2, `_mm_storeu`/`_mm_loadu` on SSE4.1, and falls back to `memcpy`. `simd_memset` uses `_mm512_set1_epi8` with store on AVX-512, `_mm256_set1_epi8` with store on AVX2, `_mm_set1_epi8` with store on SSE4.1, and falls back to `memset`. `simd_reduce_sum` uses `_mm512_reduce_add_pd` on AVX-512, `_mm256_hadd_pd` on AVX2, `_mm_hadd_pd` on SSE4.1, and a scalar loop as fallback.

Dispatch is decided once at startup via `CPUFeatures` and cached.


### Implementation details

**simd_memcpy:**
```cpp
void simd_memcpy(void* dst, const void* src, size_t n) {
    // Aligned copy for large blocks
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512d v0 = _mm512_loadu_pd((const double*)((const char*)src + i));
        _mm512_storeu_pd((double*)((char*)dst + i), v0);
    }
    // Scalar tail
    if (i < n) memcpy((char*)dst + i, (const char*)src + i, n - i);
}
```

**simd_memset:**
```cpp
void simd_memset(void* dst, int value, size_t n) {
    __m512i v = _mm512_set1_epi8((char)value);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_si512((__m512i*)((char*)dst + i), v);
    }
    if (i < n) memset((char*)dst + i, value, n - i);
}
```

**simd_reduce_sum:**
```cpp
double simd_reduce_sum(const double* data, size_t n) {
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(data + i);
        sum = _mm512_add_pd(sum, v);
    }
    double s = _mm512_reduce_add_pd(sum);
    for (; i < n; ++i) s += data[i];
    return s;
}
```


### Dispatch mechanism

```cpp
using MemcpyFn = void(*)(void*, const void*, size_t);
using MemsetFn = void(*)(void*, int, size_t);
using ReduceSumFn = double(*)(const double*, size_t);

MemcpyFn g_memcpy_impl;
MemsetFn g_memset_impl;
ReduceSumFn g_reduce_impl;

void init_simd_dispatch() {
    auto feats = CPUFeatures::detect();
    if (feats.hasAVX512F()) {
        g_memcpy_impl = simd_memcpy_avx512;
        g_memset_impl = simd_memset_avx512;
        g_reduce_impl = simd_reduce_sum_avx512;
    } else if (feats.hasAVX2()) {
        g_memcpy_impl = simd_memcpy_avx2;
        ...
    } else if (feats.hasSSE4_1()) {
        ...
    } else {
        g_memcpy_impl = memcpy_impl;
        ...
    }
}
```


### Performance

For large aligned copies, AVX-512 reaches tens of gigabytes per second throughput versus about a dozen gigabytes per second scalar. Unaligned copies are slightly slower but still significantly faster than scalar. Large memset operations see similar gains. For reduction over a million elements, AVX-512 achieves several times the scalar throughput.


### Usage notes

```cpp
#include <nerve/core/simd_ops.hpp>

double data[1024];
// ... fill data ...

// Automatically selects AVX-512, AVX2, or SSE4.1 based on CPU
nerve::core::simd_memcpy(dst, src, 4096);
nerve::core::simd_memset(buffer, 0, 8192);
double sum = nerve::core::simd_reduce_sum(data, 1024);
```

**Alignment:**
- 64-byte alignment enables `_mm512_store_pd` (aligned store, ~10% faster)
- Use `SIMD_ALIGNMENT` constant for portable aligned allocation
- Unaligned operations (`_mm512_storeu_pd`) work but are slightly slower


### Cross-references

- `pynerve.core.core`: CPU feature detection
- `pynerve.algebra.distance`: SIMD distance kernels
- `pynerve.nn.simd`: SIMD activation functions
- `pynerve.optimization.simd`: SIMD gradient clipping
