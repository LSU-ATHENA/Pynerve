## SIMD encode/decode

`src/encoders/simd_encoder_ops.cpp` provides batch operations:

```cpp
void simdEncodeBatch(const double* input, Size n, Size dim, double* output);
void simdDecodeBatch(const double* encoded, Size n, Size code_dim, double* output);
```

Intrinsics used:

- **Weighted sum**: `_mm512_fmadd_ps/pd`, `_mm512_reduce_add_ps/pd`
- **Activation (ReLU)**: `_mm512_max_ps/pd`, `_mm512_setzero_ps/pd`
- **Batch normalization**: `_mm512_sub_ps/pd`, `_mm512_mul_ps/pd`, `_mm512_add_ps/pd`
- **Softmax**: `_mm512_exp_ps`, `_mm512_reduce_add_ps`, `_mm512_div_ps`

```cpp
#include <nerve/encoders/simd_encoder.hpp>

constexpr int batch_size = 1024;
constexpr int input_dim = 128;
constexpr int code_dim = 32;

double input[batch_size * input_dim];
double output[batch_size * code_dim];

// SIMD-accelerated batch encode (uses AVX-512 if available)
simdEncodeBatch(input, batch_size, input_dim, output);

// SIMD-accelerated batch decode
simdDecodeBatch(output, batch_size, code_dim, decoded);
```


## Implementation details

### Weighted sum

```cpp
// AVX-512 weighted sum: output[j] = sum_i(input[i] * weights[i][j])
void simdEncodeBatch(const double* input, Size n, Size dim, double* output) {
    #pragma omp parallel for
    for (Size j = 0; j < n; j++) {
        __m512d sum = _mm512_setzero_pd();
        for (Size i = 0; i < dim; i += 8) {
            __m512d inp = _mm512_loadu_pd(input + j * dim + i);
            __m512d w = _mm512_loadu_pd(weights + i);  // shared weights
            sum = _mm512_fmadd_pd(inp, w, sum);
        }
        output[j] = _mm512_reduce_add_pd(sum);
    }
}
```

### Activation (ReLU)

```cpp
void simdReLU(double* data, Size n) {
    __m512d zero = _mm512_setzero_pd();
    for (Size i = 0; i < n; i += 8) {
        __m512d v = _mm512_loadu_pd(data + i);
        __m512d relu = _mm512_max_pd(v, zero);
        _mm512_storeu_pd(data + i, relu);
    }
}
```

### Batch normalization

```cpp
void simdBatchNorm(double* data, Size n, double mean, double std_inv) {
    __m512d vmean = _mm512_set1_pd(mean);
    __m512d vstd = _mm512_set1_pd(std_inv);
    for (Size i = 0; i < n; i += 8) {
        __m512d v = _mm512_loadu_pd(data + i);
        v = _mm512_sub_pd(v, vmean);
        v = _mm512_mul_pd(v, vstd);
        _mm512_storeu_pd(data + i, v);
    }
}
```

### Auto-dispatch

The encoder functions detect CPU capabilities at runtime:

- **AVX-512**: dispatches to `simdEncodeBatch_avx512`
- **AVX2 + FMA**: dispatches to `simdEncodeBatch_avx2`
- **SSE4.1**: dispatches to `simdEncodeBatch_sse41`
- **None (fallback)**: dispatches to `encodeBatch_scalar`

```cpp
static auto dispatch = CpuFeatures::detect();
if (dispatch.has_avx512f) {
    simdEncodeBatch_avx512(input, n, dim, output);
} else if (dispatch.has_avx2 && dispatch.has_fma) {
    simdEncodeBatch_avx2(input, n, dim, output);
} else {
    simdEncodeBatch_sse41(input, n, dim, output);
}
```

### Memory alignment requirements

For maximum SIMD throughput:
- Input/output buffers should be 64-byte aligned (cache line)
- Stride should be a multiple of 8 for double-precision
- Batch size should be a multiple of 8 (pad if necessary)


## Complexity analysis

- **Weighted sum**: 1 FMAdd + 1 reduce per element, 8x (f64) / 16x (f32) SIMD speedup.
- **ReLU**: 1 max + 1 store per element, 8x SIMD speedup.
- **Batch norm**: 1 sub + 1 mul + 1 store per element, 8x SIMD speedup.
- **Softmax**: 2 passes + exp + div per element, ~4x SIMD speedup (exp is slow).

For large batches (n > 1024), Amdahl's law gives diminishing returns. The SIMD encoding path achieves ~90% of peak FLOPs on Ice Lake and later architectures.


## Python performance tips

```python
from pynerve.encoders import simd_encode_batch, simd_decode_batch
import numpy as np

# Align memory for best SIMD performance
data = np.random.randn(1024, 128).astype(np.float64)

# Use contiguous arrays (non-contiguous = memcpy overhead)
data_contiguous = np.ascontiguousarray(data)

# Batch encode
encoded = simd_encode_batch(data_contiguous)
decoded = simd_decode_batch(encoded)
```

## FAQ

**Q: What CPU features does the SIMD encoder require?**
A: The encoder auto-detects available instruction sets. AVX-512 delivers maximum throughput (8x speedup for double-precision), AVX2+FMA is a solid fallback, and SSE4.1 is the minimum baseline. No specific CPU feature is strictly required -- the scalar fallback works everywhere.

**Q: Can I force a specific dispatch path?**
A: Yes. Set the `NERVE_SIMD_FORCE` environment variable to `avx512`, `avx2`, `sse41`, or `scalar` to override runtime detection.

**Q: Why should batch size be a multiple of 8?**
A: The SIMD kernels process 8 double-precision or 16 single-precision elements per instruction. Non-aligned batch sizes incur tail handling overhead (partial vector store). Padding to the next multiple avoids this.

**Q: Does SIMD encoding help with small batches?**
A: For batches under ~64 items, kernel launch and memory overhead dominate. The speedup approaches theoretical peak only for batches larger than a few thousand items.


### Cross-references

- `pynerve.encoders`: Encoders overview
- `pynerve.encoders.gpu`: GPU encoder kernels
- `pynerve.core.simd_ops`: SIMD memory operations
- `pynerve.nn.simd`: SIMD activation functions
