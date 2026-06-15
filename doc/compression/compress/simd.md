# SIMD Quantize / Dequantize

```cpp
void simdQuantize(const double* input, uint8_t* output, size_t n,
                   double scale, double offset);
void simdDequantize(const uint8_t* input, double* output, size_t n,
                     double scale, double offset);
```

Uses AVX-512 `_mm512_cvtpd_epi32` + pack to convert 8 doubles -> 8 int32 ->
8 uint8 per cycle.


[Back to Compression Index](index.md)
