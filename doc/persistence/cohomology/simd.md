# SIMD Acceleration

[Back to Index](index.md)

Cohomology column operations are vectorized using bit-packed column representations and SIMD instruction sets:

SSE4.1 uses 128-bit registers with 2 x 64-bit operations per cycle; column XOR uses `_mm_xor_si128` and pivot find uses `_mm_max_epu64`. AVX2 uses 256-bit registers with 4 x 64-bit operations per cycle; column XOR uses `_mm256_xor_si256` and pivot find uses `_mm256_max_epu64`. AVX-512F uses 512-bit registers with 8 x 64-bit operations per cycle; column XOR uses `_mm512_xor_epi64` and pivot find uses `_mm512_max_epu64`. AVX-512BW also uses 512-bit registers with 8 x 64-bit operations per cycle, supporting byte and word extensions. AVX-512VL supports 256-bit and 128-bit registers with variable operations per cycle, providing masked column XOR operations.

Runtime dispatch via CPUID detection:

```python
# AVX-512 is used automatically when available
result = pynerve.compute_persistence_cohomology(points)
# The library detects AVX-512F/AVX-512VL/AVX-512BW/AVX-512DQ
# and selects the best kernel at runtime
```

Minimum column size for AVX-512: 8 x uint64_t (>= 512 rows). For smaller columns, scalar word-wise XOR is used.

### SIMD Kernel Selection

```cpp
// Pseudo-code for runtime kernel selection
void xor_columns_bitset(uint64_t* dst, const uint64_t* src, size_t words) {
    if (words <= 1) {
        // Scalar: single word
        *dst ^= *src;
    } else if (words <= 4 && has_sse41()) {
        // SSE4.1: 128-bit path
        xor_bitset_sse41(dst, src, words);
    } else if (words <= 8 && has_avx2()) {
        // AVX2: 256-bit path
        xor_bitset_avx2(dst, src, words);
    } else if (has_avx512f()) {
        // AVX-512: 512-bit path
        xor_bitset_avx512(dst, src, words);
    } else {
        // Generic scalar fallback
        xor_bitset_scalar(dst, src, words);
    }
}
```

### SIMD Pivot Finding

Finding the pivot (the highest set bit) is accelerated with SIMD:

```cpp
// AVX2 pivot find: return index of highest 1 in bitset
int find_pivot_avx2(const uint64_t* bits, size_t words) {
    // Scan from the last word backward
    for (size_t i = words; i > 0; --i) {
        if (bits[i-1] != 0) {
            return (i-1) * 64 + __builtin_clzll(bits[i-1]);
        }
    }
    return -1;  // empty column
}
```

Using `__builtin_clzll` (count leading zeros) on each 64-bit word gives the bit position in O(1) per word. Combined with backward scanning, pivot finding is O(words) on average and O(1) for columns with pivots in the upper portion.
