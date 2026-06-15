# Internal Implementation Details

### Data Structures

The internal C++ implementation (core engine) uses the following key data structures:

```cpp
// Column stored as a bitset or sorted index list
union ColumnStorage {
    // For columns <= 64 rows: bitmask (uint64_t)
    uint64_t bitset;
    // For larger columns: sorted vector of row indices
    std::vector<uint32_t>* sparse;
};

// The reduction state
struct ReductionState {
    std::vector<ColumnStorage> columns;  // one per simplex
    std::vector<int32_t> pivot_to_column; // mapping: pivot row -> column
    std::vector<std::pair<int32_t, int32_t>> pairs;
    BitMask active_columns;  // which columns are still live
};
```

### Memory Layout

Columns are stored in a contiguous array for cache efficiency. Each column starts with a 64-bit metadata word:

```
Column memory layout (x86-64):

[ 8 bytes: metadata | data bytes ... ]

Metadata bits:
  bit 63    : cleared flag
  bits 48-62: dimension hint
  bits 32-47: column length (non-zero count)
  bits 0-31 : current pivot (or sentinel for empty)

Data (variable length):
  If length <= 64: 8 bytes (uint64_t bitset)
  If length > 64:  4 * length bytes (uint32_t indices, sorted)
```

This small-header approach keeps overhead to 8 bytes per column. For a complex with n = 10^6 simplices, the column metadata array is about eight megabytes, fitting in L3 cache on modern CPUs.

### Pivot Lookup Table

The pivot table is a plain array of int32_t of length n. It maps each row index to the column that currently claims that row as its pivot:

```
pivot_map[row] = column_index  or  -1 if unclaimed
```

Access is O(1) and the array is frequently accessed, so it is kept cache-resident. On anti-dependent workloads (many columns sharing pivots), the pivot map is read and written for each XOR operation, making it a potential bottleneck. The clearing optimization reduces pressure on this structure.

### Column XOR Implementation (Optimized)

The inner loop XOR uses SIMD for bitset columns:

```cpp
// SSE4.1: XOR 128 bits at a time
void xor_bitset_sse(uint64_t* dst, const uint64_t* src, size_t words) {
    for (size_t i = 0; i < words; i += 2) {
        __m128i a = _mm_load_si128((__m128i*)(dst + i));
        __m128i b = _mm_load_si128((__m128i*)(src + i));
        _mm_store_si128((__m128i*)(dst + i), _mm_xor_si128(a, b));
    }
}
```

For sparse columns (sorted index lists), XOR is a linear merge:

```
Algorithm: SPARSE_XOR(a, b)
  i = 0; j = 0; result = []
  while i < len(a) AND j < len(b):
      if a[i] < b[j]: result.append(a[i]); i++
      elif b[j] < a[i]: result.append(b[j]); j++
      else: i++; j++  // cancel both (1 XOR 1 = 0)
  append remaining elements from a and b
  return result
```

This merge is O(|a| + |b|) and cache-friendly when both columns are in sorted order.

<- [Standard Reduction Overview](index.md)
