# Complexity

[Back to Index](index.md)

In the typical sparse case the complexity is O(n^2), while in the dense worst case it reaches O(n^3). SIMD acceleration with AVX-512 provides 2-8x speedup over scalar code. GPU acceleration achieves 5-15x speedup over CPU on large complexes. Memory usage is O(n * k) where k is the maximum coboundary size.

### Detailed Complexity Breakdown

The cohomology algorithm's complexity depends on the coboundary sparsity pattern:

**Sparse Rips complex** (typical):

- Average coboundary size (dim d): k_d = O((N - d) * p), where p = edge probability
- For thresholded Rips (radius r), k_d depends on the number of (d+1)-simplices containing a given d-simplex
- In practice, k_d << n for dim >= 1
- Average column XOR cost: O(k_d + k_{d+1})
- Total operations: O(n * k_avg)

**Dense complex** (full simplex):

- Coboundary of d-simplex: n - d - 1 cofaces (all (d+1)-simplices containing it)
- For d near n/2, coboundary size is O(n)
- Total operations approach O(n^3)

### FLOPS Analysis

| Operation | Representation | Complexity | Notes |
|-----------|---------------|------------|-------|
| Column XOR | Sparse (CSR) | O(k + k') | k, k' = non-zero count of the two columns |
| Column XOR | Bitset (dense) | O(w) | w = number of 64-bit words covering n rows |
| Pivot find | Sparse (CSR) | O(1) | Direct lookup from pivot table |
| Pivot find | Bitset (dense) | O(w) | Scan for lowest set bit |
| Clearing | Both | O(1) | Flag-only operation |

where k and k' are per-column non-zero counts and w = ceil(n_rows / 64). Clearing is always O(1).

For a typical complex with n = 10^5 and average k = 8:

```
Total XORs:  ~80,000 (number of pivot conflicts)
Total FLOPS: ~80,000 * 8 * 2 = 1.28 x 10^6  (1.28 million operations)
```

This is negligible for modern hardware. The bottleneck is memory bandwidth, not computation.

### Memory Locality Analysis

The cohomology algorithm has better cache behavior than standard reduction:

1. **Sequential access pattern**: Columns are processed in order (reverse filtration), with mostly sequential access to both the column data and the pivot table.

2. **Sparse columns stay sparse**: Column density does not grow significantly, so each column fits in L1/L2 cache.

3. **Pivot table locality**: The pivot table is accessed with stride-1 in the inner loop (pivotOf[p] for varying p), which is cache-friendly.

Standard reduction, by contrast, has:
- Columns that grow dense, eventually exceeding L2 cache size
- Random access to the pivot table (pivotOf[p] where p is the column's pivot, which can be anywhere in the range [0, n))
