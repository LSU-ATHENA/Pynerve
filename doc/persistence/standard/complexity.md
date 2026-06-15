# Complexity Analysis

### Time Complexity

The worst-case time complexity is O(n^3), with each of n columns processed, each requiring O(n) XOR operations, and each XOR operating on O(n) rows. The average case for sparse data is O(n^2 * k), where k is the average column size (non-zeros per column) and typical k is much smaller than n. Clearing provides an empirical speedup of 20-40% by eliminating a fraction of column operations. Compression contributes an additional 10-30% benefit through fewer columns processed and shorter per-column operations.

Where *n* is the number of simplices and *k* is the average boundary size.

### Derivation of Worst-Case Bound

The three nested loops:

1. **Outer loop**: n columns (j = 1..n)
2. **Inner (while) loop**: each column may conflict with up to O(n) previously paired columns. In the worst case (e.g., a complete filtration where many columns share pivots), each column requires O(n) XOR operations.
3. **Column XOR**: adding two columns requires iterating over O(k) non-zero entries. In the worst case, k = O(n) (dense columns).

Product: O(n * n * n) = O(n^3). This worst case is realized by certain adversarial filtrations, such as the "full simplex" on d vertices with all simplices present.

### Average-Case Analysis

For realistic Vietoris-Rips complexes:

- Vertices (dim 0): k = 0 (empty boundary columns). O(1) per column.
- Edges (dim 1): k = 2 (two endpoints). O(1) per column.
- Triangles (dim 2): k = 3 (three edges). O(1) per column.
- d-simplices (dim d): k = d+1 (boundary size = d+1). O(d) per column.

However, the *while-loop* iterations depend on how many pairs share pivots. For random Rips complexes, this is well-approximated by:

    E[iterations] = O(log n) per column

giving an expected total of O(n^2 log n). The O(n^3) worst case is rarely encountered in practice.

### Memory Complexity

The dense matrix requires O(n^2) bits or bytes and is impractical for n greater than 10^5. Sparse columns use O(n * k) entries, where k is the average number of non-zeros per column. The pivot array uses O(n) integers at 4-8 bytes per entry. The pairing output is O(n) pairs at constant storage per simplex.

The dense representation is O(n^2) bits, which for n = 10^5 is approximately one and a quarter gigabytes (bit-packed) or around ten gigabytes (byte). Sparse representation typically requires 10-100x less memory for realistic complexes.

### FLOPS Analysis

Each column XOR operation involves:
- Iterating over non-zero entries of the two columns being XORed
- For each matching index: a bit flip (XOR)
- For each unique index: insertion into the result

Let nnz(C) be the number of non-zeros in column C. An XOR between columns j and k:

    cost(j, k) = nnz(R[:, j]) + nnz(R[:, k])   (in the worst case)

The total operation count:

    total_ops = sum_{j=1..n} sum_{t=1..iterations(j)} (nnz(R[:, j]) + nnz(R[:, k_t]))

For Rips complexes in practice, total_ops ranges from 10^6 to 10^11 depending on n and the data distribution.

### Memory Bandwidth

Column operations are memory-bandwidth-bound on modern CPUs:
- Each XOR reads two columns and writes one column.
- Columns are stored as arrays of indices (typically uint32 or uint64).
- If columns are in L1/L2 cache, throughput is high.
- If columns exceed L2 cache (typical for large complexes), operations bottleneck on main memory bandwidth.

The clearing optimization improves effective bandwidth by reducing the total data read/written.

<- [Standard Reduction Overview](index.md)
