# Common Pitfalls

[Back to Index](index.md)

### Pitfall 1: Reverse Pair Ordering

Cohomology produces pairs in the opposite direction from homology. The library normalizes this, but if you implement cohomology yourself, you must invert the pairing:

```python
# Cohomology pair (pivot, col_idx) means:
#   birth = col_idx  (the column being reduced)
#   death = pivot     (the pivot row)
# But the caller expects (birth, death) = (pivot, col_idx) in homology convention.
# The library handles this internally.
```

### Pitfall 2: Forgetting the Dimension Filter

In cohomology reduction, you only eliminate pivots in dimension d+1 (where d is the current column's dimension). Eliminating pivots in other dimensions produces incorrect results.

```python
# WRONG: eliminate any pivot conflict
while pivot >= 0 and pivot in pivot_of:
    ...

# CORRECT: only eliminate pivots of cohomology-relevant columns
while pivot >= 0 and pivot in pivot_of:
    # pivot_of[pivot] is guaranteed to be a (d+1)-simplex
    # because only (d+1)-simplices can have coboundary entries at this dim
    ...
```

### Pitfall 3: Emergent Pair Misidentification

Not every column with a single entry is an emergent pair. The single entry must be in the pivot position and must correspond to a valid coface. Checking `len(column) == 1` is not sufficient -- verify that the single entry is in the expected dimension range.

### Pitfall 4: Memory Overhead for Coface Index

Building the coface index requires storing, for each simplex, the list of cofaces. This doubles memory usage compared to standard reduction (which only needs the boundary matrix). For n = 10^7 and average k = 8, the coface index adds several hundred megabytes.

```python
# Estimate coface index memory
n = 10_000_000
avg_cofaces = 8
bytes_per_entry = 8  # uint64_t
total_bytes = n * avg_cofaces * bytes_per_entry  # megabytes

# The boundary matrix itself is similar size
total_peak = 2 * total_bytes  # ~gigabytes
```

### Pitfall 5: GPU Transfer Overhead

For small complexes (n < 10^5), the overhead of transferring data to GPU and back can exceed the computation time:

For n = 10^4, the CPU takes 5 ms while the GPU computation takes 1 ms with 3 ms of transfer time, totaling 4 ms which is faster than CPU. For n = 10^3, the CPU takes 1 ms while the GPU computation takes 0.1 ms but requires 1.5 ms of transfer time, totaling 1.6 ms which is slower than CPU.

**Fix**: The library uses a threshold (default: n > 10^4) below which it uses the CPU path even when the GPU backend is selected. This threshold can be configured.

### Pitfall 6: Determinism with Floating-Point Filtration Values

When filtration values (birth radii) are floating-point, different reductions (homology vs cohomology) may produce different pairings for equal filtration values. This is not a bug -- it reflects that the complex is not generic. Use the `error_tolerance` parameter to control tie-breaking behavior.
