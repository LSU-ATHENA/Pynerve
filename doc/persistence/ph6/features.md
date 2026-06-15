# Features

PH4, PH5, and PH6 all support standard reduction and cohomology reduction. PH4 supports clearing optimization, column compression, and deterministic mode. PH5 extends clearing optimization and column compression with advanced variants, adds checksum validation, stability checks, and differentiable ops, and enhances deterministic mode. PH6 further extends these with experimental variants for clearing optimization and column compression, bleeding-edge deterministic mode, the latest differentiable ops, and introduces experimental algorithm innovations.

## Experimental Algorithms

PH6 may include any of the following when they are in development:

- **New reduction orderings**: Alternative column traversal strategies that may reduce pivot conflicts
- **Approximate clearing**: Heuristic column elimination with tunable recall
- **Speculative reduction**: Multi-path reduction with majority voting on results
- **Adaptive pivoting**: Runtime pivot-strategy selection based on column density
- **Block-sparse reduction**: Cache-blocked column operations for hierarchical memory

### New Reduction Orderings

Standard and cohomology reduction use fixed column orderings (increasing or decreasing filtration). PH6 experiments with *adaptive* orderings that reorder columns mid-reduction:

```
Algorithm: ADAPTIVE_ORDERING(D)
 1. Partition columns by dimension and filtration value
 2. Sort within each partition by estimated column density
 3. Process low-density columns first (fewer pivot conflicts)
 4. After each batch, re-sort remaining columns by updated density
```

The adaptive ordering can reduce pivot conflicts by 15-30% on complexes with highly variable column density, but the re-sorting overhead (O(n log n) per batch) can negate the benefit for small complexes.

### Approximate Clearing

Standard clearing is exact: a birth column is cleared only after its paired death column is fully reduced. Approximate clearing clears columns *before* they are definitively paired:

```
Algorithm: APPROXIMATE_CLEARING(D, threshold)
  for j = 1..n:
      if columnDensity(j) > threshold:
          clear(j)   // heuristically clear dense columns
      else:
          reduce(j)
          if isDeath(j):
              clear(birth(j))
```

The threshold controls the tradeoff. With a threshold of 0.0, all dense columns are cleared early, yielding a 2-5x speedup versus exact but with a 10-30% false clear rate. At a threshold of 0.5, only very dense columns are cleared early, producing a 1.5-2x speedup with a 1-5% false clear rate. At 0.8, only extremely dense columns are cleared, achieving a 1.1-1.3x speedup with less than 1% false clears. A threshold of 1.0 means no early clearing (exact mode), giving a 1x speedup with a 0% false clear rate.

A false clear (clearing a column that would later be needed as an additive column) produces incorrect results. PH6's approximate clearing includes a *verification pass* that detects and corrects false clears by re-processing affected columns.

**Status**: Approximate clearing is available in PH6 but **not recommended for production use**. The verification pass adds overhead that often cancels the speedup from early clearing. Research on better heuristics is ongoing.

### Speculative Reduction

Speculative reduction explores multiple reduction paths in parallel and uses majority voting to select the correct pairing:

```
Algorithm: SPECULATIVE_REDUCTION(D, k)
  // Launch k parallel reduction threads
  for t = 1..k:
      R_t = copy(D)
      // Each thread uses a different random column ordering
      permuteColumns(R_t, seed=t)
      pairs_t = reduce(R_t)

  // Majority voting on each pair
  for each simplex sigma:
      pairings = [pairs_t[sigma] for t = 1..k]
      if all k pairings agree:
          accept pairing as correct
      else:
          // Inconsistent: rerun deterministically
          pairs[sigma] = deterministicReduce(sigma)
```

The speculative approach is motivated by the observation that most columns have unique pivots -- the correct pairing is independent of column ordering. Only a small fraction (typically 1-5%) of columns are sensitive to ordering.

**Performance**: With 2 parallel threads, the ideal speedup is 2x while actual speedup is 1.5-1.8x, with correctness guaranteed by majority verification. With 4 threads, ideal speedup is 4x with actual at 2.5-3x. With 8 threads, ideal speedup is 8x but actual drops to 3-4x.

The diminishing returns come from:
- Thread synchronization overhead
- Memory bandwidth contention
- Increasing probability of disagreement (more paths = more variance)
- The deterministic rerun for disagreed pairs

**Status**: Speculative reduction is experimental and not yet proven to be correct in all cases. Use only for research and benchmarking.

### Adaptive Pivoting

Different pivot-finding strategies perform better on different column types. Scan from bottom works best for dense columns with pivot near top and worst for large columns with pivot near bottom, at O(k) complexity. Binary search is best for sorted sparse columns and worst for unsorted dense columns, at O(log k) complexity. SIMD max scan is best for medium bitset columns (64-4096 rows) and worst for very small or very large columns, at O(nwords) complexity. Cached pivot works well for any column with amortized O(1) complexity but performs worst after each XOR operation (must update), at O(1) or O(k) on update.

Adaptive pivoting switches between strategies at runtime based on column characteristics:

```cpp
PivotStrategy selectPivotStrategy(const Column& col) {
    if (col.is_bitset && col.nwords <= 64) {
        return PivotStrategy::SIMD_MAX;
    } else if (col.is_sparse && col.size() < 32) {
        return PivotStrategy::CACHED;
    } else if (col.is_sparse && col.hasPivotCache()) {
        return PivotStrategy::CACHED;  // use cache, verify on lookup
    } else {
        return PivotStrategy::SCAN_BOTTOM;
    }
}
```

**Status**: Adaptive pivoting is stable and correct in PH6. It typically provides 5-15% speedup over any single strategy.

### Block-Sparse Reduction

Modern CPUs have deep cache hierarchies (L1: under 100 kilobytes, L2: a few hundred kilobytes per core, L3: tens to hundreds of megabytes shared). Block-sparse reduction partitions the boundary matrix into blocks that fit in L2 cache:

```
Algorithm: BLOCK_SPARSE_REDUCTION(D, blockSize)
  // Partition the matrix into column blocks
  for blockStart = 1..n step blockSize:
      blockEnd = min(blockStart + blockSize, n)
      block = D[:, blockStart:blockEnd]

      // Load block into L2 cache
      prefetchToL2(block)

      // Reduce columns within the block
      for j = blockStart..blockEnd:
          reduce(block, j)

      // Write block back to main memory
      flushFromL2(block)
```

Block size selection depends on the cache level. For L1 cache (typically under 100 kilobytes), a block size of 500-2000 columns yields a 5-10% performance gain. For L2 cache (typically a few hundred kilobytes), a block size of 4000-16000 columns yields a 10-20% gain. For L3 cache (typically tens of megabytes), a block size of 50K-200K columns yields a 5-15% gain.

The gains come from improved memory locality: column XOR operations access the pivot table and a few other columns, which ideally are all in cache.

**Status**: Block-sparse reduction is correct in PH6 but the performance gains are hardware-dependent. On CPUs with large L2 caches (AMD Zen 4: on the order of a megabyte per core), the benefit is more pronounced than on CPUs with small L2 caches (some Intel Xeon: a few hundred kilobytes per core).

## What "Experimental" Means

- Algorithms in PH6 have passed unit tests and integration tests on standard benchmarks
- Performance characteristics are promising but may regress on edge cases
- The reduction strategy may change between minor versions
- Result formats are stable within the same major version
- Determinism is always enabled and guaranteed

## Graduation Path

An experimental algorithm that proves successful in PH6 graduates to PH5 (and later PH4's adaptive selector) in a subsequent release. The graduation criteria:

1. **Correctness**: 100% agreement with PH4 on a suite of 100+ test complexes (diverse types and sizes).
2. **Performance**: At least 10% faster than PH4/PH5 on at least 3 benchmark categories.
3. **Stability**: No regressions across 10+ library releases.
4. **Determinism**: Verified bitwise reproducibility on CPU and GPU.
5. **Memory**: No significant increase in peak memory usage.

To stay informed about graduation status, check the release notes.


[Back to PH6 Index](index.md)
