# Advanced usage

### High-dimensional extension

For data with >3 intrinsic dimensions:

```python
summary = summarize(points_highdim, max_size=1024)
if summary.hasHighdimData():
    ext = summary.getHighdimExtension()
    print(f"High-dim Betti: {ext.highdim_betti_top8}")
    print(f"Dimension complexity: {ext.dimension_complexity}")
    print(f"Compression ratio: {ext.compression_ratio}")
```

The high-dim extension stores:
- Top 8 Betti numbers across all dimensions
- Lifetime statistics (mean, std, max) per dimension
- Dimension complexity scores (PCA-based residual)
- Simplex counts to estimate computational cost

### Streaming update

```python
# Accumulate summary across data chunks
accumulator = CompactSummaryPipeline(PipelineConfig())
for chunk in data_stream:
    chunk_summary = accumulator.computeApproximateSummary(chunk)
    accumulator.updateSummary(chunk_summary)

final_summary = accumulator.getAccumulatedSummary()
```

Each `updateSummary` merges lifetimes, recalculates entropy, and updates eigenvalue statistics. Merging two summaries is O(1).

### Binary comparison

```python
# Compare two summaries directly from serialized bytes
from pynerve.summary import compare_summaries

s1 = summarize(points_a).serialize()
s2 = summarize(points_b).serialize()

result = compare_summaries(s1, s2)
# result.overlap_score, result.lifetime_cosine_similarity
# result.betti_distance, result.spectral_distance
# result.composite_similarity (0-1)
```

Binary comparison avoids deserialization overhead. Useful for large-scale nearest-summary search.

### Serialization format

```python
summary = summarize(points)
data = summary.serialize()

# Binary layout (1024 bytes total):
# - Header: timestamp_ns, symbol_id, computation_time_us (24 bytes)
# - Lifetimes: 10 * (4+4+1+4) = 130 bytes
# - Betti: 5 * 2 = 10 bytes
# - Eigenvalues: 10 * (4+2) = 60 bytes
# - Entropy: 3 * 4 = 12 bytes
# - High-dim extension: variable, ~400 bytes
# - Reserved: remaining bytes

# Verify integrity
assert summary.isValid()
assert summary.isUnderSizeLimit()
```


[Back to index](index.md)
