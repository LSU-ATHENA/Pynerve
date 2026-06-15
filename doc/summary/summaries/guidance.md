# Practical guidance

### When to use which entropy

- **Persistence entropy** detects the spread of lifetime distribution -- useful for cluster homogeneity checks.
- **Betti entropy** detects dimensional balance -- useful for comparing topological complexity.
- **Spectral entropy** detects Laplacian spectral spread -- useful for graph regularity assessment.

### Common pitfalls

1. **Summary size too small**: around one kilobyte fits ~10 lifetimes + 5 Betti + 10 eigenvalues. If your data has more persistent features, increase `max_size` or enable the `highdim_extension`.
2. **Noise sensitivity**: Very noisy point clouds produce many short-lived pairs that pollute `top_lifetimes`. Increase `min_persistence` to filter noise, or preprocess with DMT.
3. **Not enough eigenvalues**: For graphs with >100k nodes, 10 eigenvalues under-represent the spectrum. Use `SpectralFeatureExtractor` independently and store results alongside the summary.
4. **Timestamp overflow**: `timestamp_ns` uses int64 and will overflow in 292 years. For millisecond precision, subtract a baseline epoch.

### Performance tuning

```python
# Fast approximation for real-time use
summary = summarize(points, max_size=1024, approximate=True)

# Exact summary with more features
summary = summarize(points, max_size=2048, max_persistence_dim=3)

# Batch summarization
from pynerve.core import CompactSummaryPipeline
pipeline = CompactSummaryPipeline(
    PipelineConfig(max_computation_time_ms=10.0)
)
summaries = [pipeline.computeApproximateSummary(p) for p in point_clouds]
```


[Back to index](index.md)
