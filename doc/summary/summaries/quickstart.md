# Quick start

```python
from pynerve import summarize

# Compact topological summary of a point cloud
points = np.random.randn(1000, 3).astype(np.float32)
summary = summarize(points, max_size=1024)

# summary contains:
# - top_lifetimes:  10 most persistent pairs (birth, death, dimension)
# - betti_counts:   Betti numbers for dimensions 0-4
# - top_eigenvalues: top 10 Laplacian eigenvalues
# - persistence_entropy, betti_entropy, spectral_entropy
# - metadata:       timestamp, computation time, data point count
```

Fixed-size persistence diagram summarization. Reduces large point clouds to a
compact `CompactSummary` of around one kilobyte that preserves topological features for
downstream analysis, visualization, or storage. Streaming-compatible update.


[Back to index](index.md)
