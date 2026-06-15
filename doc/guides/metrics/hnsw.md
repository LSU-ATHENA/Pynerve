# HNSW approximate nearest neighbors

For fast VR construction with custom metrics, nearest neighbors can be computed via HNSW instead of brute force:

```python
# When n is large and metric is expensive, HNSW reduces
# VR construction from O(n^2) to O(n log n) distance calls
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=1.0,
    metric=my_expensive_distance,
    # HNSW parameters (when available):
    # hnsw_ef_construction=200,
    # hnsw_m=16,
)
```

HNSW builds a hierarchical navigable small-world graph of the point cloud. During VR construction, nearest-neighbor queries use the HNSW index instead of full distance scans. This is particularly beneficial for:
- High-dimensional data (dim > 10)
- Expensive custom metrics
- n > 100,000

### HNSW parameter guide

`M` (default 16) controls the maximum neighbors per layer; lower values use less memory with lower recall, while higher values provide better recall with more memory. `ef_construction` (default 200) controls the search width during construction; lower values give faster build with lower quality, while higher values give slower build with higher quality. `ef_search` (default 100) controls the search width at query time; lower values give faster queries with lower recall, while higher values give slower queries with higher recall.

[Back to index](index.md)
