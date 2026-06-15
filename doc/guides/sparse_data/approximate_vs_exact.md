# Approximate vs exact

### epsilon-interleaving guarantee

The sparse VR diagram is epsilon-interleaved with the full VR diagram:

```
d_B(dgm_sparse, dgm_full) <= epsilon * max_radius
```

where `d_B` is bottleneck distance and `epsilon` is controlled by the landmark density (epsilon-net radius). With `landmark_ratio=0.1` (10% landmarks), epsilon ~ 0.1-0.3x the feature scale depending on data geometry.

Error tolerance is also controllable via the `error_tolerance` parameter:

```python
# Tight tolerance -- near-exact
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=1e-6,
)

# Loose tolerance -- faster, more aggressive sparsification
result = pynerve.compute_persistence(
    points, max_dim=2,
    mode=pynerve.PersistenceMode.APPROX,
    error_tolerance=0.1,
)
```

### Memory scaling

For n of 10^3 points, full VR uses a few megabytes of memory, sparse VR with 0.1% landmarks uses a few kilobytes, and sparse VR with 1% landmarks uses tens of kilobytes. For n of 10^4, full VR uses hundreds of megabytes, sparse VR with 0.1% landmarks uses tens of kilobytes, and sparse VR with 1% landmarks uses hundreds of kilobytes. For n of 10^5, full VR uses tens of gigabytes, sparse VR with 0.1% landmarks uses hundreds of kilobytes, and sparse VR with 1% landmarks uses a few megabytes. For n of 10^6, full VR uses a few terabytes, sparse VR with 0.1% landmarks uses a few megabytes, and sparse VR with 1% landmarks uses tens of megabytes. For n of 10^7, full VR is not feasible, sparse VR with 0.1% landmarks uses tens of megabytes, and sparse VR with 1% landmarks uses hundreds of megabytes.

Sparse mode works with 10^7+ simplices using bounded allocation. Memory is pre-allocated based on the landmark count and max dimension, with no reallocation during reduction.

Back to [Sparse Workflows Overview](index.md)
