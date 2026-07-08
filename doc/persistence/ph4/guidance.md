# Practical Guidance

### Choosing the Right Configuration

```python
# Scenario 1: Small data, quick exploration
points = np.random.rand(50, 3)
result = pynerve.compute_persistence(points, max_dim=1)

# Scenario 2: Moderate data, production pipeline
opts = PersistenceOptions(threads=os.cpu_count())
result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_dim=2, max_radius=0.5)

# Scenario 3: Large data, GPU-accelerated
opts = PersistenceOptions(backend=PersistenceBackend.CUDA_HYBRID)
result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_dim=2, max_radius=0.3)

# Scenario 4: Memory-constrained, approximate mode
opts = PersistenceOptions(mode=PersistenceMode.APPROX)
result = pynerve.compute_persistence_up_to_dim_4(points, opts, max_dim=2)
```

### Performance Tuning Checklist

1. **Start with defaults**: `compute_persistence()` with no options.
2. **If too slow**: Reduce `max_dim` or `max_radius`.
3. **If still too slow**: Enable approximate mode with witness sampling.
4. **If GPU is available**: Use `backend=PersistenceBackend.CUDA_HYBRID`.
5. **If memory is limited**: Set `memory_budget_megabytes` explicitly.
6. **Determinism is always on**: Bitwise reproducibility is guaranteed by default.
7. **If the complex is tiny**: Standard reduction is fine.
8. **If the complex is huge**: Use streaming or approximate mode.

### Common Mistakes

1. **Setting max_dim too high for the data**: For N points, the maximum meaningful homology dimension is at most N-2. Setting max_dim to 5 for 20 points is unnecessary and expensive.

2. **Using exact mode for n > 10^6**: At this scale, exact computation may take minutes to hours. Use approximate mode unless exact results are essential.

3. **Not setting max_radius**: Default is infinity, which evaluates all possible edges. For N=2000, that's ~2 million edges. Always set a reasonable radius.

4. **Ignoring the density heuristic**: If your data is known to be dense (e.g., a full grid), you can help PH4 by specifying `reduction="standard"` to avoid the cohomology overhead.

5. **Not checking Betti numbers before detailed analysis**: Betti numbers are computed for free alongside the pairs. Always check them first.

Back to [PH4 Engine Overview](index.md)
