# Common pitfalls

**Cup product:**
1. Memory grows as O(s * n_p * n_q). For large complexes with many cohomology classes, the cup product table can dominate memory. Use `max_dim` and `max_radius` to bound the complex size.
2. The cup product is only defined for cochains on the same complex. When comparing results from different filtrations, ensure the underlying complex is identical.
3. GPU cup product requires all simplices to fit in device memory. For out-of-memory, fall back to CPU with `backend="cpu"`.

**Reeb graph:**
1. The merge tree component assumes the scalar function is piecewise-linear on the graph. For noisy scalar fields, apply smoothing or increase `persistence_threshold`.
2. Connected component labeling on GPU uses atomic operations. With many vertices (>1M), switch to CPU union-find for deterministic results.
3. The simplified Reeb graph may disconnect if `persistence_threshold` is too aggressive. Use `compute_merge_tree` to inspect the full hierarchy before simplification.

**Zigzag persistence:**
1. Time slices must cover the same point set (possibly with deletions). When points move between slices, use a correspondence vector to map points across time.
2. Memory scales with O(max_window_size * num_timesteps). For long sequences, use the streaming variant which processes windows incrementally.
3. The GPU zigzag kernel assumes each time slice fits in GPU memory. For very large slices, process in chunks with CPU fallback.


[Back to index](index.md)
