# FAQ

**Q: When should I use cup product vs persistence?**
A: Cup product reveals ring structure (how cohomology classes multiply), which persistence alone cannot see. Use cup product when you need to distinguish spaces with identical Betti numbers but different cohomology rings.

**Q: Why does my Reeb graph have disconnected components?**
A: The merge tree connects components when their scalar values merge. If branches appear disconnected, check that: (1) the graph is connected, (2) the scalar function is defined on all vertices, and (3) the persistence threshold is not too large.

**Q: How many time slices do I need for meaningful zigzag?**
A: At minimum 3 slices to detect a feature that appears, persists, and disappears. For robust tracking, use 5-10 slices covering the expected lifetime of features.

**Q: Can I compute cup product on a subset of dimensions?**
A: Yes. Specify `dims=[1, 1]` to compute only the product H^1 x H^1 -> H^2. This reduces memory and computation when only specific products are needed.

**Q: How does zigzag persistence differ from standard persistence?**
A: Standard persistence processes a single filtration (increasing). Zigzag processes a sequence where simplices can appear AND disappear across time. This captures features that are transient -- they appear, persist for some time, and then disappear.

**Q: Can I use the Reeb graph for topological simplification of meshes?**
A: Yes. The Reeb graph provides a topological skeleton. Remove branches below a persistence threshold to simplify the mesh while preserving its overall structure. Use `persistence_threshold` to control the level of simplification.

**Q: What limits the maximum dimension for cup product computation?**
A: The cup product at dimension (p,q) requires (p+q)-simplices. For dim > 3, the number of simplices grows exponentially. The GPU kernel is limited by GPU memory for high-dimension complexes. Use `max_dim=2` for most practical applications.

**Q: Is zigzag persistence differentiable?**
A: Zigzag persistence is not currently differentiable. The birth/death events depend on the ordering of time slices, which has discontinuous changes. For differentiable time-varying topology, consider the `pynerve.diff` module with separate per-slice persistence.


### Cross-references

- `pynerve.algebra`: Simplicial complex for cup product
- `pynerve.graphs`: Graph structures for Reeb graph
- `pynerve.persistence`: Standard persistence (non-zigzag)
- `pynerve.cuda`: GPU acceleration
- `pynerve.dmt.parallel`: Streaming Morse builder (alternative to zigzag for large data)
- `pynerve.validation.benchmarks`: Specialized algorithm benchmarking

[Back to index](index.md)
