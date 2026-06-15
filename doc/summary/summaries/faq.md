# FAQ

**Q: Why around one kilobyte?**
A: Around one kilobyte fits in a single memory page, two cache lines of 512-bit SIMD, and one Ethernet MTU. It is the smallest practical size for a topologically informative summary.

**Q: How are ties in persistence resolved?**
A: When two pairs have equal persistence, the one with earlier birth time takes priority. If births are also equal, the tie is broken by symbol_id.

**Q: Can I customize which features go into the summary?**
A: Yes. Subclass `CompactSummaryPipeline` and override `extractFeatures()`. The base class provides `extractLifetimes`, `extractBetti`, `extractEigenvalues`, and `computeEntropy` as protected methods.

**Q: Is the summary GPU-compatible?**
A: Yes. `computeApproximateSummary` runs on GPU when `enable_gpu=True` is set in the pipeline config. The summary itself is CPU memory, so it can be serialized or broadcast after GPU computation.

**Q: What happens if the summary exceeds around one kilobyte?**
A: The `isUnderSizeLimit()` method checks the constraint. If exceeded, enable the high-dimensional extension which packs data more efficiently. The serialization format truncates to around one kilobyte by dropping the lowest-persistence features first.

**Q: Can the summary be used for similarity search?**
A: Yes. The `compare_summaries` function computes a composite similarity score from lifetime, Betti, and spectral components. Use this score for k-NN search over pre-computed summaries. Binary comparison is O(1) per pair.

**Q: How often can I update a streaming summary?**
A: `updateSummary()` runs in O(1) time. You can call it on every new data window (e.g., every 100ms for real-time monitoring). The summary accumulates information without growing beyond around one kilobyte.

**Q: Does the summary preserve homotopy type?**
A: The summary stores the most persistent features, which approximate the homotopy type. It does NOT guarantee homotopy equivalence -- it is a lossy compression that preserves the most significant topological information.


### Cross-references

- `pynerve.optimization.compact_summary`: Optimized summary computation
- `pynerve.serialization`: Serialization of summaries
- `pynerve.spectral`: Eigenvalue computation for spectral entropy
- `pynerve.ml`: ML using summary features
- `pynerve.validation.benchmarks`: Summary computation benchmarks
- `pynerve.metrics`: Distance metrics for comparing summaries

[Back to index](index.md)
