# Witness Complex

The Witness complex approximates the topology of a large point cloud by
building a simplicial complex on a small set of *landmark* points, using
the full set of points as *witnesses* to determine which simplices appear.
It dramatically reduces memory and computation compared to the full
Vietoris-Rips complex while preserving topological structure.

- [Definition](definition.md) - Weak/strong witness definitions and Pynerve's implementation
- [Landmark selection strategies](landmarks.md) - Farthest-point, random, and k-means landmark selection
- [Approximation guarantees](approximation.md) - Bottleneck distance bound and practical implications
- [Comparison to VR on large datasets](comparison_vr.md) - Key differences and scaling comparison
- [Complexity and memory analysis](complexity.md) - Time complexity, memory layout, and optimization
- [Code examples](code_examples.md) - Usage examples for PyTorch, NumPy, and nn.Module
- [API](api.md) - PyTorch, NumPy, nn.Module, and selective VR dispatch interfaces
- [When to use](when_to_use.md) - Recommended scenarios for witness complex
- [When not to use](when_not_to_use.md) - Scenarios where other methods are preferred
- [Practical guidance](practical_guidance.md) - Landmark count, max radius, batched mode, strong witness
- [Theoretical background](theory.md) - Delaunay relaxation, stability, and Cech relationship
- [Distance computation details](distance_details.md) - Memory layout, chunked computation, ordering
- [Performance analysis](performance.md) - Benchmarks with varying landmarks and point counts
- [Landmark selection comparison](landmark_comparison.md) - Empirical comparison and quality evaluation
- [FAQ](faq.md) - Frequently asked questions
