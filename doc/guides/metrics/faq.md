# FAQ

**Q: Can I use a custom distance function with GPU acceleration?**
A: Custom Python callables and numegabytesa JIT functions run on the CPU only. For GPU-accelerated custom metrics, implement a C++ `DistanceMetric` subclass that uses CUDA, or use the built-in SIMD metrics which utilize Tensor Cores on compatible GPUs.

**Q: What is the fastest way to compute persistence with a custom metric for large datasets?**
A: For n > 100,000 with expensive custom metrics, use HNSW approximate nearest neighbors to reduce distance evaluations from O(n^2) to O(n log n). For moderately sized data, precompute the distance matrix or use a numegabytesa JIT-compiled metric.

**Q: How do I register a custom metric for reuse across calls?**
A: Use the `@register_metric` decorator to associate a name with your custom metric function, then reference it by string name in subsequent `compute_persistence` calls without re-specifying the function.

**Q: What happens if my distance function is not symmetric?**
A: Pynerve assumes metrics are symmetric but does not enforce it. Non-symmetric distances may produce incorrect persistence diagrams. Ensure your metric satisfies `metric(a, b) == metric(b, a)`.

**Q: Can I use a precomputed distance matrix with PyTorch tensors?**
A: Yes. Compute your distance matrix using PyTorch operations (e.g., `torch.cdist`), convert it to a NumPy array with `.numpy()`, and pass it with `metric="precomputed"`.

**Q: When should I use a kernel method instead of an explicit distance?**
A: Kernel methods are useful when the data has unknown or complex structure that a simple distance cannot capture. They reveal non-linear similarities but require additional computation per evaluation.

[Back to index](index.md)
