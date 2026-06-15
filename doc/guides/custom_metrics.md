# Custom Metrics

Use any distance function, precomputed distance matrix, or kernel method in place of Euclidean distance.

This page has been split into subpages:

- [Distance function interface](metrics/distance_function.md)  --  Callable protocol and custom metric implementations
- [Precomputed distance matrix](metrics/precomputed.md)  --  Dense and sparse precomputed matrices
- [Kernel methods](metrics/kernel_methods.md)  --  Gaussian, polynomial, and other kernel-based distances
- [HNSW approximate nearest neighbors](metrics/hnsw.md)  --  Fast VR construction with custom metrics via HNSW
- [Lazy distance computation](metrics/lazy.md)  --  On-the-fly distance computation with O(1) memory
- [Distance callable protocol (C++)](metrics/cpp_protocol.md)  --  C++ DistanceMetric interface
- [API reference](metrics/api.md)  --  Python API for custom metrics
- [Performance considerations](metrics/performance.md)  --  Cost breakdown, guidelines, memory sizing, auto-detection
- [PyTorch integration](metrics/pytorch.md)  --  Custom metrics with PyTorch persistence modules
- [Available built-in metric strings](metrics/builtin_metrics.md)  --  Built-in metric names and their properties
- [Metric registration for reuse](metrics/registration.md)  --  Registering and reusing custom metrics by name
- [FAQ](metrics/faq.md)  --  Frequently asked questions
