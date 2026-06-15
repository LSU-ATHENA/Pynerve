# FAQ

### How does PH4 choose between standard and cohomology reduction?

PH4 automatically selects the reduction strategy based on problem size and density. For small complexes (n < 10^4), standard reduction is used due to lower overhead. For dense complexes with estimated density above 0.5 and max_dim <= 2, standard reduction may still be competitive. For larger sparse complexes, cohomology-style reduction is preferred. The density estimate is computed as actual edges divided by possible edges for Vietoris-Rips complexes, or actual cells divided by total cells for cubical complexes.

### What does the approximate mode do?

Approximate mode uses witness sampling to compute persistence on a much smaller complex derived from the original data. It selects a subset of landmark points, builds a witness complex where non-landmark points vote for nearby simplices, computes persistence on this reduced complex, and then extrapolates the results back to the original dataset. This provides significant speedups (10-50x) at the cost of some accuracy, measured as normalized bottleneck distance between exact and approximate barcodes.

### How does the memory budget work?

The `memory_budget_megabytes` option lets you specify a maximum memory limit. PH4 estimates memory requirements before computation by accounting for column storage, pivot tables, coface indices, and thread working buffers. If the estimate exceeds the budget, PH4 first switches to sparse column representation, then enables more aggressive compression, then falls back to witness sampling (approximate mode), and finally raises `MemoryBudgetExceeded` if still over budget. This prevents out-of-memory crashes in constrained environments.

### Is PH4 deterministic?

Yes, PH4 is bitwise reproducible by default. Every run with identical inputs produces identical persistence pairs. This is enforced through fixed seed propagation via a `DeterminismContract`, a deterministic reduction tree (using `schedule(static)` in OpenMP to avoid thread-dependent scheduling), and no floating-point atomics in the GPU path.

### When should I use PH5 or PH6 instead?

PH5 and PH6 offer advanced features not available in PH4. Use PH5 if you need checksum validation, advanced clearing, or extended determinism contracts. Use PH6 if you need differentiable persistence for gradient-based learning, or if you want access to the latest experimental algorithms. For general-purpose persistence computation, PH4 remains the recommended default.

Back to [PH4 Engine Overview](index.md)
