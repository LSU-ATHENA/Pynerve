# Determinism

Pynerve guarantees **bitwise reproducibility** by default. Every computation with the same inputs and same seed produces identical output bits, regardless of CPU or GPU backend.

### Full determinism specification

The following inputs determine a computation's output:

1. **Point cloud data**: exact floating-point values (not byte-level identical, but IEEE 754 bit-exact)
2. **Options**: max_dim, max_radius, mode, backend, threads, error_tolerance
3. **Seed**: the RNG seed (default entropy source, or explicit via `seed=<value>` argument)
4. **Algorithm version**: the Pynerve library version string

If all four inputs are identical, the output is guaranteed bit-identical.

### Determinism API

```python
import pynerve

# Default: non-deterministic across runs
result = pynerve.compute_persistence(points)

# Reproducible: pass seed argument
result = pynerve.compute_persistence(points, seed=42)
```

```cpp
#include <nerve/determinism.hpp>

nerve::determinism::seed(42);
uint64_t s = nerve::determinism::get_seed();   // 42
uint64_t r = nerve::determinism::next_seed();   // mt19937-64 value
```

### DeterminismContract (C++)

For fine-grained control:

```cpp
#include <nerve/core/determinism_contract.hpp>

auto contract = nerve::core::DeterminismEnforcer::createContract(
    nerve::core::DeterminismLevel::STRICT,
    "experiment_1"
);
contract.setRngSeed(42);
contract.enable_checksum_validation = true;
contract.record_intermediate_results = true;

// RAII context enforces contract for all operations within scope
{
    nerve::core::DeterminismContext ctx(contract);
    auto result = nerve::persistence::compute(points_view, dim, options);
    // Checksum validated automatically
}
```

### Contract Levels

`NONE` is reserved for internal use; determinism is always enabled in user-facing APIs. `BASIC` provides thread seeding, canonical filtration ordering, and deterministic CPU reduction. `STRICT` ensures full bitwise reproducibility with fixed-tree GPU reductions, no FMA, precise division and square root, and deterministic MPI reduction. `AUDIT` includes everything in STRICT plus checksum validation on all intermediate results and failure on any nondeterministic operation.

### What is NOT guaranteed

- **Cross-architecture bit-identity**: A computation on an AMD CPU may produce slightly different results than the same computation on an Intel CPU, due to differences in FMA and transcendental instruction implementations. STRICT mode minimizes this but cannot eliminate it for non-IEEE-754-compliant instructions.
- **Cross-GPU-architecture bit-identity without RFA**: Different GPU architectures (e.g., Ampere vs Hopper) may produce different results due to different reduction tree structures. Enable RFA (`NERVE_GPU_DETERMINISM=1`) for cross-GPU reproducibility.
- **Cross-MPI-process-count bit-identity**: Different numbers of MPI ranks change the order of floating-point accumulation. Use binned accumulation for cross-count reproducibility.
- **Parallel reducer pair-value determinism**: The lockfree (CPU multi-threaded) and HyphaReducer (GPU warp-level) persistence reducers are **non-deterministic at the pair-value level** due to racing pivot claims via atomic operations. Different runs of the same input produce different but topologically valid persistence diagrams. The lockfree reducer achieves **0.0000% count-level accuracy** vs the sequential ground truth, while the GPU reducer has a **~0.22% residual count error** (see [GPU Determinism](gpu_determinism.md)). For deterministic output at the pair-value level, use the sequential reducer.


[Back to Correctness Index](index.md)
