# Deterministic Computation

Pynerve is deterministic by default. Given the same input data, parameters, and execution environment, repeated calls to `compute_persistence` produce identical results.

Pynerve guarantees **bitwise reproducibility**: the returned persistence results are identical at the binary level, not just mathematically equivalent.

Deterministic execution is useful for:

- reproducible scientific experiments
- regression testing
- debugging numerical issues
- comparing results across development environments

No additional configuration is required.

```python
import pynerve
import numpy as np

points = np.random.randn(100, 3)

result1 = pynerve.compute_persistence(points, max_dim=2)
result2 = pynerve.compute_persistence(points, max_dim=2)

assert result1 == result2
```

## What determinism means

A deterministic run guarantees that:

- The same input produces the same output.
- Parallel execution order does not affect results.
- Floating-point rounding behavior is controlled.
- GPU execution does not introduce run-to-run variation.

Determinism applies across all supported CPU and GPU backends.

## CPU implementation

On CPU, Pynerve maintains determinism through controlled randomness and ordered numerical operations.

### Random number generation

Pynerve does not rely on implicit randomness. All random operations use an explicitly seeded generator.

The internal generator:

- uses `std::mt19937_64`
- requires an explicit seed
- never falls back to time-based seeding

See [RNG](../core/rng.md) for details.

### Parallel execution

Thread-level randomness is deterministic:

- A single master seed is created.
- Each worker thread receives a deterministic derived seed.
- Thread scheduling does not change generated values.

### Numerical reductions

Parallel reductions use fixed ordering rules that prevent floating-point non-associativity from causing variation:

- filtration order is preserved
- pivot selection follows deterministic tie-breaking
- reduction trees are fixed

## GPU implementation

GPU execution is deterministic by default, but requires additional controls because GPU hardware executes many operations concurrently.

Pynerve ensures deterministic GPU results through:

- fixed reduction trees
- deterministic warp-level operations
- avoidance of non-deterministic atomic reductions
- controlled floating-point behavior

GPU builds enforce deterministic compiler settings:

```
--fmad=false
--prec-div=true
--prec-sqrt=true
--ftz=false
```

These settings prevent compiler transformations that can change floating-point results.

### How it works: deterministic warp reduction

Inside each warp, reductions use a fixed butterfly pattern:

```cpp
unsigned mask = 0xffffffff;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(mask, val, offset, 32);
}
```

## Performance considerations

Deterministic GPU execution requires additional synchronization and fixed-order operations.

Typical overhead:

| Backend | Expected overhead |
|---------|------------------|
| CPU | <1% |
| GPU | ~20-30% |

If maximum throughput is more important than reproducibility, deterministic execution can be disabled.

## Validating determinism

Pynerve provides a validation API for applications that require explicit reproducibility checks.

```python
from pynerve.validation import DeterminismLevel, DeterminismContract

contract = DeterminismContract(
    level=DeterminismLevel.STRICT,
    seed=42,
    require_bitwise_reproducible=True,
)

r1 = compute_persistence(points)
r2 = compute_persistence(points)

assert r1 == r2
```

## Common issues

### GPU synchronization

GPU operations are asynchronous. Reading results before computation finishes can produce incomplete results. Pynerve automatically synchronizes GPU execution before returning results.

### External GPU libraries

Some CUDA libraries have their own determinism requirements. For deterministic cuBLAS operations, configure:

```
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Equal filtration values

When multiple simplices have identical filtration values, their ordering must be resolved deterministically. Pynerve applies deterministic tie-breaking rules. If you need custom behavior, configure the error tolerance parameter:

```python
result = pynerve.compute_persistence(points, error_tolerance=1e-9)
```

## Frequently asked questions

### Why is determinism important?

Persistent homology computations involve many floating-point operations and parallel reductions. Small numerical differences can otherwise lead to different outputs across runs. Deterministic execution ensures experiments are reproducible and results can be compared reliably.

### Does determinism change the mathematical result?

No. Determinism does not change the algorithm. It only ensures that the same algorithm produces the same output every time.

### What is the performance cost of determinism?

On CPU the overhead is negligible (under 1%). On GPU the overhead is approximately 20-30% due to `--fmad=false` and fixed-tree reductions. This is a deliberate trade-off: reproducibility is the default, and users can opt in to maximum performance by disabling determinism checks.

## Further reading

- [Deterministic RNG](../core/rng.md)
- [GPU determinism](../cuda/determinism.md)
- [Validation framework](../validation/determinism.md)
