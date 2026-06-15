# Features

Both PH4 and PH5 support standard reduction, cohomology reduction, clearing optimization, column compression, and a deterministic mode. PH5 extends these with extended clearing, advanced compression, and an enhanced deterministic mode. Additional PH5-only features include checksum validation via SHA-256 result checksums, multi-run stability checks, fine-grained determinism contracts, configurable numerical tolerance (default 1e-12), structured error logging, cross-run result validation, and differentiable PH5 operations for gradient support.

### Extended Clearing

PH5 implements dimension-cascading clearing that aggressively zeroes columns across the entire complex, not just within the current dimension. This reduces the active column count by 30-50% compared to standard clearing.

**How dimension-cascading clearing works**:

```
Algorithm: CASCADING_CLEAR(R, pivot, col)
    // Standard clearing: clear the birth column
    p = findPivot(R, col)
    clear(R, p)

    // Cascading: recursively clear columns that were paired
    // through this birth column
    while isColumnEmpty(R, p):
        // Check if column p's simplex was itself a death
        // (i.e., its birth column was cleared earlier)
        birth_of_p = pairBirth[p]
        if birth_of_p != -1 AND not isColumnCleared(R, birth_of_p):
            clear(R, birth_of_p)
            p = birth_of_p
        else:
            break
```

In practice, cascading clearing chains through 2-5 columns per death on average, clearing additional birth columns that would otherwise participate in future pivot conflicts.

**Clearing comparison**:

With no clearing, zero columns are cleared and correctness is always guaranteed, serving as the baseline. Standard clearing clears one column per death, achieving an estimated 20-40% speedup while remaining always correct. Dimension-cascading clearing clears approximately 2-5 columns per death on average, yielding an estimated 35-55% speedup, and is correct with careful tracking. The aggressive variant, which clears all descendants (roughly 5-15 columns per death), achieves an estimated 40-60% speedup but may clear incorrectly if dimension boundaries are not tracked.

PH5 uses dimension-cascading with dimension-boundary checks to ensure correctness. The aggressive variant is reserved for PH6 experimental mode.

### Advanced Compression

PH5's compression extends PH4's trailing-zero stripping with three additional techniques:

**1. Inter-column deduplication**: If two columns become identical during reduction, one is marked as a duplicate of the other:

```cpp
void deduplicate_columns(std::vector<Column>& columns, size_t col_a, size_t col_b) {
    if (columns[col_a] == columns[col_b]) {
        // Mark col_b as a duplicate of col_a
        columns[col_b].setDuplicateOf(col_a);
        columns[col_b].clear();  // free storage
    }
}
```

This is checked opportunistically after XOR operations. In practice, 5-15% of columns become duplicates and can share storage.

**2. Run-length encoded rows**: Columns with runs of consecutive ones are stored as (start, length) pairs:

```
Raw column (64 rows, bitset):      0b0011100111000000...
Run-length encoded:                (2,3), (6,2), ...   // start positions and lengths
```

This is beneficial for columns with densely clustered non-zero entries, which occur in alpha complexes and cubical complexes.

**3. Adaptive representation switching**: Each column can switch between representations based on its density:

The bitset representation is used when the density is above roughly one non-zero per 64 rows, with both storage and XOR costs of O(nwords). Sparse sorted indices are used when the density drops below that threshold, with costs of O(nnz). Run-length encoding targets columns with clustered non-zeros, incurring O(runs) for both storage and XOR. A duplicate reference representation has no density requirement and costs O(1) for storage and O(deref) for XOR.

The switching threshold is adaptive: after each XOR, if the column's density crosses the threshold, it is converted to the more efficient representation. This adds O(k) overhead per conversion (where k is the column size) but happens infrequently.

### Checksum Validation

Every PH5 computation produces a SHA-256 checksum of the result, which can be used for:

- Cross-version reproducibility
- Distributed computation validation
- CI/CD pipeline verification

```python
engine = PH5PH6Engine()
result = pynerve.compute_persistence_ph5(points, max_dim=2)
metrics = engine.getComputationMetrics()

# Checksum is available for downstream validation
checksum = metrics.result_checksum
print(f"Result checksum: {checksum.hex()}")
```

**What is checksummed**:

The SHA-256 hash is computed over a canonical serialization of:

1. Number of pairs (4 bytes, little-endian)
2. For each pair: birth filtration value (f64), death filtration value (f64), dimension (i32)
3. Betti numbers array (length = max_dim + 1, each i32)
4. Metadata string (library version, algorithm used, configuration)

The canonical serialization ensures cross-version compatibility: the same input data and same configuration produce the same checksum across library versions (as long as the result format is stable).

**Verification in CI/CD**:

```python
import hashlib
import json

# Stored expected checksum
with open("expected_checksum.json") as f:
    expected = json.load(f)

# Compute current result
result = pynerve.compute_persistence_ph5(points, max_dim=2)
engine = PH5PH6Engine()
checksum = engine.getComputationMetrics().result_checksum

assert checksum.hex() == expected["checksum"], \
    f"Checksum mismatch: got {checksum.hex()}, expected {expected['checksum']}"
```

### Determinism Contracts

PH5 supports multiple determinism levels through the `DeterminismContract`:

Four determinism levels are available. NONE provides no guarantees and no overhead, intended for development and debugging. BASIC guarantees the same output for the same input with negligible overhead, suitable for standard reproducibility. STRICT adds RNG seed control with under 5% overhead for research reproducibility. PARANOID includes cross-run validation with 10-20% overhead for regulated use.

**NONE**: No determinism guarantees. Thread scheduling may cause variations. Fastest mode.

**BASIC**: The default. Fixed seed for all RNG, deterministic OpenMP scheduling (`schedule(static)`), deterministic reduction tree. Produces the same result for the same input on the same hardware.

**STRICT**: All BASIC guarantees plus: explicit seed propagation through all random processes (witness sampling, tie-breaking), no auto-vectorization instructions that could vary by CPU model, and fixed-point arithmetic for filtration value comparisons.

**PARANOID**: All STRICT guarantees plus: runs the computation twice internally and compares results. If results differ, raises `DeterminismViolation` with diagnostic information. This catches hardware faults, memory corruption, and compiler-induced non-determinism.

```python
from pynerve.nn import PersistentHomology
# PH5 is used when ph5 engine is specified
ph = PersistentHomology(max_dim=2, reduction="clearing")
```

### Stability Testing

PH5 can run multiple computations on the same input and verify they produce identical results:

```python
engine = PH5PH6Engine()
engine.runStabilityTest(points, max_dimension=2, num_runs=10)
metrics = engine.getComputationMetrics()
print(f"Stable: {metrics.passed_stability_checks}")
print(f"Numerical errors: {metrics.numerical_errors}")
```

**What stability testing checks**:

1. **Bitwise reproducibility**: All 10 runs produce identical result checksums.
2. **Numerical stability**: For floating-point filtration values, checks that the tolerance-based ordering is consistent across runs.
3. **Memory consistency**: Verifies that no out-of-bounds writes occurred (using allocation guard bands).
4. **Thread safety**: Runs with varying thread counts (1, 2, 4, 8, ...) and verifies results match.

**When to use stability testing**:
- After upgrading hardware (new CPU, GPU)
- After compiler updates
- In CI/CD for each release
- When debugging intermittent failures in production

### Differentiable PH5 Operations

PH5 supports gradient-based learning via differentiable persistence operations:

```python
import torch
from pynerve.nn import PersistentHomology

ph = PersistentHomology(max_dim=2, reduction="clearing")

# Gradient flow through persistence computation
x = torch.randn(1, 100, 3, requires_grad=True)
diagrams = ph(x)  # list of tensors, one per dimension

# Each diagram is [batch, n_pairs, 2] (birth, death)
dim_1_pairs = diagrams[1]  # H1 pairs

# Compute a loss
loss = dim_1_pairs[..., 1].sum()  # sum of death values
loss.backward()  # gradients flow back to x
```

**How differentiation works**:

The PH5 differentiable backend implements the approach from Carriere et al. (2020) and Leygonie et al. (2021):

1. **Forward pass**: Standard persistent homology computation (exact, non-differentiable).
2. **Pairing stabilization**: The pairing is treated as a piecewise-constant function of the input. At points where the pairing changes (due to a simplex order swap), the gradient is zero almost everywhere.
3. **Subgradient computation**: When a simplex moves relative to its paired partner, the gradient is:

       dL / dx_i = dL / d(birth) * d(birth) / dx_i + dL / d(death) * d(death) / dx_i

   where d(birth)/dx_i and d(death)/dx_i are 1 if simplex i is the birth or death, respectively.

4. **Stabilization through tolerance**: PH5's numerical tolerance ensures that small perturbations do not change the pairing, making gradients well-defined.

**Performance impact of differentiability**:

Non-differentiable mode (PH4) has a forward time of 1x, no backward pass, and memory of 1x. Differentiable mode (PH5) has a forward time of roughly 1.2-1.5x, backward time of 1.5-2x, and memory of 1.3-1.5x. Differentiable mode with PARANOID determinism (PH5) has forward and backward times of 2-3x and memory of 2x.

The additional cost comes from storing the computation graph (which columns were XORed) for the backward pass.

### Structured Error Logging

PH5 provides detailed structured error logs:

```python
engine = PH5PH6Engine()
result = pynerve.compute_persistence_ph5(points, max_dim=2)
metrics = engine.getComputationMetrics()

# Access error log
for entry in metrics.error_log:
    print(f"[{entry.level}] {entry.timestamp}: {entry.message}")
    if entry.details:
        print(f"  Details: {entry.details}")
        print(f"  Location: {entry.file}:{entry.line}")
```

Error log entries include:
- **Numerical warnings**: Filtration values outside expected range
- **Memory warnings**: Approaching configured budget
- **Determinism violations**: Non-reproducible behavior detected
- **Convergence warnings**: Column reduction exceeding iteration limits
- **Performance warnings**: Unusually slow column operations

Back to [PH5 Engine Overview](index.md)
