## Determinism

Validate that persistence computations are bitwise reproducible across runs.

```python
from pynerve.validation import check_determinism, DeterminismLevel

result1 = pynerve.compute_persistence(points, max_dim=2)
result2 = pynerve.compute_persistence(points, max_dim=2)

check = check_determinism(result1, result2, level=DeterminismLevel.STRICT)
# check.is_deterministic: bool
# check.errors: list[str]
# check.mismatches: list[{"field": str, "first": ..., "second": ...}]
# check.checksum_first, check.checksum_second: str
```

### Tolerance

```python
check = check_determinism(result1, result2, tolerance=1e-12)
```

### Determinism levels

Four determinism levels are available. **NONE** provides no guarantees and may use atomics. **BASIC** applies thread seeding and canonical filtration. **STRICT** guarantees bitwise reproducibility with fixed-tree GPU reductions. **AUDIT** extends STRICT with checksums and intermediate recording.

### Determinism contract

```python
from pynerve.validation import DeterminismContract, DeterminismEnforcer

contract = DeterminismEnforcer.create_contract(DeterminismLevel.STRICT, "my_analysis")
contract.set_rng_seed(42)
ok = DeterminismEnforcer.can_satisfy_contract(contract)
violations = DeterminismEnforcer.get_contract_violations(contract)
```

### Validation functions

```python
from pynerve.validation import (
    validate_mathematical_correctness,
    validate_betti_numbers,
    validate_topological_invariants,
    validate_computation_determinism,
)

valid = validate_mathematical_correctness(diagram, contract)
valid = validate_betti_numbers(diagram, expected_betti=[1, 0, 1])
valid = validate_topological_invariants(diagram)
valid = validate_computation_determinism(result_a, result_b, contract)
```


## Sources of non-determinism

Non-determinism can arise from several sources in the computation pipeline:

- **Atomic operations** -- occur in GPU cohomology reduction and histograms; mitigated by using `DeterminismLevel::AUDIT` or switching to CPU.
- **Floating-point associativity** -- affects parallel reduction where tree vs linear order differs; addressed with a fixed reduction tree.
- **Hash table iteration** -- occurs in simplex storage via `unordered_map`; mitigated by using sorted iteration.
- **RNG seeding** -- affects landmark selection in witness complex; fix by setting `rng_seed` explicitly.
- **OpenMP scheduling** -- affects parallel for loops; mitigated by calling `omp_set_dynamic(0)`.
- **CUDA block scheduling** -- affects `__syncthreads`-dependent kernels; fix by using a fixed block count.

### GPU determinism flags

```cpp
// Enable deterministic GPU operations
#include <nerve/cuda/determinism.hpp>

nerve::cuda::setDeterministicMode(true);
// Sets:
//   CUBLAS_DETERMINISTIC=1
//   CUDNN_DETERMINISTIC=1
//   CUDA_LAUNCH_BLOCKING=1
```

### Testing determinism across hardware

```python
from pynerve.validation import check_determinism_cross_platform

# Run on two different GPU architectures
result_a = pynerve.compute_persistence(points, backend="cuda")  # A100
result_b = pynerve.compute_persistence(points, backend="cuda")  # H100

check = check_determinism(result_a, result_b,
                          level=DeterminismLevel.STRICT)
# May show minor floating-point differences due to different
# Tensor Core FP16 accumulation paths
```

Tolerance of 1e-6 is typically sufficient for cross-architecture reproducibility.

### Determinism contract example

```python
from pynerve.validation import (
    DeterminismContract, DeterminismEnforcer,
    validate_computation_determinism,
)

# Create a strict contract
contract = DeterminismEnforcer.create_contract(
    DeterminismLevel.STRICT, "production_analysis"
)
contract.set_rng_seed(42)
contract.add_constraint("gpu_atomics", False)
contract.add_constraint("parallel_reduction", "fixed_tree")

# Verify the environment can satisfy it
if not DeterminismEnforcer.can_satisfy_contract(contract):
    violations = DeterminismEnforcer.get_contract_violations(contract)
    for v in violations:
        print(f"Cannot satisfy: {v.description}")
    # Fall back to BASIC level
    contract.set_level(DeterminismLevel.BASIC)

# Run with contract
result1 = pynerve.compute_persistence(points)
result2 = pynerve.compute_persistence(points)

# Validate
valid = validate_computation_determinism(result1, result2, contract)
assert valid, "Non-deterministic result detected!"
```

### Integration with serialization

```python
# Store checksum alongside serialized diagram
from pynerve.validation import compute_checksum
from pynerve.serialization import save

diagram = pynerve.compute_persistence(points)
digest = compute_checksum(diagram)

save("diagram.nvf", diagram, metadata={
    "checksum": digest,
    "determinism_level": "STRICT",
})

# On load, verify integrity
loaded = load("diagram.nvf")
assert loaded.metadata["checksum"] == compute_checksum(loaded)
```


## FAQ

**Q: Why does my computation produce different results on CPU vs GPU?**
A: This is expected. GPU floating-point arithmetic uses different associativity than CPU (especially with Tensor Core FMAs). For bitwise reproducibility, use the same backend for all comparisons. For cross-backend comparisons, use a tolerance of 1e-5.

**Q: How much slower is AUDIT mode?**
A: AUDIT mode forces fixed-tree reductions and disables atomics, typically 10-30% slower than STRICT mode. BASIC mode adds minimal overhead (<5%).

**Q: Can I get determinism without the performance hit?**
A: Use STRICT mode with a fixed RNG seed and disable atomics. This avoids the most common sources of non-determinism while keeping most GPU optimizations enabled.


### Cross-references

- `pynerve.validation`: Validation overview
- `pynerve.cuda.determinism`: GPU determinism compiler flags
- `pynerve.core.rng`: Deterministic RNG
- `pynerve.validation.benchmarks`: Performance overhead of determinism
