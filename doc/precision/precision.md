# Precision

## Quick start

```python
from pynerve.precision import PrecisionPolicyManager, PrecisionLevel

# Query current precision settings
pm = PrecisionPolicyManager()
level = pm.getGlobalLevel()
# -> PrecisionLevel.BALANCED

# Set precision for a specific algorithm
pm.setGlobalPrecisionLevel(PrecisionLevel.CONSERVATIVE)
prec = pm.getAlgorithmPrecision("vietoris_rips")
# -> NumericalPrecision.FP64

# Strict mode: downgrade is an error
pm.setStrictFpMode(True)
pm.enforcePrecisionPolicy("vietoris_rips", NumericalPrecision.FP64)
```

Mixed precision support (FP32/FP64) with numerical stability guarantees. The
policy framework automatically selects precision based on algorithm requirements,
available hardware, and stability constraints. Use float64 for reference
computation; use float32 for throughput when stability permits.


## API

```cpp
#include <nerve/precision/precision_policy.hpp>

namespace nerve::precision {

enum class PrecisionLevel {
    CONSERVATIVE = 0,   // FP64 everywhere
    BALANCED = 1,       // FP64 critical paths, FP32 elsewhere
    PERFORMANCE = 2,    // FP32 with FP64 accumulation
    ADVANCED = 3,       // FP16/FP32 mixed, auto-tuning
};

enum class NumericalPrecision {
    FP64 = 0,
    FP32 = 1,
    FP16 = 2,
    MIXED = 3,
    ADAPTIVE = 4,
};

enum class DowngradeReason {
    NONE,
    MEMORY_PRESSURE,
    PERFORMANCE_REQUIREMENT,
    NUMERICAL_INSTABILITY,
    USER_REQUEST,
    ALGORITHM_LIMITATION,
    BUDGET_CONSTRAINT,
};

struct AlgorithmPolicy {
    string algorithmName;
    PrecisionLevel defaultLevel;
    NumericalPrecision maxPrecision;
    NumericalPrecision minPrecision;
    bool allowMixedPrecision;
    bool allowAutomaticDowngrade;
    double stabilityThreshold;
};

class PrecisionPolicyManager {
    PrecisionPolicyManager();
    void setGlobalPrecisionLevel(PrecisionLevel level);
    void setStrictFpMode(bool strict);
    void registerAlgorithmPolicy(const AlgorithmPolicy& policy);
    NumericalPrecision getAlgorithmPrecision(const string& name) const;
    PrecisionState getPrecisionState(const string& name) const;
    vector<PolicyViolation> getPolicyViolations() const;
    bool validatePrecisionUsage(const string& name,
                                NumericalPrecision used) const;
    void enforcePrecisionPolicy(const string& name,
                                NumericalPrecision requested);
};

template <typename ComputationType>
class PrecisionAwareComputation {
    ResultType executeWithPrecisionPolicy(
        const InputType& input,
        NumericalPrecision precision = NumericalPrecision::FP64);
    ResultType executeWithValidation(const InputType& input,
                                     bool validate_result = true);
};

}
```

### When to use FP32 vs FP64

Reference and deterministic computations should use FP64 for bitwise reproducibility. High-dimensional distances (dimension greater than 100) also benefit from FP64 since the condition number grows with dimension. Batch inference can use FP32 for twice the throughput at half the memory cost. Training with gradient paths uses FP32 since it is sufficient for the SGD noise scale. Streaming aggregation requires FP64 to control accumulation error in sums. GPU memory-bound workloads should use FP32 so larger batches fit in VRAM.

### Stability guarantees

- `CONSERVATIVE` mode: all internal operations use FP64. No precision
  downgrade is permitted.
- `BALANCED` mode: distance computation and reduction use FP64; diagram
  post-processing may use FP32.
- `PERFORMANCE` mode: FP32 with FP64 accumulation in critical inner loops.
- Automatic downgrade records a `PolicyViolation` and updates the stability
  certificate's `precision_event_count`.


## FAQ

**Q: When should I use FP64 instead of FP32?**
A: Use FP64 for reference computations, deterministic mode, high-dimensional data (dimension greater than 100), and streaming aggregation where accumulation error matters. Use FP32 for batch inference, training, and GPU memory-constrained scenarios.

**Q: How do I detect precision loss in my computation?**
A: Enable strict mode with `setStrictFpMode(True)` and check `getPolicyViolations()` after computation. A non-empty violations list indicates precision downgrades occurred. You can also monitor the stability certificate's `precision_event_count`.

**Q: What does the BALANCED precision level actually do?**
A: BALANCED mode uses FP64 for distance computation and matrix reduction (the critical paths), while allowing FP32 for diagram post-processing and downstream analysis. This gives most of the accuracy benefit of FP64 with performance closer to FP32.
