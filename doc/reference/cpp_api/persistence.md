# `nerve::persistence` -- Persistence Computation

### Core Types

```cpp
#include <nerve/persistence/utils/api.hpp>

namespace nerve::persistence;

enum class PersistenceMode { EXACT, APPROX };
enum class PersistenceBackend { CPU_EXACT, CPU_ADAPTIVE_ACCELERATION, CUDA_HYBRID };

struct PersistenceOptions {
    PersistenceMode mode = PersistenceMode::EXACT;
    PersistenceBackend backend = PersistenceBackend::CPU_EXACT;
    Size max_dim = 2;
    double max_radius = 1.0;
    Size threads = 0;           // 0 = hardware_concurrency
    double error_tolerance = 1e-12;
};

struct PersistenceResult {
    std::vector<Pair> pairs;        // (birth, death, dimension)
    std::vector<Size> betti_numbers;
    PersistenceDiagnostics diagnostics;
};

struct Pair {
    double birth;
    double death;
    Dimension dimension;

    double lifetime() const noexcept;
    bool isInfinite() const noexcept;
};
```

**Cost semantics:**

**PersistenceOptions** construction is O(1) with no dynamic allocation. **PersistenceResult** construction is O(p) where p is the number of pairs, with pairs moved and diagnostics computed. **Pair** construction is O(1) -- just three scalars.

### Entry Points

```cpp
// General compute -- dispatches to best engine based on options
ErrorResult<PersistenceResult> compute(
    const core::BufferView<const double>& points,
    Size point_dim,
    const PersistenceOptions& options = {}
);
```

**Cost:** O(n^2 * d) distance + O(n^{d+1}) reduction.

**Errors:**

The function can return `E20_NUM_NAN` if NaN is encountered in points, `E50_PH_ABORT` if the engine aborted, or `E10_GPU_OOM` if the GPU runs out of memory (CUDA_HYBRID backend).

```cpp
// Specific engines
ErrorResult<PersistenceResult> computePersistencePh4(
    const core::BufferView<const double>& points,
    Size point_dim,
    const PersistenceOptions& options = {}
);

ErrorResult<PersistenceResult> computePersistencePh5(
    const core::BufferView<const double>& points,
    Size point_dim,
    const PersistenceOptions& options = {}
);

ErrorResult<PersistenceResult> computePersistencePh6(
    const core::BufferView<const double>& points,
    Size point_dim,
    const PersistenceOptions& options = {}
);

ErrorResult<PersistenceResult> computePersistenceCohomology(
    const core::BufferView<const double>& points,
    Size point_dim,
    const PersistenceOptions& options = {}
);

// Incremental updates
ErrorResult<PersistenceResult> updatePersistence(
    const std::vector<PersistenceEvent>& events,
    const PersistenceOptions& options = {}
);
```

**Cost (updatePersistence):** O(k * log m) per event where k = affected simplices, m = total simplices.

**Example:**

```cpp
#include <nerve/persistence/utils/api.hpp>
#include <nerve/core/buffer.hpp>
#include <vector>

std::vector<double> points = { /* n * dim values */ };
size_t n = 500, dim = 3;

auto view = nerve::core::BufferView<const double>(points.data(), points.size());
nerve::persistence::PersistenceOptions opts;
opts.max_dim = 2;
opts.max_radius = 2.0;
opts.backend = nerve::persistence::PersistenceBackend::CPU_ADAPTIVE_ACCELERATION;

auto result = nerve::persistence::compute(view, dim, opts);
if (result.is_error()) {
    // handle error
}
auto& pairs = result.value().pairs;  // std::vector<Pair>
```

### PH5PH6Engine

```cpp
#include <nerve/persistence/ph5_ph6.hpp>

namespace nerve::persistence;

class PH5PH6Engine {
    struct Config {
        double numerical_tolerance = 1e-12;
        size_t max_iterations = 1000;
        bool enable_stability_checks = false;
        bool validate_results = true;
        bool require_bitwise_reproducibility = true;
        bool enable_checksum_validation = false;
        std::string computation_id;
    };

    struct ComputationMetrics {
        double computation_time_ms;
        size_t peak_memory_bytes;
        size_t original_simplices;
        size_t final_simplices;
        double compression_ratio;
        double quality_score;
        bool passed_stability_checks;
        size_t numerical_errors;
        bool checksum_validation_passed;
    };

    explicit PH5PH6Engine(const Config& config = {});

    std::optional<ResultType> computePersistenceCohomology(
        const PointContainer& points, size_t max_dim);
    std::optional<ResultType> computePersistenceCompressedWitness(
        const PointContainer& points, size_t max_dim);
    std::optional<ResultType> computePersistenceBlockSparse(
        const PointContainer& points, size_t max_dim);
    std::optional<ResultType> computePersistenceHybrid(
        const PointContainer& points, size_t max_dim);

    ComputationMetrics getComputationMetrics() const;
    bool validateDeterministicResult(const PointContainer& points, size_t max_dim);
    bool runStabilityTest(const PointContainer& points, size_t max_dim, size_t num_runs = 10);
};
```

**Cost (computePersistenceCohomology):** O(n^{d+1}) worst-case, typically O(n^2) with clearing.

**Cost (computePersistenceBlockSparse):** O(k^{d+1}) for k landmarks, k << n.

**Cost (runStabilityTest):** O(num_runs * base_computation_cost).

<- [C++ API Overview](index.md)
