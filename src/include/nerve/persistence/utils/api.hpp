
#pragma once

#include "nerve/algebra/simplex.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <vector>

namespace nerve::persistence
{

enum class PersistenceMode
{
    EXACT,
    APPROX
};

enum class PersistenceBackend
{
    CPU_EXACT,
    CPU_ADAPTIVE_ACCELERATION,
    CUDA_HYBRID
};

struct PersistenceOptions
{
    PersistenceMode mode = PersistenceMode::EXACT;
    PersistenceBackend backend = PersistenceBackend::CPU_EXACT;
    Size max_dim = 2;
    double max_radius = 1.0;
    Size threads = 0;
    double error_tolerance = 1e-12;
};

struct PersistenceEvent
{
    enum class Type
    {
        ADD_SIMPLEX,
        REMOVE_SIMPLEX
    };
    Type type = Type::ADD_SIMPLEX;
    algebra::Simplex simplex;
};

struct PersistenceDiagnostics
{
    double elapsed_ms = 0.0;
    Size operations = 0;
    Size pairs = 0;
    Size memory_bytes = 0;
    PersistenceBackend backend = PersistenceBackend::CPU_EXACT;
    PersistenceMode mode = PersistenceMode::EXACT;
    bool approximation_applied = false;
    double approximation_tolerance = 0.0;
};

struct PersistenceResult
{
    std::vector<Pair> pairs;
    std::vector<Size> betti_numbers;
    PersistenceDiagnostics diagnostics;
};

errors::ErrorResult<PersistenceResult> compute(core::BufferView<const double> points,
                                               Size point_dim,
                                               const PersistenceOptions &options = {});

errors::ErrorResult<PersistenceResult>
updatePersistence(const std::vector<PersistenceEvent> &events,
                  const PersistenceOptions &options = {});

// Algorithm entrypoints routed to the canonical engine.
errors::ErrorResult<PersistenceResult>
computePersistencePh4(core::BufferView<const double> points, Size point_dim,
                      const PersistenceOptions &options = {});

errors::ErrorResult<PersistenceResult>
computePersistencePh5(core::BufferView<const double> points, Size point_dim,
                      const PersistenceOptions &options = {});

errors::ErrorResult<PersistenceResult>
computePersistencePh6(core::BufferView<const double> points, Size point_dim,
                      const PersistenceOptions &options = {});

errors::ErrorResult<PersistenceResult>
computePersistenceCohomology(core::BufferView<const double> points, Size point_dim,
                             const PersistenceOptions &options = {});

} // namespace nerve::persistence
