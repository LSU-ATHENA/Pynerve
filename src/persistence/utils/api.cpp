
#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/utils/diagram_statistics.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <string_view>

namespace nerve::persistence
{

namespace
{

void applyApproximationPolicy(std::vector<Pair> &pairs, double error_tolerance)
{
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [&](const Pair &p) {
                                   if (p.isInfinite())
                                   {
                                       return false;
                                   }
                                   return std::fabs(p.death - p.birth) < error_tolerance;
                               }),
                pairs.end());
}

errors::ErrorResult<PersistenceResult> inputError(std::string_view message)
{
    return errors::ErrorResult<PersistenceResult>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT,
                                                         message);
}

errors::ErrorResult<PersistenceResult> validateOptions(const PersistenceOptions &options)
{
    if (!std::isfinite(options.max_radius) &&
        options.max_radius != std::numeric_limits<double>::infinity())
    {
        return inputError("persistence max_radius must be finite and non-negative");
    }
    if (options.max_radius < 0.0)
    {
        return inputError("persistence max_radius must be non-negative");
    }
    if (!std::isfinite(options.error_tolerance) || options.error_tolerance < 0.0)
    {
        return inputError("persistence error_tolerance must be finite and non-negative");
    }
    return errors::ErrorResult<PersistenceResult>::success(PersistenceResult{});
}

errors::ErrorResult<PersistenceResult>
validatePointCloud(const core::BufferView<const double> &points, Size point_dim)
{
    if (point_dim == 0)
    {
        return inputError("point dimension must be positive");
    }
    if (points.empty())
    {
        return inputError("point buffer must not be empty");
    }
    if ((points.size() % point_dim) != 0)
    {
        return inputError("point buffer size must be divisible by point dimension");
    }
    for (double value : points)
    {
        if (!std::isfinite(value))
        {
            return inputError("point coordinates must be finite");
        }
    }
    return errors::ErrorResult<PersistenceResult>::success(PersistenceResult{});
}

bool simplexIsValid(const algebra::Simplex &simplex)
{
    if (simplex.numVertices() == 0)
    {
        return false;
    }
    return std::ranges::all_of(simplex.vertices(), [](Index vertex) { return vertex >= 0; });
}

bool hasRequiredFaces(const algebra::SimplicialComplex &complex, const algebra::Simplex &simplex)
{
    if (simplex.dimension() == 0)
    {
        return true;
    }
    const auto existing = complex.getSimplices();
    for (const auto &face : simplex.faces({}))
    {
        if (std::ranges::find(existing, face) == existing.end())
        {
            return false;
        }
    }
    return true;
}

errors::ErrorResult<PersistenceResult>
computeWithDimCap(const core::BufferView<const double> &points, Size point_dim,
                  const PersistenceOptions &options, Size dim_cap)
{
    PersistenceOptions wrapped = options;
    wrapped.max_dim = std::min(options.max_dim, dim_cap);
    return compute(points, point_dim, wrapped);
}

} // namespace

errors::ErrorResult<PersistenceResult> compute(const core::BufferView<const double> &points,
                                               Size point_dim, const PersistenceOptions &options)
{
    if (auto validation = validateOptions(options); validation.isError())
    {
        return validation;
    }
    if (auto validation = validatePointCloud(points, point_dim); validation.isError())
    {
        return validation;
    }
    const PersistenceOptions normalized = options;
    const auto start = std::chrono::high_resolution_clock::now();

    VRConfig fast_config;
    fast_config.max_dim = normalized.max_dim;
    fast_config.max_radius = normalized.max_radius;
    fast_config.num_threads = normalized.threads;
    fast_config.enable_approximation = (normalized.mode == PersistenceMode::APPROX);
    fast_config.approximation_error_budget = normalized.error_tolerance;
    fast_config.approximation_error_tolerance = normalized.error_tolerance;

    // Configure acceleration based on backend
    const PersistenceBackend resolved_backend = normalized.backend;
    const bool use_acceleration =
        (resolved_backend == PersistenceBackend::CPU_ADAPTIVE_ACCELERATION ||
         resolved_backend == PersistenceBackend::CUDA_HYBRID);
    fast_config.use_accelerated_runtime = use_acceleration;
    fast_config.auto_detect_accelerated_runtime = use_acceleration;
    fast_config.use_adaptive_acceleration = use_acceleration;
    fast_config.auto_detect_adaptive_acceleration = use_acceleration;
    (resolved_backend == PersistenceBackend::CUDA_HYBRID);
    (resolved_backend == PersistenceBackend::CUDA_HYBRID);

    std::vector<Pair> pairs;
    try
    {
        pairs = computeVrPersistenceFast(points, point_dim, fast_config);
    }
    catch (const std::exception &ex)
    {
        return errors::ErrorResult<PersistenceResult>::error(errors::ErrorCode::E50_PH_ABORT,
                                                             ex.what());
    }
    // Always filter zero-persistence pairs (birth == death)
    applyApproximationPolicy(pairs, 1e-12);
    if (normalized.mode == PersistenceMode::APPROX)
    {
        applyApproximationPolicy(pairs, normalized.error_tolerance);
    }

    PersistenceResult result;
    result.pairs = std::move(pairs);
    result.betti_numbers = bettiNumbersFromPairs(result.pairs);
    result.diagnostics.pairs = result.pairs.size();
    result.diagnostics.operations = result.pairs.size();
    result.diagnostics.memory_bytes =
        points.size() * sizeof(double) + result.pairs.size() * sizeof(Pair);
    result.diagnostics.backend = resolved_backend;
    result.diagnostics.mode = normalized.mode;
    result.diagnostics.approximation_applied = (normalized.mode == PersistenceMode::APPROX);
    result.diagnostics.approximation_tolerance = normalized.error_tolerance;
    const auto end = std::chrono::high_resolution_clock::now();
    result.diagnostics.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return errors::ErrorResult<PersistenceResult>::success(std::move(result));
}

errors::ErrorResult<PersistenceResult>
updatePersistence(const std::vector<PersistenceEvent> &events, const PersistenceOptions &options)
{
    if (auto validation = validateOptions(options); validation.isError())
    {
        return validation;
    }
    const PersistenceOptions normalized = options;
    const auto start = std::chrono::high_resolution_clock::now();
    algebra::SimplicialComplex complex;

    for (const auto &event : events)
    {
        if (!simplexIsValid(event.simplex))
        {
            return inputError("persistence event simplex must be non-empty and non-negative");
        }
        try
        {
            if (event.type == PersistenceEvent::Type::ADD_SIMPLEX)
            {
                if (!hasRequiredFaces(complex, event.simplex))
                {
                    return inputError(
                        "persistence event stream must add simplex faces before cofaces");
                }
                complex.addSimplex(event.simplex);
            }
            else
            {
                complex.removeSimplex(event.simplex);
            }
        }
        catch (const std::exception &ex)
        {
            return errors::ErrorResult<PersistenceResult>::error(errors::ErrorCode::E50_PH_ABORT,
                                                                 ex.what());
        }
    }

    auto exact = computeExactPersistenceZ2(complex, normalized.max_dim);
    auto pairs = std::move(exact.pairs);

    if (normalized.mode == PersistenceMode::APPROX)
    {
        applyApproximationPolicy(pairs, normalized.error_tolerance);
    }

    PersistenceResult result;
    result.pairs = std::move(pairs);
    result.betti_numbers = bettiNumbersFromPairs(result.pairs);
    result.diagnostics.pairs = result.pairs.size();
    result.diagnostics.operations = exact.reduction_operations;
    result.diagnostics.memory_bytes =
        events.size() * sizeof(PersistenceEvent) + result.pairs.size() * sizeof(Pair);
    result.diagnostics.backend = PersistenceBackend::CPU_EXACT;
    result.diagnostics.mode = normalized.mode;
    result.diagnostics.approximation_applied = (normalized.mode == PersistenceMode::APPROX);
    result.diagnostics.approximation_tolerance = normalized.error_tolerance;
    const auto end = std::chrono::high_resolution_clock::now();
    result.diagnostics.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return errors::ErrorResult<PersistenceResult>::success(std::move(result));
}

errors::ErrorResult<PersistenceResult>
computePersistencePh4(const core::BufferView<const double> &points, Size point_dim,
                      const PersistenceOptions &options)
{
    return computeWithDimCap(points, point_dim, options, 4);
}

errors::ErrorResult<PersistenceResult>
computePersistencePh5(const core::BufferView<const double> &points, Size point_dim,
                      const PersistenceOptions &options)
{
    return computeWithDimCap(points, point_dim, options, 5);
}

errors::ErrorResult<PersistenceResult>
computePersistencePh6(const core::BufferView<const double> &points, Size point_dim,
                      const PersistenceOptions &options)
{
    return computeWithDimCap(points, point_dim, options, 6);
}

errors::ErrorResult<PersistenceResult>
computePersistenceCohomology(const core::BufferView<const double> &points, Size point_dim,
                             const PersistenceOptions &options)
{
    PersistenceOptions cohomology_options = options;
    cohomology_options.backend = PersistenceBackend::CPU_ADAPTIVE_ACCELERATION;
    return compute(points, point_dim, cohomology_options);
}

} // namespace nerve::persistence
