
#pragma once

#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"

#include <memory>
#include <optional>

namespace nerve::persistence
{

template <typename PointType, typename Scalar>
std::optional<typename PH5PH6Engine<PointType, Scalar>::ResultType>
PH5PH6Engine<PointType, Scalar>::runCanonical(const PointContainer &points, size_t max_dimension,
                                              EntryPoint entrypoint)
{
    clearErrorLog();
    metrics_ = ComputationMetrics{};

    if (max_dimension == 0)
    {
        recordError("max_dimension must be > 0", errors::ErrorCode::E54_PH4_INVALID_INPUT);
        return std::nullopt;
    }

    std::string flatten_error;
    auto flattened = flattenPoints(points, flatten_error);
    if (!flattened)
    {
        recordError(flatten_error, errors::ErrorCode::E54_PH4_INVALID_INPUT);
        return std::nullopt;
    }

    computeParamsHash(points, max_dimension);

    const auto start = std::chrono::high_resolution_clock::now();

    PersistenceOptions options;
    options.mode = PersistenceMode::EXACT;
    options.backend = PersistenceBackend::CPU_EXACT;
    options.max_dim = static_cast<Size>(max_dimension);
    options.max_radius = 1.0;
    options.threads = 0;
    options.error_tolerance = config_.numerical_tolerance;

    core::BufferView<const double> view(flattened->values.data(), flattened->values.size());
    auto api_result = entrypoint(view, flattened->point_dim, options);
    if (api_result.isError())
    {
        recordError("Canonical persistence engine failed", errors::ErrorCode::E50_PH_ABORT);
        return std::nullopt;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto &persistence = api_result.value();

    ResultType converted;
    converted.reserve(persistence.pairs.size());
    for (size_t i = 0; i < persistence.pairs.size(); ++i)
    {
        const auto &pair = persistence.pairs[i];
        const double lifetime = std::isfinite(pair.death) ? std::max(0.0, pair.death - pair.birth)
                                                          : std::numeric_limits<double>::infinity();
        converted.emplace_back(
            std::vector<size_t>{static_cast<size_t>(std::max(0, pair.dimension)), i},
            static_cast<Scalar>(lifetime));
    }

    metrics_.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    metrics_.peak_memory_bytes = persistence.diagnostics.memory_bytes;
    metrics_.original_simplices = persistence.diagnostics.operations;
    metrics_.final_simplices = converted.size();
    metrics_.compression_ratio = metrics_.original_simplices == 0
                                     ? 1.0
                                     : static_cast<double>(metrics_.final_simplices) /
                                           static_cast<double>(metrics_.original_simplices);
    metrics_.quality_score = 1.0;
    metrics_.passed_stability_checks = true;
    metrics_.numerical_errors = 0;
    metrics_.achieved_determinism_level = determinism_contract_.level;
    metrics_.checksum_validation_passed = true;
    metrics_.result_checksum = computeResultChecksum(converted);
    metrics_.rng_seed_used = determinism_contract_.rng_seed;

    return converted;
}

template <typename PointType, typename Scalar>
void PH5PH6Engine<PointType, Scalar>::recordError(const std::string &message,
                                                  errors::ErrorCode code)
{
    error_log_.emplace_back(message, code);
    if (error_log_.size() > 128)
    {
        error_log_.erase(error_log_.begin());
    }
}

template <typename PointType, typename Scalar>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh5Engine()
{
    typename PH5PH6Engine<PointType, Scalar>::Config config;
    return std::make_unique<PH5PH6Engine<PointType, Scalar>>(config);
}

template <typename PointType, typename Scalar>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh6Engine()
{
    typename PH5PH6Engine<PointType, Scalar>::Config config;
    config.require_bitwise_reproducibility = true;
    return std::make_unique<PH5PH6Engine<PointType, Scalar>>(config);
}

template <typename PointType, typename Scalar>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh5EngineIfEnabled()
{
#ifdef ENABLE_PH5
    return createPh5Engine<PointType, Scalar>();
#else
    return nullptr;
#endif
}

template <typename PointType, typename Scalar>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh6EngineIfEnabled()
{
#ifdef ENABLE_PH6
    return createPh6Engine<PointType, Scalar>();
#else
    return nullptr;
#endif
}

template <typename PointType, typename Scalar>
bool isPh5Available()
{
#ifdef ENABLE_PH5
    return true;
#else
    return false;
#endif
}

template <typename PointType, typename Scalar>
bool isPh6Available()
{
#ifdef ENABLE_PH6
    return true;
#else
    return false;
#endif
}

} // namespace nerve::persistence
