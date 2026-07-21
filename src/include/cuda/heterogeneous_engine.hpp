#pragma once

#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/gpu_compute.hpp"
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nerve::gpu
{

inline constexpr std::size_t BYTES_PER_KB = 1024ULL;
inline constexpr std::size_t BYTES_PER_MB = 1024ULL * 1024ULL;
inline constexpr std::size_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t HETEROGENEOUS_MAX_MEMORY = 4ULL * BYTES_PER_GB;

struct WorkDistribution
{
    static constexpr double GPU_RATIO = 0.2;

    nerve::Size gpu_columns = 0;
    nerve::Size cpu_columns = 0;

    static WorkDistribution compute(nerve::Size total_columns)
    {
        if (!nerve::gpu::isAvailable())
        {
            return {.gpu_columns = 0, .cpu_columns = total_columns};
        }
        const nerve::Size gpu_work = static_cast<nerve::Size>(total_columns * GPU_RATIO);
        return {
            .gpu_columns = gpu_work,
            .cpu_columns = total_columns - gpu_work,
        };
    }
};

class HeterogeneousEngine
{
public:
    struct Config
    {
        nerve::Size gpu_memory_mb = 512;
        int max_dimension = 2;
        double max_radius = 1.0;
        bool enable_gpu = true;
        nerve::Size gpu_threshold_points = 1000;
        bool enable_gpu_recovery = false;
    };

    struct PerformanceStats
    {
        double gpu_time_ms = 0.0;
        double cpu_time_ms = 0.0;
        double total_time_ms = 0.0;
        nerve::Size gpu_columns_processed = 0;
        nerve::Size cpu_columns_processed = 0;
        double memory_utilization = 0.0;
    };

    static nerve::error::Result<std::unique_ptr<HeterogeneousEngine>> create()
    {
        return create(Config{});
    }

    static nerve::error::Result<std::unique_ptr<HeterogeneousEngine>> create(Config cfg)
    {
        auto config_status = validateConfig(cfg);
        if (config_status.isErr())
        {
            return nerve::error::Result<std::unique_ptr<HeterogeneousEngine>>::err(
                static_cast<nerve::error::TDAErrorCode>(config_status.error().value()),
                std::string(config_status.detail()));
        }
        auto backend_cfg = toBackendConfig(cfg);
        auto backend_result =
            nerve::persistence::accelerated::HeterogeneousFastVR::create(backend_cfg);
        if (backend_result.isError())
        {
            return nerve::error::Result<std::unique_ptr<HeterogeneousEngine>>::err(
                mapBackendError(backend_result.errorCode()),
                "failed to create heterogeneous backend: " +
                    errorCodeString(backend_result.errorCode()));
        }

        auto engine = std::unique_ptr<HeterogeneousEngine>(
            new HeterogeneousEngine(cfg, backend_result.moveValue()));
        return nerve::error::Result<std::unique_ptr<HeterogeneousEngine>>::ok(std::move(engine));
    }

    nerve::error::Result<std::vector<nerve::Pair>>
    computeVrPersistence(const double *points, nerve::Size n_points, nerve::Size dim)
    {
        if (points == nullptr || n_points == 0 || dim == 0)
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                nerve::error::TDAErrorCode::InvalidInput,
                "points pointer must be non-null and dimensions must be positive");
        }
        if (backend_ == nullptr)
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                nerve::error::TDAErrorCode::ResourceLimit,
                "heterogeneous backend is not initialized");
        }

        if (n_points > std::numeric_limits<nerve::Size>::max() / dim)
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                nerve::error::TDAErrorCode::ResourceLimit, "point buffer size overflows size_t");
        }
        const nerve::Size total_values = n_points * dim;
        for (nerve::Size i = 0; i < total_values; ++i)
        {
            if (!std::isfinite(points[i]))
            {
                return nerve::error::Result<std::vector<nerve::Pair>>::err(
                    nerve::error::TDAErrorCode::NaNInInput, "point coordinates must be finite");
            }
        }
        nerve::core::BufferView<const double> view(points, total_values);

        auto backend_result = backend_->computeVrPersistence(view, dim, {});
        updateStats(n_points);
        if (backend_result.isError())
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                mapBackendError(backend_result.errorCode()),
                "heterogeneous backend failed: " + errorCodeString(backend_result.errorCode()));
        }

        auto backend_pairs = backend_result.moveValue();
        std::vector<nerve::Pair> pairs;
        pairs.reserve(backend_pairs.size());
        for (const auto &pair : backend_pairs)
        {
            pairs.push_back(nerve::Pair(pair.birth, pair.death, pair.dimension));
        }
        return nerve::error::Result<std::vector<nerve::Pair>>::ok(std::move(pairs));
    }

    nerve::error::Result<std::vector<nerve::Pair>>
    computeVrPersistence(const std::vector<double> &points, nerve::Size point_dim)
    {
        if (point_dim == 0 || points.empty() || points.size() % point_dim != 0)
        {
            return nerve::error::Result<std::vector<nerve::Pair>>::err(
                nerve::error::TDAErrorCode::InvalidInput,
                "point vector must be non-empty and divisible by point_dim");
        }
        return computeVrPersistence(points.data(), points.size() / point_dim, point_dim);
    }

    PerformanceStats getLastStats() const noexcept { return stats_; }

private:
    explicit HeterogeneousEngine(
        Config cfg, std::unique_ptr<nerve::persistence::accelerated::HeterogeneousFastVR> backend)
        : cfg_(cfg)
        , backend_(std::move(backend))
    {}

    static nerve::persistence::accelerated::HeterogeneousFastVR::Config
    toBackendConfig(const Config &cfg)
    {
        nerve::persistence::accelerated::HeterogeneousFastVR::Config backend_cfg;
        backend_cfg.num_threads = 0;
        backend_cfg.max_radius = cfg.max_radius;
        backend_cfg.max_dim = static_cast<std::size_t>(std::max(0, cfg.max_dimension));
        return backend_cfg;
    }

    static nerve::error::Result<void> validateConfig(const Config &cfg)
    {
        if (!std::isfinite(cfg.max_radius) || cfg.max_radius <= 0.0)
        {
            return nerve::error::Result<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                                   "max_radius must be finite and greater than 0");
        }
        if (cfg.max_dimension < 0)
        {
            return nerve::error::Result<void>::err(nerve::error::TDAErrorCode::InvalidDimension,
                                                   "max_dimension must be non-negative");
        }
        return nerve::error::Result<void>::ok();
    }

    static nerve::error::TDAErrorCode mapBackendError(nerve::errors::ErrorCode code) noexcept
    {
        switch (code)
        {
            case nerve::errors::ErrorCode::E10_GPU_OOM:
                return nerve::error::TDAErrorCode::AllocationFailed;
            case nerve::errors::ErrorCode::E11_GPU_LAUNCH_FAIL:
                return nerve::error::TDAErrorCode::ReductionFailed;
            case nerve::errors::ErrorCode::E20_NUM_NAN:
                return nerve::error::TDAErrorCode::NaNInInput;
            case nerve::errors::ErrorCode::E50_PH_ABORT:
                return nerve::error::TDAErrorCode::PHAbort;
            case nerve::errors::ErrorCode::E41_RESOURCE_LIMIT:
                return nerve::error::TDAErrorCode::ResourceLimit;
            default:
                return nerve::error::TDAErrorCode::Unknown;
        }
    }

    static std::string errorCodeString(nerve::errors::ErrorCode code)
    {
        return std::to_string(static_cast<unsigned int>(code));
    }

    void updateStats(nerve::Size n_points)
    {
        const auto backend_stats = backend_->getPerformanceStats();
        stats_.gpu_time_ms = backend_stats.gpu_time_ms;
        stats_.cpu_time_ms = backend_stats.cpu_time_ms;
        stats_.total_time_ms = backend_stats.total_time_ms;
        stats_.gpu_columns_processed = backend_stats.gpu_used ? n_points : 0;
        stats_.cpu_columns_processed = backend_stats.gpu_used ? 0 : n_points;
        if (cfg_.gpu_memory_mb > 0)
        {
            stats_.memory_utilization =
                std::min(1.0, static_cast<double>(backend_stats.memory_usage_mb) /
                                  static_cast<double>(cfg_.gpu_memory_mb));
        }
        else
        {
            stats_.memory_utilization = 0.0;
        }
    }

    Config cfg_;
    std::unique_ptr<nerve::persistence::accelerated::HeterogeneousFastVR> backend_;
    PerformanceStats stats_{};
};

} // namespace nerve::gpu
