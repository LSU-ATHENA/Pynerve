
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/accelerated/nerve_data_wrapper.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

class HeterogeneousFastVR
{
public:
    struct Config
    {
        size_t num_threads = 0; // 0 = auto-detect
        double max_radius = std::numeric_limits<double>::infinity();
        size_t max_dim = 2;
    };

    static errors::ErrorResult<std::unique_ptr<HeterogeneousFastVR>> create()
    {
        return create(Config{});
    }

    static errors::ErrorResult<std::unique_ptr<HeterogeneousFastVR>> create(const Config &config)
    {
        try
        {
            auto engine = std::unique_ptr<HeterogeneousFastVR>(new HeterogeneousFastVR(config));
            return errors::ErrorResult<std::unique_ptr<HeterogeneousFastVR>>::success(
                std::move(engine));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<HeterogeneousFastVR>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(core::BufferView<const double> points, size_t point_dim,
                         const core::DeterminismContract &contract = {})
    {
        if (points.size() == 0 || point_dim == 0 || points.size() % point_dim != 0)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                                 "invalid point buffer");
        }
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!std::isfinite(points[i]))
            {
                return errors::ErrorResult<std::vector<Pair>>::error(
                    errors::ErrorCode::E50_PH_ABORT, "Point coordinates must be finite");
            }
        }

        // Validate input according to determinism level
        if (contract.level == core::DeterminismLevel::STRICT)
        {
            const bool valid = validateDeterministicInput(points, point_dim, contract);
            if (!valid)
            {
                return errors::ErrorResult<std::vector<Pair>>::error(
                    errors::ErrorCode::E30_DET_MISMATCH);
            }
        }

        return computeCpuOnly(points, point_dim, contract);
    }

    errors::ErrorResult<std::vector<Pair>>
    computeVrPersistence(const std::vector<double> &points, size_t point_dim,
                         const core::DeterminismContract &contract = {})
    {
        auto buffer_view = NerveDataWrapper::createBufferView(points);
        return computeVrPersistence(buffer_view, point_dim, contract);
    }

    struct PerformanceStats
    {
        double total_time_ms = 0.0;
        double gpu_time_ms = 0.0;
        double cpu_time_ms = 0.0;
        size_t memory_usage_mb = 0;
        bool gpu_used = false;
    };

    const PerformanceStats &getPerformanceStats() const { return performance_stats_; }

private:
    explicit HeterogeneousFastVR(const Config &config)
        : config_(config)
        , performance_stats_{}
    {}

    errors::ErrorResult<std::vector<Pair>>
    computeCpuOnly(core::BufferView<const double> points, size_t point_dim,
                   const core::DeterminismContract &contract);

    bool validateDeterministicInput(core::BufferView<const double> points, size_t point_dim,
                                    const core::DeterminismContract &contract)
    {
        if (points.size() == 0 || point_dim == 0)
        {
            return false;
        }

        if (points.size() % point_dim != 0)
        {
            return false;
        }

        if (contract.level == core::DeterminismLevel::STRICT)
        {
            for (size_t i = 0; i < points.size(); ++i)
            {
                if (!std::isfinite(points[i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    Config config_;
    PerformanceStats performance_stats_;
};

} // namespace nerve::persistence::accelerated
