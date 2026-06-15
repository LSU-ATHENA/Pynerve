
#pragma once

#include "nerve/errors/errors.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

enum class WorkDistributionStrategy
{
    BASIC,
    ADAPTIVE,
    ACCELERATED
};

struct WorkDistribution
{
    size_t gpuColumns = 0;
    size_t cpuColumns = 0;
    bool enableGpu = false;
    double confidence_score = 1.0;
    WorkDistributionStrategy strategy = WorkDistributionStrategy::BASIC;

    WorkDistribution() = default;
    WorkDistribution(size_t gpu, size_t cpu, bool enableGpu)
        : gpuColumns(gpu)
        , cpuColumns(cpu)
        , enableGpu(enableGpu)
    {}
};

class WorkDistributor
{
public:
    struct Config
    {
        double gpu_work_ratio = 0.0;
        size_t min_gpu_workload = 100;
        size_t max_cpu_workload = 1000;
        bool force_cpu = false;
        bool force_gpu = false;
    };

    static errors::ErrorResult<std::unique_ptr<WorkDistributor>> create()
    {
        return create(Config{});
    }

    static errors::ErrorResult<std::unique_ptr<WorkDistributor>> create(const Config &config)
    {
        auto valid = validateConfig(config);
        if (valid.isError())
        {
            return errors::ErrorResult<std::unique_ptr<WorkDistributor>>::error(
                valid.errorCode(), valid.error().message);
        }
        try
        {
            auto distributor = std::unique_ptr<WorkDistributor>(new WorkDistributor(config));
            return errors::ErrorResult<std::unique_ptr<WorkDistributor>>::success(
                std::move(distributor));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<WorkDistributor>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    WorkDistribution computeDistribution(size_t total_columns) const
    {
        if (config_.force_cpu)
        {
            return WorkDistribution(0, total_columns, false);
        }

        if (config_.force_gpu)
        {
            return WorkDistribution(total_columns, 0, true);
        }

        if (total_columns == 0 || total_columns < config_.min_gpu_workload)
        {
            return WorkDistribution(0, total_columns, false);
        }
        if (config_.gpu_work_ratio <= 0.0)
        {
            return WorkDistribution(0, total_columns, false);
        }

        return splitWork(total_columns, config_.gpu_work_ratio, WorkDistributionStrategy::BASIC);
    }

    WorkDistribution computeAdaptiveDistribution(size_t total_columns, size_t n_points,
                                                 size_t point_dimension, double max_radius) const
    {
        if (config_.force_cpu || config_.force_gpu)
        {
            return computeDistribution(total_columns);
        }
        if (total_columns == 0 || n_points < 100 || point_dimension == 0 ||
            !std::isfinite(max_radius) || max_radius <= 0.0)
        {
            return WorkDistribution(0, total_columns, false);
        }

        const double dimension = static_cast<double>(std::clamp<size_t>(point_dimension, 1, 6));
        const double radius = std::clamp(max_radius, 0.0, 1.0);
        const double density = std::pow(radius, std::min(dimension, 4.0));
        const double estimated_work =
            static_cast<double>(n_points) * static_cast<double>(n_points) * std::max(density, 0.01);
        const double threshold =
            static_cast<double>(std::max(config_.min_gpu_workload, config_.max_cpu_workload));

        if (total_columns < config_.min_gpu_workload || estimated_work < threshold)
        {
            return WorkDistribution(0, total_columns, false);
        }
        if (config_.gpu_work_ratio <= 0.0)
        {
            return WorkDistribution(0, total_columns, false);
        }

        auto distribution =
            splitWork(total_columns, config_.gpu_work_ratio, WorkDistributionStrategy::ADAPTIVE);
        distribution.confidence_score = std::clamp(estimated_work / (threshold * 10.0), 0.0, 1.0);
        return distribution;
    }

    errors::ErrorResult<void> updateConfig(const Config &config)
    {
        auto valid = validateConfig(config);
        if (valid.isError())
        {
            return valid;
        }
        config_ = config;
        return errors::ErrorResult<void>::success();
    }

    const Config &getConfig() const { return config_; }

private:
    explicit WorkDistributor(const Config &config)
        : config_(config)
    {}

    static errors::ErrorResult<void> validateConfig(const Config &config)
    {
        if (!std::isfinite(config.gpu_work_ratio) || config.gpu_work_ratio < 0.0 ||
            config.gpu_work_ratio > 1.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                    "gpu_work_ratio must be in [0, 1]");
        }
        if (config.force_cpu && config.force_gpu)
        {
            return errors::ErrorResult<void>::error(
                errors::ErrorCode::E50_PH_ABORT, "force_cpu and force_gpu are mutually exclusive");
        }
        return errors::ErrorResult<void>::success();
    }

    static WorkDistribution splitWork(size_t total_columns, double gpu_ratio,
                                      WorkDistributionStrategy strategy)
    {
        const size_t gpu_columns =
            static_cast<size_t>(static_cast<double>(total_columns) * gpu_ratio);
        WorkDistribution distribution(gpu_columns, total_columns - gpu_columns, gpu_columns > 0);
        distribution.strategy = strategy;
        return distribution;
    }

    Config config_;
};

} // namespace nerve::persistence::accelerated
