
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace nerve::persistence::accelerated
{

class DeterminismContext
{
public:
    struct DeterminismState
    {
        uint64_t seed = 0;
        uint64_t sequence_number = 0;
        bool strict_determinism = false;
        bool gpu_determinism_enabled = false;
    };

    explicit DeterminismContext(const core::DeterminismContract &contract)
        : contract_(contract)
        , state_(resolveSeed(contract), 0, contract.level >= core::DeterminismLevel::STRICT, false)
        , rng_(state_.seed)
    {}

    const DeterminismState &getState() const { return state_; }

    uint64_t nextRandom()
    {
        const uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed);
        state_.sequence_number = sequence + 1;
        return sequence;
    }

    double nextDouble()
    {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return distribution(rng_);
    }

    errors::ErrorResult<void> validateDeterminism() const
    {
        if (state_.seed == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
        }
        if (state_.gpu_determinism_enabled)
        {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess)
            {
                return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
            }
        }
        return errors::ErrorResult<void>::success();
    }

private:
    static uint64_t foldSeedBytes(const std::array<uint8_t, 16> &bytes)
    {
        uint64_t seed = 0xcbf29ce484222325ULL;
        for (const uint8_t value : bytes)
        {
            seed ^= static_cast<uint64_t>(value);
            seed *= 0x100000001b3ULL;
        }
        return seed;
    }

    static uint64_t resolveSeed(const core::DeterminismContract &contract)
    {
        if (contract.rng_seed_provided)
        {
            return foldSeedBytes(contract.rng_seed);
        }
        return 0x9e3779b97f4a7c15ULL;
    }

    core::DeterminismContract contract_;
    mutable DeterminismState state_;
    mutable std::atomic<uint64_t> sequence_ = 0;
    mutable std::mt19937_64 rng_;
};

class DeterministicWorkDistribution
{
public:
    explicit DeterministicWorkDistribution(const DeterminismContext &context)
        : context_(context)
    {}

    WorkDistribution computeDistribution(Size total_columns, Size n_points, Size point_dim,
                                         double max_radius = 1.0) const
    {
        if (total_columns == 0 || !context_.getState().gpu_determinism_enabled)
        {
            WorkDistribution distribution(0, total_columns, false);
            distribution.confidence_score = context_.getState().strict_determinism ? 1.0 : 0.8;
            distribution.strategy = WorkDistributionStrategy::BASIC;
            return distribution;
        }

        const double size_factor = std::min(1.0, static_cast<double>(n_points) / 10000.0);
        const double dim_factor = std::min(1.0, static_cast<double>(point_dim) / 8.0);
        const double radius_factor = std::clamp(max_radius, 0.0, 1.0);
        const double gpu_ratio =
            std::clamp(0.2 + 0.6 * size_factor + 0.2 * dim_factor * radius_factor, 0.0, 0.999);

        WorkDistribution distribution;
        distribution.gpuColumns = static_cast<Size>(gpu_ratio * static_cast<double>(total_columns));
        distribution.cpuColumns = total_columns - distribution.gpuColumns;
        distribution.enableGpu = distribution.gpuColumns > 0;
        distribution.confidence_score = context_.getState().strict_determinism ? 1.0 : 0.8;
        distribution.strategy = WorkDistributionStrategy::ADAPTIVE;
        return distribution;
    }

private:
    const DeterminismContext &context_;
};

class DeterministicGPUOperations
{
public:
    explicit DeterministicGPUOperations(const DeterminismContext &context)
        : context_(context)
    {}

    errors::ErrorResult<void> validateGpuState() const
    {
        if (!context_.getState().gpu_determinism_enabled)
        {
            return errors::ErrorResult<void>::success();
        }
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
        }
        return errors::ErrorResult<void>::success();
    }

private:
    const DeterminismContext &context_;
};

class DeterminismValidator
{
public:
    static errors::ErrorResult<void>
    validateInputDeterminism(const core::BufferView<const double> &points,
                             const core::DeterminismContract &contract)
    {
        if (points.size() == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
        }
        if (contract.level >= core::DeterminismLevel::STRICT)
        {
            for (Size index = 0; index < points.size(); ++index)
            {
                if (!std::isfinite(points[index]))
                {
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
                }
            }
        }
        return errors::ErrorResult<void>::success();
    }

    static errors::ErrorResult<void> validateOutputConsistency(const std::vector<Pair> &diagram1,
                                                               const std::vector<Pair> &diagram2,
                                                               double tolerance)
    {
        if (diagram1.size() != diagram2.size())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
        }
        for (Size index = 0; index < diagram1.size(); ++index)
        {
            const auto &lhs = diagram1[index];
            const auto &rhs = diagram2[index];
            if (lhs.dimension != rhs.dimension)
            {
                return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
            }
            if (std::abs(lhs.birth - rhs.birth) > tolerance ||
                std::abs(lhs.death - rhs.death) > tolerance)
            {
                return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
            }
        }
        return errors::ErrorResult<void>::success();
    }

    static errors::ErrorResult<void> validateGpuDeterminism()
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E30_DET_MISMATCH);
        }
        return errors::ErrorResult<void>::success();
    }
};

namespace determinism
{

std::unique_ptr<DeterminismContext> createContext(const core::DeterminismContract &contract)
{
    return std::make_unique<DeterminismContext>(contract);
}

errors::ErrorResult<void> validateDeterminism(const DeterminismContext &context)
{
    return context.validateDeterminism();
}

std::unique_ptr<DeterministicWorkDistribution>
createWorkDistributor(const DeterminismContext &context)
{
    return std::make_unique<DeterministicWorkDistribution>(context);
}

std::unique_ptr<DeterministicGPUOperations> createGpuOperations(const DeterminismContext &context)
{
    return std::make_unique<DeterministicGPUOperations>(context);
}

errors::ErrorResult<void> validateInputDeterminism(const core::BufferView<const double> &points,
                                                   const core::DeterminismContract &contract)
{
    return DeterminismValidator::validateInputDeterminism(points, contract);
}

errors::ErrorResult<void> validateOutputConsistency(const std::vector<Pair> &diagram1,
                                                    const std::vector<Pair> &diagram2,
                                                    double tolerance)
{
    return DeterminismValidator::validateOutputConsistency(diagram1, diagram2, tolerance);
}

errors::ErrorResult<void> validateGpuDeterminism()
{
    return DeterminismValidator::validateGpuDeterminism();
}

template <typename Func>
errors::ErrorResult<std::invoke_result_t<Func>>
deterministicComputation(const core::DeterminismContract &contract, Func &&computation)
{
    using ResultType = std::invoke_result_t<Func>;
    auto context = createContext(contract);
    auto pre_check = validateDeterminism(*context);
    if (pre_check.isError())
    {
        return errors::ErrorResult<ResultType>::error(pre_check.errorCode());
    }
    try
    {
        return errors::ErrorResult<ResultType>::success(computation());
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return errors::ErrorResult<ResultType>::error(errors::ErrorCode::E50_PH_ABORT);
    }
}

} // namespace determinism

} // namespace nerve::persistence::accelerated
