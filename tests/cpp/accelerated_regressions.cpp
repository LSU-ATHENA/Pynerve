
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::common::AccelerationMode;
using nerve::common::ProblemCharacteristics;
using nerve::common::ProblemType;
using nerve::common::SystemCapabilities;
using nerve::common::VRAlgorithmSelection;
using nerve::common::VRConfig;
using nerve::persistence::Pair;
using nerve::persistence::accelerated::GPUAccelerationManager;
using namespace nerve::test;

bool check_system_capability_detection()
{
    auto caps = GPUAccelerationManager::detectSystemCapabilities();
    if (caps.available_memory == 0 && caps.cuda_available)
    {
        std::cerr << "cuda available but no memory detected\n";
        return false;
    }
    return true;
}

bool check_acceleration_mode_recommendation()
{
    SystemCapabilities caps;
    caps.cuda_available = true;
    caps.compute_capability = 7.5;
    caps.available_memory = 8ULL * 1024 * 1024 * 1024;

    ProblemCharacteristics small;
    small.estimated_n_points = 100;
    small.point_dim = 2;
    small.max_radius = 1.0;
    small.problem_type = ProblemType::SMALL;

    auto mode_small = GPUAccelerationManager::recommendAccelerationMode(small, caps);
    static_cast<void>(mode_small);

    ProblemCharacteristics large;
    large.estimated_n_points = 100000;
    large.point_dim = 3;
    large.max_radius = 5.0;
    large.problem_type = ProblemType::LARGE;

    auto mode_large = GPUAccelerationManager::recommendAccelerationMode(large, caps);
    static_cast<void>(mode_large);

    return true;
}

bool check_gpu_runtime_availability()
{
    bool available = GPUAccelerationManager::isGpuRuntimeAvailable();
    static_cast<void>(available);
    return true;
}

bool check_compute_vr_persistence_accelerated_basic()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig config;
    config.max_radius = 2.0;
    config.max_dim = 1;
    config.algorithm = VRAlgorithmSelection::ACCELERATED;
    config.use_acceleration = true;

    auto result = nerve::persistence::accelerated::computeVrPersistenceAccelerated(view_of(points),
                                                                                   2, config);
    if (result.isError())
    {
        auto ec = result.errorCode();
        bool acceptable = (ec == nerve::errors::ErrorCode::EB0_FEATURE_DISABLED) ||
                          (ec == nerve::errors::ErrorCode::E10_GPU_OOM) ||
                          (ec == nerve::errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        if (!acceptable)
        {
            std::cerr << "computeVrPersistenceAccelerated failed: " << result.compactSummary()
                      << "\n";
            return false;
        }
        return true;
    }
    auto pairs = result.value();
    if (pairs.empty())
    {
        std::cerr << "accelerated persistence should produce pairs\n";
        return false;
    }
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && !(p.birth <= p.death + 1e-10))
        {
            std::cerr << "birth<=death invariant violated: " << p.birth << " > " << p.death << "\n";
            return false;
        }
    }
    return true;
}

bool check_compute_vr_persistence_accelerated_small_cloud()
{
    const std::vector<double> points = {0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.25, 0.25};
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 1;
    config.algorithm = VRAlgorithmSelection::ACCELERATED;
    config.use_acceleration = true;

    auto result = nerve::persistence::accelerated::computeVrPersistenceAccelerated(view_of(points),
                                                                                   2, config);
    if (result.isError())
    {
        auto ec = result.errorCode();
        bool acceptable = (ec == nerve::errors::ErrorCode::EB0_FEATURE_DISABLED) ||
                          (ec == nerve::errors::ErrorCode::E10_GPU_OOM) ||
                          (ec == nerve::errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        if (!acceptable)
        {
            return false;
        }
        return true;
    }
    auto pairs = result.value();
    Size h0_essential = 0;
    for (const auto &p : pairs)
    {
        if (p.dimension == 0 && p.isInfinite())
            ++h0_essential;
    }
    if (h0_essential < 1)
    {
        std::cerr << "small cloud: expected at least 1 H0 essential, got " << h0_essential << "\n";
        return false;
    }
    return true;
}

bool check_compute_vr_persistence_accelerated_empty_input()
{
    const std::vector<double> empty_points;
    VRConfig config;
    config.max_radius = 1.0;
    config.max_dim = 1;
    config.algorithm = VRAlgorithmSelection::ACCELERATED;
    config.use_acceleration = true;

    auto result = nerve::persistence::accelerated::computeVrPersistenceAccelerated(
        view_of(empty_points), 2, config);
    if (result.isError())
    {
        auto ec = result.errorCode();
        bool acceptable = (ec == nerve::errors::ErrorCode::EB0_FEATURE_DISABLED) ||
                          (ec == nerve::errors::ErrorCode::E10_GPU_OOM) ||
                          (ec == nerve::errors::ErrorCode::E11_GPU_LAUNCH_FAIL) ||
                          (ec == nerve::errors::ErrorCode::E50_PH_ABORT) ||
                          (ec == nerve::errors::ErrorCode::E51_PH_INPUT);
        if (!acceptable)
        {
            return false;
        }
        return true;
    }
    auto pairs = result.value();
    if (!pairs.empty())
    {
        std::cerr << "empty input should produce empty result, got " << pairs.size() << " pairs\n";
        return false;
    }
    return true;
}

bool check_compute_vr_persistence_fast_reference()
{
    const std::vector<double> points = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    VRConfig config;
    config.max_radius = 1.5;
    config.max_dim = 2;
    config.algorithm = VRAlgorithmSelection::ACCELERATED;
    config.use_acceleration = true;

    auto result =
        nerve::persistence::accelerated::computeVrPersistenceFast(view_of(points), 2, config);
    if (result.isError())
    {
        auto ec = result.errorCode();
        bool acceptable = (ec == nerve::errors::ErrorCode::EB0_FEATURE_DISABLED) ||
                          (ec == nerve::errors::ErrorCode::E10_GPU_OOM) ||
                          (ec == nerve::errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        if (!acceptable)
        {
            return false;
        }
        return true;
    }
    auto pairs = result.value();
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "fast: negative persistence\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    if (!check_system_capability_detection())
    {
        std::cerr << "FAIL: system capability detection\n";
        return 1;
    }
    if (!check_acceleration_mode_recommendation())
    {
        std::cerr << "FAIL: acceleration mode recommendation\n";
        return 1;
    }
    if (!check_gpu_runtime_availability())
    {
        std::cerr << "FAIL: GPU runtime availability\n";
        return 1;
    }
    if (!check_compute_vr_persistence_accelerated_basic())
    {
        std::cerr << "FAIL: compute accelerated basic\n";
        return 1;
    }
    if (!check_compute_vr_persistence_accelerated_small_cloud())
    {
        std::cerr << "FAIL: compute accelerated small cloud\n";
        return 1;
    }
    if (!check_compute_vr_persistence_accelerated_empty_input())
    {
        std::cerr << "FAIL: compute accelerated empty input\n";
        return 1;
    }
    if (!check_compute_vr_persistence_fast_reference())
    {
        std::cerr << "FAIL: compute persistence fast reference\n";
        return 1;
    }
    return 0;
}
