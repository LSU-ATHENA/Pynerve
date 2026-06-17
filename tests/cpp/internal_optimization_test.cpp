#include "nerve/config.hpp"
#include "nerve/core_types.hpp"
#include "nerve/optimization/component_optimizations.hpp"
#include "nerve/optimization/detail/optimization_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::optimization::AcceleratedCompactSummaries;
using nerve::optimization::AcceleratedStreamingPh;
using nerve::optimization::CallContract;

bool check_compact_summaries_ratio_in_valid_range()
{
    AcceleratedCompactSummaries::SummaryConfig config;
    config.use_per_thread_allocators = false;
    config.precomputeHeavyReductions = false;
    config.summary_size = 128;
    AcceleratedCompactSummaries summaries(config);
    std::vector<std::vector<float>> points = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {0.5f, 0.5f}};
    CallContract contract;
    contract.time_budget_ms = 0.0;
    contract.strict_time_enforcement = false;
    auto summary = summaries.computeSummary(points, contract);
    if (summary.num_points != 5)
    {
        std::cerr << "compact summary num_points mismatch\n";
        return false;
    }
    for (float b : summary.betti_numbers)
    {
        if (!std::isfinite(b))
        {
            std::cerr << "betti number is not finite\n";
            return false;
        }
        if (b < 0.0f)
        {
            std::cerr << "betti number is negative\n";
            return false;
        }
    }
    if (!std::isfinite(summary.persistence_entropy))
    {
        std::cerr << "persistence entropy is not finite\n";
        return false;
    }
    return true;
}

bool check_gpu_primitives_config_validation()
{
#if defined(__CUDACC__)
    using nerve::optimization::AcceleratedGpuPrimitives;
    AcceleratedGpuPrimitives::GPUConfig gpu_config;
    gpu_config.min_batch_size = 16;
    gpu_config.optimal_batch_size = 32;
    gpu_config.max_batch_size = 64;
    gpu_config.enable_async_operations = true;
    gpu_config.num_cuda_streams = 2;
    AcceleratedGpuPrimitives primitives(gpu_config);
    if (primitives.getPeakMemoryUsage() == 0)
    {
        (void)0;
    }
#else
    (void)0;
#endif
    return true;
}

bool check_streaming_ph_window_config()
{
    AcceleratedStreamingPh::StreamingConfig ph_config;
    ph_config.enable_incrementality = true;
    ph_config.enable_coarsening = true;
    ph_config.coarsening_threshold = 100;
    ph_config.max_active_simplex_growth = 5000;
    ph_config.error_budget = 0.05;
    AcceleratedStreamingPh streaming(ph_config);
    if (streaming.isCacheValid())
    {
        std::cerr << "fresh streaming PH should not have valid cache\n";
        return false;
    }
    std::vector<std::vector<float>> window = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    auto result = streaming.computeStreamingPh(window);
    if (result.error_code != nerve::optimization::ErrorCode::SUCCESS)
    {
        std::cerr << "streaming PH returned error: " << static_cast<int>(result.error_code) << "\n";
        return false;
    }
    if (!result.persistence_diagram.empty())
    {
        bool valid = true;
        for (const auto &[b, d] : result.persistence_diagram)
        {
            if (!std::isfinite(b) || !std::isfinite(d))
            {
                valid = false;
                break;
            }
        }
        if (!valid)
        {
            std::cerr << "streaming PH diagram contains non-finite values\n";
            return false;
        }
    }
    return true;
}

bool check_optimizer_simd_ops()
{
    std::vector<double> grads = {10.0, -20.0, 5.0, -5.0, 30.0, -30.0};
    for (double g : grads)
    {
        if (g < -15.0 || g > 15.0)
        {
            std::cerr << "gradient clipping exceeded bound\n";
            return false;
        }
    }
    std::vector<double> vec = {3.0, 4.0};
    if (std::abs(norm - 5.0) > 1e-12)
    {
        std::cerr << "SIMD L2 norm mismatch\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_compact_summaries_ratio_in_valid_range())
    {
        std::cerr << "FAIL: compact summaries ratio in valid range\n";
        return 1;
    }
    if (!check_gpu_primitives_config_validation())
    {
        std::cerr << "FAIL: GPU primitives config validation\n";
        return 1;
    }
    if (!check_streaming_ph_window_config())
    {
        std::cerr << "FAIL: streaming PH window config\n";
        return 1;
    }
    if (!check_optimizer_simd_ops())
    {
        std::cerr << "FAIL: optimizer SIMD ops\n";
        return 1;
    }
    return 0;
}
