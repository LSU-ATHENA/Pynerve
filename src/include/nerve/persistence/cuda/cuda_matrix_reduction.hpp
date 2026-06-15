#pragma once

#include "nerve/errors/errors.hpp"
#include "nerve/persistence/cuda/cuda_safe_arithmetic.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

namespace gpu_kernels
{
constexpr Size MAX_DIM = 10;
constexpr Size BLOCK_SIZE = 256;
constexpr Size STREAMING_THRESHOLD = 1000000;
constexpr Size SHARED_MEMORY_THRESHOLD = 65536;
constexpr Size EARLY_TERMINATION_THRESHOLD = 100000;
} // namespace gpu_kernels

struct MatrixReductionConfig
{
    Size max_dim = 2;
    bool enable_clearing = true;
    bool enable_streaming = false;
    Size streaming_threshold = gpu_kernels::STREAMING_THRESHOLD;
    Size streaming_chunk_size = 10000;
    bool use_memory_manager = true;
    bool enable_early_termination = true;
    bool enable_hybrid_processing = false;
    Size gpu_columns_threshold = 1000;
    double gpu_work_ratio = 0.999;
    bool enable_optimization = false;
    bool enable_apparent_pairs = false;
    bool enable_performance_monitoring = true;
    Size max_edges = 1000000;
    Size max_degree = 1000;
    bool enable_filtering = false;
    bool sort_by_weight = false;
    bool use_shared_memory = true;

    errors::ErrorResult<void> validate() const
    {
        if (max_dim == 0 || max_dim > gpu_kernels::MAX_DIM || streaming_chunk_size == 0 ||
            gpu_columns_threshold == 0 || max_edges == 0 || max_degree == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
        }
        if (!std::isfinite(gpu_work_ratio) || gpu_work_ratio < 0.0 || gpu_work_ratio > 1.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
        }
        return errors::ErrorResult<void>::ok();
    }
};

struct MatrixReductionStats
{
    double total_time_ms = 0.0;
    double gpu_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    Size columns_processed = 0;
    Size pivots_found = 0;
    Size pairs_created = 0;
    Size peak_memory_bytes = 0;
    double reduction_efficiency = 0.0;
    double clearing_efficiency = 0.0;

    double get_reduction_rate() const
    {
        return total_time_ms > 0.0 ? (columns_processed * 1000.0) / total_time_ms : 0.0;
    }
    double get_pivot_rate() const
    {
        return total_time_ms > 0.0 ? (pivots_found * 1000.0) / total_time_ms : 0.0;
    }
    double get_efficiency_score() const
    {
        if (columns_processed == 0 || total_time_ms <= 0.0)
        {
            return 0.0;
        }
        const double expected_time = columns_processed * 0.001;
        return expected_time / total_time_ms;
    }
    double get_memory_efficiency() const
    {
        if (peak_memory_bytes == 0)
        {
            return 1.0;
        }
        const Size theoretical_min = detail::saturatingProduct(columns_processed, Size(16));
        if (peak_memory_bytes <= theoretical_min)
        {
            return 1.0;
        }
        return std::min(1.0, static_cast<double>(theoretical_min) / peak_memory_bytes);
    }
};

struct ApparentPairsConfig
{
    Size max_pairs = 100000;
    bool use_optimization = true;
    bool enable_gpu_acceleration = true;
    bool enable_hybrid_processing = false;
    Size gpu_pairs_threshold = 1000;
    double min_edge_weight_threshold = 0.0;
    bool enable_performance_monitoring = true;

    errors::ErrorResult<void> validate() const
    {
        if (max_pairs == 0 || gpu_pairs_threshold == 0 ||
            !std::isfinite(min_edge_weight_threshold) || min_edge_weight_threshold < 0.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
        }
        return errors::ErrorResult<void>::ok();
    }
};

struct HybridProcessingConfig
{
    Size gpu_columns_threshold = 1000;
    double gpu_work_ratio = 0.999;
    bool enable_adaptive_distribution = true;
    bool enable_performance_monitoring = true;
    bool enable_streaming = true;
    Size streaming_threshold = gpu_kernels::STREAMING_THRESHOLD;
    Size min_problem_size_for_gpu = 1000;
    double max_problem_size_for_cpu = 100000;

    errors::ErrorResult<void> validate() const
    {
        if (gpu_columns_threshold == 0 || streaming_threshold == 0 ||
            min_problem_size_for_gpu == 0 || !std::isfinite(gpu_work_ratio) ||
            gpu_work_ratio < 0.0 || gpu_work_ratio > 1.0 ||
            !std::isfinite(max_problem_size_for_cpu) || max_problem_size_for_cpu < 0.0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
        }
        return errors::ErrorResult<void>::ok();
    }
};

class CUDAMatrixReduction
{
public:
    static errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
    create(const MatrixReductionConfig &config = {});
    ~CUDAMatrixReduction();
    errors::ErrorResult<void> compute_reduction(const int *columns, const Size *column_sizes,
                                                const double *weights, Size n_columns,
                                                Size max_dim);
    errors::ErrorResult<std::vector<int>>
    compute_apparent_pairs(const int *low_row_to_col, const int *col_pivot, const double *weights,
                           Size n_columns, Size max_dim, const ApparentPairsConfig &config = {});
    const MatrixReductionStats &get_performance_stats() const;
    const MatrixReductionConfig &get_config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

errors::ErrorResult<void> computeMatrixReductionGpu(const int *columns, const Size *column_sizes,
                                                    const double *weights, Size n_columns,
                                                    Size max_dim,
                                                    const MatrixReductionConfig &config);
errors::ErrorResult<std::vector<int>>
computeApparentPairsGpu(const int *low_row_to_col, const int *col_pivot, const double *weights,
                        Size n_columns, Size max_dim, const ApparentPairsConfig &config);
void createPersistenceDiagramFromReduction(const int *col_pivot, const double *weights,
                                           Size n_columns, Size max_dim,
                                           std::vector<Pair> &diagram);

namespace utils
{

inline Size estimate_matrix_reduction_memory_usage(Size n_columns, Size max_dim,
                                                   const MatrixReductionConfig &config)
{
    const Size effective_dim = std::min(max_dim, gpu_kernels::MAX_DIM);
    const Size column_slots = detail::saturatingProduct(n_columns, effective_dim);
    Size total = detail::saturatingProduct(column_slots, sizeof(int));
    total = detail::saturatingAdd(total, detail::saturatingProduct(n_columns, sizeof(Size)));
    total = detail::saturatingAdd(total, detail::saturatingProduct(n_columns, sizeof(double)));
    total = detail::saturatingAdd(total, detail::saturatingProduct(column_slots, sizeof(int)));
    total = detail::saturatingAdd(total, detail::saturatingProduct(n_columns, sizeof(int)));
    if (config.enable_streaming)
    {
        total = detail::saturatingAdd(total, sizeof(Size));
    }
    return total;
}

inline double estimate_matrix_reduction_time(Size n_columns, Size max_dim, bool use_gpu = true)
{
    double base_time = std::pow(static_cast<double>(n_columns), 1.5) * max_dim;
    if (use_gpu && n_columns >= 1000)
    {
        const double acceleration =
            std::clamp(std::sqrt(static_cast<double>(n_columns) / 128.0), 1.0, 48.0);
        base_time /= acceleration;
    }
    return base_time;
}

inline Size get_optimal_block_size(Size n_columns, Size max_threads = 1024)
{
    return getOptimalBlockSize(detail::saturatingProduct(n_columns, gpu_kernels::MAX_DIM),
                               max_threads);
}

inline Size get_optimal_grid_size(Size total_elements, Size block_size)
{
    return detail::ceilDiv(total_elements, block_size);
}

inline Size get_optimal_max_edges(Size n_points, double max_radius, double density_factor = 0.1)
{
    if (!std::isfinite(max_radius) || max_radius <= 0.0 || !std::isfinite(density_factor) ||
        density_factor <= 0.0)
    {
        return 0;
    }
    const Size possible_edges = detail::saturatingCompleteGraphEdgeCount(n_points);
    const Size estimated = detail::saturatingScale(possible_edges, std::min(density_factor, 1.0));
    return std::min(estimated, Size(1000000));
}

inline bool should_use_streaming(Size n_columns, Size max_dim, Size available_gpu_memory)
{
    const Size memory_needed =
        estimate_matrix_reduction_memory_usage(n_columns, max_dim, MatrixReductionConfig{});
    return shouldUseStreaming(memory_needed, available_gpu_memory);
}

inline bool should_enable_filtering(Size n_columns, double max_radius, double density_factor = 0.1)
{
    return std::isfinite(max_radius) && max_radius > 0.0 && std::isfinite(density_factor) &&
           (density_factor > 0.3 || n_columns > 10000);
}

inline bool should_enable_sorting(Size max_edges, bool use_gpu = true)
{
    return use_gpu && max_edges > 10000 && max_edges < 1000000;
}

inline bool should_use_shared_memory(Size n_columns, Size max_edges)
{
    const Size total_elements = detail::saturatingProduct(n_columns, gpu_kernels::MAX_DIM);
    return total_elements <= gpu_kernels::SHARED_MEMORY_THRESHOLD && max_edges <= 10000;
}

inline bool should_enable_early_termination(Size max_edges, bool use_gpu = true)
{
    return max_edges > gpu_kernels::EARLY_TERMINATION_THRESHOLD || (use_gpu && max_edges > 10000);
}

inline errors::ErrorResult<void>
validate_matrix_reduction_params(Size n_columns, Size max_dim, const MatrixReductionConfig &config)
{
    if (n_columns == 0 || max_dim == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT);
    }
    if (max_dim > gpu_kernels::MAX_DIM)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    auto config_status = config.validate();
    if (config_status.isError())
    {
        return config_status;
    }
    const Size memory_needed = estimate_matrix_reduction_memory_usage(n_columns, max_dim, config);
    if (memory_needed == std::numeric_limits<Size>::max())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<void>::ok();
}

} // namespace utils

namespace factory
{

inline errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
create_accelerated_matrix_reduction(Size n_columns = 0, Size max_dim = 2, double max_radius = 1.0)
{
    if (!std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>::error(
            errors::ErrorCode::E51_PH_INPUT);
    }
    MatrixReductionConfig config;
    config.max_dim = max_dim;
    config.enable_streaming = n_columns > 50000;
    return CUDAMatrixReduction::create(config);
}

inline errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
create_batch_matrix_reduction(Size batch_size = 1, Size avg_columns = 1000, Size max_dim = 2,
                              double max_radius = 1.0)
{
    if (batch_size == 0 || avg_columns == 0 || !std::isfinite(max_radius) || max_radius <= 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>::error(
            errors::ErrorCode::E51_PH_INPUT);
    }
    MatrixReductionConfig config;
    config.enable_streaming = batch_size > 1;
    config.max_dim = max_dim;
    config.max_edges = detail::saturatingCompleteGraphEdgeCount(avg_columns);
    return CUDAMatrixReduction::create(config);
}

inline errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
create_high_dimensional_matrix_reduction(Size n_columns = 0, Size max_dim = 5,
                                         double max_radius = 1.0)
{
    return create_accelerated_matrix_reduction(n_columns, max_dim, max_radius);
}

inline errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
create_sparse_matrix_reduction(Size n_points, Size point_dim, double max_radius,
                               double density_factor = 0.05)
{
    if (n_points < 2 || point_dim == 0 || !std::isfinite(density_factor) || density_factor <= 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>::error(
            errors::ErrorCode::E51_PH_INPUT);
    }
    const double dimension_scale =
        1.0 / std::sqrt(static_cast<double>(std::max<Size>(1, point_dim)));
    const double effective_density = std::clamp(density_factor * dimension_scale, 1.0e-6, 1.0);
    const Size n_columns = utils::get_optimal_max_edges(n_points, max_radius, effective_density);
    const Size max_dim = std::min<Size>(2, point_dim);
    return create_accelerated_matrix_reduction(n_columns, max_dim, max_radius);
}

inline errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>
create_streaming_matrix_reduction(Size problem_size, Size n_points, Size point_dim,
                                  double max_radius)
{
    if (problem_size == 0 || n_points == 0 || point_dim == 0 || !std::isfinite(max_radius) ||
        max_radius <= 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<CUDAMatrixReduction>>::error(
            errors::ErrorCode::E51_PH_INPUT);
    }
    MatrixReductionConfig config;
    config.max_dim = std::min<Size>(2, point_dim);
    config.enable_streaming = true;
    config.streaming_threshold = detail::saturatingProduct(n_points, point_dim);
    config.streaming_chunk_size =
        std::max<Size>(1, std::min(problem_size, config.streaming_threshold) / point_dim);
    config.max_edges = std::max<Size>(1, utils::get_optimal_max_edges(n_points, max_radius));
    return CUDAMatrixReduction::create(config);
}

} // namespace factory

} // namespace nerve::persistence::accelerated
