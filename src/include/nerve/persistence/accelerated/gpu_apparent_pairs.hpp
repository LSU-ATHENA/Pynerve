
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <vector>

namespace nerve::persistence::accelerated
{

// CUDA kernel declaration (must be free function, not class member)
__global__ void detectApparentPairsKernel(const int *vertex_data, const int *vertex_counts,
                                          const int *vertex_offsets,
                                          const double *filtration_values, int *apparentPairs,
                                          int n_simplices, int determinism_level);

struct SimplexGPU
{
    std::vector<int> vertices;
    double filtrationValue;
    int dimension;

    SimplexGPU(std::vector<int> verts, double filt_val, int dim)
        : vertices(std::move(verts))
        , filtrationValue(filt_val)
        , dimension(dim)
    {}
};

class GPUApparentPairs
{
public:
    struct Config
    {
        size_t max_simplices = 1000000;
        size_t max_dimension = 10;
        // Default bias keeps most candidate checks on GPU while allowing
        // conservative CPU spillover when needed.
        double gpu_work_ratio = 0.999;
        size_t threads_per_block = 256;
        size_t blocks_per_grid = 0; // 0 = auto-calculate

        errors::ErrorResult<void> validate() const
        {
            if (max_simplices == 0 || max_dimension == 0 || threads_per_block == 0 ||
                !std::isfinite(gpu_work_ratio) || gpu_work_ratio < 0.0 || gpu_work_ratio > 1.0)
            {
                return errors::ErrorResult<void>::error(errors::ErrorCode::E52_PH_CONFIG);
            }
            return errors::ErrorResult<void>::ok();
        }
    };

    static errors::ErrorResult<std::unique_ptr<GPUApparentPairs>> create()
    {
        return create(Config{});
    }

    static errors::ErrorResult<std::unique_ptr<GPUApparentPairs>> create(const Config &config)
    {
        auto config_status = config.validate();
        if (config_status.isError())
        {
            return errors::ErrorResult<std::unique_ptr<GPUApparentPairs>>::error(
                config_status.errorCode(), config_status.error().message);
        }
        try
        {
            auto gpu_apparent = std::unique_ptr<GPUApparentPairs>(new GPUApparentPairs(config));
            auto init_result = gpu_apparent->initialize();
            if (init_result.isError())
            {
                return errors::ErrorResult<std::unique_ptr<GPUApparentPairs>>::error(
                    init_result.errorCode());
            }
            return errors::ErrorResult<std::unique_ptr<GPUApparentPairs>>::success(
                std::move(gpu_apparent));
        }
        catch (const std::exception &e)
        {
            return errors::ErrorResult<std::unique_ptr<GPUApparentPairs>>::error(
                errors::ErrorCode::E50_PH_ABORT, e.what());
        }
    }

    ~GPUApparentPairs() { cleanup(); }

    errors::ErrorResult<std::vector<int>>
    detectApparentPairs(const std::vector<SimplexGPU> &filtration,
                        const core::DeterminismContract &contract = {})
    {
        if (filtration.size() > config_.max_simplices)
        {
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Number of simplices exceeds GPU limit");
        }
        for (const auto &simplex : filtration)
        {
            if (simplex.dimension < 0 ||
                static_cast<size_t>(simplex.dimension) > config_.max_dimension ||
                !std::isfinite(simplex.filtrationValue))
            {
                return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E51_PH_INPUT,
                                                                    "Invalid simplex metadata");
            }
            for (int vertex : simplex.vertices)
            {
                if (vertex < 0)
                {
                    return errors::ErrorResult<std::vector<int>>::error(
                        errors::ErrorCode::E51_PH_INPUT, "Invalid simplex vertex");
                }
            }
        }

#if !defined(__CUDACC__)
        (void)contract;
        return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                            "CUDA kernels require a CUDA build");
#else
        std::vector<int> apparentPairs(filtration.size(), -1);

        // Allocate GPU memory
        size_t filtration_size = filtration.size();

        // Convert filtration to GPU-compatible format
        std::vector<int> vertex_data;
        std::vector<double> filtration_values;
        std::vector<int> vertex_counts;
        std::vector<int> vertex_offsets;

        size_t total_vertices = 0;
        for (const auto &simplex : filtration)
        {
            vertex_counts.push_back(simplex.vertices.size());
            vertex_offsets.push_back(total_vertices);
            filtration_values.push_back(simplex.filtrationValue);
            total_vertices += simplex.vertices.size();
        }

        vertex_data.resize(total_vertices);
        size_t offset = 0;
        for (const auto &simplex : filtration)
        {
            for (int vertex : simplex.vertices)
            {
                vertex_data[offset++] = vertex;
            }
        }

        // Allocate GPU memory
        int *d_vertex_data = nullptr;
        int *d_vertex_counts = nullptr;
        int *d_vertex_offsets = nullptr;
        double *d_filtration_values = nullptr;
        int *d_apparent_pairs = nullptr;

        size_t vertex_data_size = total_vertices * sizeof(int);
        size_t index_buffer_size = filtration_size * sizeof(int);
        size_t vertex_counts_size = index_buffer_size;
        size_t vertex_offsets_size = index_buffer_size;
        size_t filtration_values_size = filtration_size * sizeof(double);
        size_t apparent_pairs_size = filtration_size * sizeof(int);

        cudaError_t err = cudaMalloc(&d_vertex_data, vertex_data_size);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to allocate GPU memory for vertex data");
        }

        err = cudaMalloc(&d_vertex_counts, vertex_counts_size);
        if (err != cudaSuccess)
        {
            cudaFree(d_vertex_data);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to allocate GPU memory for vertex counts");
        }

        err = cudaMalloc(&d_vertex_offsets, vertex_offsets_size);
        if (err != cudaSuccess)
        {
            cudaFree(d_vertex_data);
            cudaFree(d_vertex_counts);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT,
                "Failed to allocate GPU memory for vertex offsets");
        }

        err = cudaMalloc(&d_filtration_values, filtration_values_size);
        if (err != cudaSuccess)
        {
            cudaFree(d_vertex_data);
            cudaFree(d_vertex_counts);
            cudaFree(d_vertex_offsets);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT,
                "Failed to allocate GPU memory for filtration values");
        }

        err = cudaMalloc(&d_apparent_pairs, apparent_pairs_size);
        if (err != cudaSuccess)
        {
            cudaFree(d_vertex_data);
            cudaFree(d_vertex_counts);
            cudaFree(d_vertex_offsets);
            cudaFree(d_filtration_values);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT,
                "Failed to allocate GPU memory for apparent pairs");
        }

        // Copy data to GPU
        err =
            cudaMemcpy(d_vertex_data, vertex_data.data(), vertex_data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to copy vertex data to GPU");
        }

        err = cudaMemcpy(d_vertex_counts, vertex_counts.data(), vertex_counts_size,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to copy vertex counts to GPU");
        }

        err = cudaMemcpy(d_vertex_offsets, vertex_offsets.data(), vertex_offsets_size,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to copy vertex offsets to GPU");
        }

        err = cudaMemcpy(d_filtration_values, filtration_values.data(), filtration_values_size,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to copy filtration values to GPU");
        }

        // Initialize apparent pairs to -1
        err = cudaMemset(d_apparent_pairs, -1, apparent_pairs_size);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to initialize apparent pairs on GPU");
        }

        // Launch kernel (CUDA-only path).
        size_t blocks_per_grid = config_.blocks_per_grid;
        if (blocks_per_grid == 0)
        {
            blocks_per_grid =
                (filtration_size + config_.threads_per_block - 1) / config_.threads_per_block;
        }

        detectApparentPairsKernel<<<blocks_per_grid, config_.threads_per_block>>>(
            d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values, d_apparent_pairs,
            filtration_size, static_cast<int>(contract.level));

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(errors::ErrorCode::E50_PH_ABORT,
                                                                "GPU kernel launch failed");
        }

        // Copy results back
        err = cudaMemcpy(apparentPairs.data(), d_apparent_pairs, apparent_pairs_size,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                             d_apparent_pairs);
            return errors::ErrorResult<std::vector<int>>::error(
                errors::ErrorCode::E50_PH_ABORT, "Failed to copy apparent pairs from GPU");
        }

        // Cleanup
        cleanupGpuMemory(d_vertex_data, d_vertex_counts, d_vertex_offsets, d_filtration_values,
                         d_apparent_pairs);

        return errors::ErrorResult<std::vector<int>>::success(std::move(apparentPairs));
#endif
    }

private:
    explicit GPUApparentPairs(const Config &config)
        : config_(config)
    {}

    errors::ErrorResult<void> initialize()
    {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E50_PH_ABORT,
                                                    "No CUDA devices available");
        }

        return errors::ErrorResult<void>::ok();
    }

    void cleanup()
    {
        // Cleanup any allocated resources
    }

    void cleanupGpuMemory(int *d_vertex_data, int *d_vertex_counts, int *d_vertex_offsets,
                          double *d_filtration_values, int *d_apparent_pairs)
    {
        if (d_vertex_data)
            cudaFree(d_vertex_data);
        if (d_vertex_counts)
            cudaFree(d_vertex_counts);
        if (d_vertex_offsets)
            cudaFree(d_vertex_offsets);
        if (d_filtration_values)
            cudaFree(d_filtration_values);
        if (d_apparent_pairs)
            cudaFree(d_apparent_pairs);
    }

    Config config_;
};

} // namespace nerve::persistence::accelerated
