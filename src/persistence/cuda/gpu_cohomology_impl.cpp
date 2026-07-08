#include "nerve/algebra/boundary.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/reduction/reduction_clearing_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace detail
{

extern void launchCochainReduction(const int *coboundary_data, const int *coboundary_indices,
                                   const int *coboundary_starts, int *pivots, int *processed,
                                   int n_cochains, int max_height, cudaStream_t stream);

extern void launchDetectBirthSimplices(const int *boundary_data, const int *boundary_indices,
                                       const int *boundary_starts, const int *simplex_dimensions,
                                       bool *out_cleared, int *out_accelerated_count,
                                       int n_simplices, int max_height, cudaStream_t stream);

extern void launchDetectEmergentPairs(const int *boundary_data, const int *boundary_indices,
                                      const int *boundary_starts, bool *out_emergent,
                                      int *out_emergent_count, int n_simplices, int max_height,
                                      cudaStream_t stream);

namespace
{

bool checkedBytes(std::size_t count, std::size_t element_size, std::size_t &bytes)
{
    if (count != 0 && element_size > std::numeric_limits<std::size_t>::max() / count)
    {
        return false;
    }
    bytes = count * element_size;
    return true;
}

bool fitsCudaInt(std::size_t value)
{
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

errors::ErrorResult<void> resourceLimitError()
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
}

} // namespace

class GPUCohomologyEngine
{
public:
    struct CohomologyMatrixGPU
    {
        int *data = nullptr;
        int *indices = nullptr;
        int *starts = nullptr;
        int n_cochains = 0;
        int max_height = 0;
        size_t data_size = 0;
    };

    struct GpuBoundaryMatrix
    {
        int *data = nullptr;
        int *indices = nullptr;
        int *starts = nullptr;
        int *simplex_dimensions = nullptr;
        int n_simplices = 0;
        int max_height = 0;
        size_t data_size = 0;
    };

    static errors::ErrorResult<void> performCohomologyReduction(
        const algebra::BoundaryMatrix &boundary_matrix, std::vector<Index> &out_cocycle_pivots,
        std::vector<bool> &out_processed, std::vector<std::vector<Size>> &out_coboundary_matrix)
    {
        out_cocycle_pivots.clear();
        out_processed.clear();
        out_coboundary_matrix.clear();

        if (boundary_matrix.rows() == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        CohomologyMatrixGPU gpu_matrix;
        auto build_result = buildCoboundaryMatrixGpu(boundary_matrix, gpu_matrix);
        if (build_result.isError())
        {
            cleanup(gpu_matrix);
            return build_result;
        }

        const std::size_t n_cochains = static_cast<std::size_t>(gpu_matrix.n_cochains);
        if (gpu_matrix.data_size == 0 || gpu_matrix.max_height == 0)
        {
            out_cocycle_pivots.assign(n_cochains, static_cast<Index>(-1));
            out_processed.assign(n_cochains, false);
            out_coboundary_matrix.resize(n_cochains);
            cleanup(gpu_matrix);
            return errors::ErrorResult<void>::success();
        }

        std::size_t cochain_bytes = 0;
        if (!checkedBytes(n_cochains, sizeof(int), cochain_bytes))
        {
            cleanup(gpu_matrix);
            return resourceLimitError();
        }

        int *d_pivots = nullptr;
        int *d_processed = nullptr;
        auto cleanup_runtime = [&]() {
            if (d_pivots)
                cudaFree(d_pivots);
            if (d_processed)
                cudaFree(d_processed);
            cleanup(gpu_matrix);
        };

        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_pivots), cochain_bytes);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(reinterpret_cast<void **>(&d_processed), cochain_bytes);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        std::vector<int> hPivots(n_cochains, -1);
        std::vector<int> hProcessed(n_cochains, 0);

        err = cudaMemcpy(d_pivots, hPivots.data(), cochain_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMemcpy(d_processed, hProcessed.data(), cochain_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        launchCochainReduction(gpu_matrix.data, gpu_matrix.indices, gpu_matrix.starts, d_pivots,
                               d_processed, gpu_matrix.n_cochains, gpu_matrix.max_height, nullptr);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMemcpy(hPivots.data(), d_pivots, cochain_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMemcpy(hProcessed.data(), d_processed, cochain_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            cleanup_runtime();
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_cocycle_pivots.resize(n_cochains);
        out_processed.resize(n_cochains);

        for (std::size_t i = 0; i < n_cochains; ++i)
        {
            out_cocycle_pivots[i] = static_cast<Index>(hPivots[i]);
            out_processed[i] = (hProcessed[i] != 0);
        }

        out_coboundary_matrix.resize(n_cochains);

        const size_t n_rows = boundary_matrix.rows();
        const size_t n_cols = boundary_matrix.cols();
        for (size_t col = 0; col < n_cols; ++col)
        {
            for (size_t row = 0; row < n_rows; ++row)
            {
                if (boundary_matrix.getMatrixEntry(row, col) != 0.0 &&
                    row < out_coboundary_matrix.size())
                {
                    out_coboundary_matrix[row].push_back(static_cast<Size>(col));
                }
            }
        }

        cleanup_runtime();
        return errors::ErrorResult<void>::success();
    }

#include "detail/gpu_cohomology_engine_methods.inl"
}
}
}
