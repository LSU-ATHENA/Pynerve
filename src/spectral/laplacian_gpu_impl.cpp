
#include "nerve/errors/errors.hpp"
#include "nerve/gpu/compute_manager.hpp"
#include "nerve/spectral/laplacian.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

namespace nerve::gpu::kernels
{
extern void launchUpLaplacianKernel(const double *d_boundary_matrix,
                                    const int *d_simplex_dimensions, double *d_up_laplacian,
                                    size_t size, int d, size_t n_d, const size_t *d_indices,
                                    cudaStream_t stream);

extern void launchDownLaplacianKernel(const double *d_boundary_matrix,
                                      const int *d_simplex_dimensions, double *d_down_laplacian,
                                      size_t size, int d, size_t n_d, const size_t *d_indices,
                                      cudaStream_t stream);

extern void launchAddMatricesKernel(const double *d_matrix_a, const double *d_matrix_b,
                                    double *d_result, size_t n, cudaStream_t stream);

extern void launchUpLaplacianTiledKernel(const double *d_boundary_matrix, const size_t *d_indices,
                                         const size_t *d1_indices, size_t n_d, size_t n_d1,
                                         double *d_up_laplacian, size_t size, cudaStream_t stream);
} // namespace nerve::gpu::kernels

namespace nerve::spectral::detail
{

namespace
{

bool checkedProduct(size_t lhs, size_t rhs, size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedSquareCount(size_t count, size_t &out) noexcept
{
    return checkedProduct(count, count, out);
}

bool checkedByteCount(size_t count, size_t element_size, size_t &out) noexcept
{
    return checkedProduct(count, element_size, out);
}

errors::ErrorResult<void> resourceLimit(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT, message);
}

errors::ErrorResult<void> invalidSimplices(std::string_view message)
{
    return errors::ErrorResult<void>::error(errors::ErrorCode::E88_INVALID_SIMPLICES, message);
}

} // namespace

class GPULaplacianCompute
{
public:
    static errors::ErrorResult<void>
    computeAllLaplaciansGpu(const std::vector<std::vector<double>> &boundary_matrix,
                            const std::vector<int> &simplex_dimensions, int max_dimension,
                            std::vector<std::vector<std::vector<double>>> &out_laplacians,
                            std::vector<std::vector<std::vector<double>>> &out_up_laplacians,
                            std::vector<std::vector<std::vector<double>>> &out_down_laplacians,
                            std::vector<std::vector<std::vector<double>>> &out_hodge_laplacians)
    {
        size_t size = boundary_matrix.size();

        if (size == 0)
        {
            return errors::ErrorResult<void>::success();
        }

        if (simplex_dimensions.size() != size)
        {
            return invalidSimplices("GPU Laplacian simplex dimension count does not match matrix");
        }

        if (max_dimension < 0)
        {
            return invalidSimplices("GPU Laplacian max dimension must be nonnegative");
        }

        if (max_dimension > std::numeric_limits<int>::max() - 2)
        {
            return resourceLimit("GPU Laplacian dimension count exceeds supported range");
        }

        const size_t dimension_count = static_cast<size_t>(max_dimension) + 2;
        const size_t output_dimension_count = static_cast<size_t>(max_dimension) + 1;
        if (dimension_count > std::vector<std::vector<size_t>>().max_size() ||
            output_dimension_count > std::vector<std::vector<std::vector<double>>>().max_size())
        {
            return resourceLimit("GPU Laplacian dimension count exceeds host limits");
        }

        size_t matrix_entries = 0;
        size_t matrix_bytes = 0;
        size_t dimension_bytes = 0;
        size_t index_bytes = 0;
        if (!checkedSquareCount(size, matrix_entries) ||
            !checkedByteCount(matrix_entries, sizeof(double), matrix_bytes) ||
            !checkedByteCount(size, sizeof(int), dimension_bytes) ||
            !checkedByteCount(size, sizeof(size_t), index_bytes))
        {
            return resourceLimit("GPU Laplacian dense matrix allocation exceeds host limits");
        }

        std::vector<std::vector<size_t>> indicesByDim(dimension_count);
        const int adjacent_dimension_limit = max_dimension + 1;
        for (size_t i = 0; i < simplex_dimensions.size(); ++i)
        {
            int dim = simplex_dimensions[i];
            if (dim >= 0 && dim <= adjacent_dimension_limit)
            {
                indicesByDim[dim].push_back(i);
            }
        }

        std::vector<double> flatBoundary(matrix_entries);
        for (size_t i = 0; i < size; ++i)
        {
            if (boundary_matrix[i].size() != size)
            {
                return invalidSimplices("GPU Laplacian boundary matrix must be square");
            }
            for (size_t j = 0; j < size; ++j)
            {
                flatBoundary[i * size + j] = boundary_matrix[i][j];
            }
        }

        double *d_boundary = nullptr;
        int *d_dimensions = nullptr;
        double *d_result = nullptr;
        double *d_temp = nullptr;
        size_t *d_indices = nullptr;
        size_t *d_indices_d1 = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_boundary, matrix_bytes);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_dimensions, dimension_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_boundary);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_indices, index_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_boundary);
            cudaFree(d_dimensions);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMalloc(&d_indices_d1, index_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_boundary);
            cudaFree(d_dimensions);
            cudaFree(d_indices);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

        err = cudaMemcpy(d_boundary, flatBoundary.data(), matrix_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        err = cudaMemcpy(d_dimensions, simplex_dimensions.data(), dimension_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        out_laplacians.assign(output_dimension_count, {});
        out_up_laplacians.assign(output_dimension_count, {});
        out_down_laplacians.assign(output_dimension_count, {});
        out_hodge_laplacians.assign(output_dimension_count, {});

        for (int dim = 0; dim <= max_dimension; ++dim)
        {
            const auto &d_indices_vec = indicesByDim[dim];
            size_t n_d = d_indices_vec.size();

            if (n_d == 0)
            {
                continue;
            }

            size_t laplacian_entries = 0;
            size_t laplacian_bytes = 0;
            size_t n_d_index_bytes = 0;
            if (!checkedSquareCount(n_d, laplacian_entries) ||
                !checkedByteCount(laplacian_entries, sizeof(double), laplacian_bytes) ||
                !checkedByteCount(n_d, sizeof(size_t), n_d_index_bytes))
            {
                cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                return resourceLimit(
                    "GPU Laplacian dimension matrix allocation exceeds host limits");
            }

            std::vector<double> upLaplacian(laplacian_entries, 0.0);
            std::vector<double> downLaplacian(laplacian_entries, 0.0);

            err = cudaMalloc(&d_result, laplacian_bytes);
            if (err != cudaSuccess)
            {
                cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
            }

            err = cudaMalloc(&d_temp, laplacian_bytes);
            if (err != cudaSuccess)
            {
                cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
            }

            err = cudaMemcpy(d_indices, d_indices_vec.data(), n_d_index_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
            }

            const auto &d1_indices_vec = indicesByDim[dim + 1];
            size_t n_d1 = d1_indices_vec.size();

            if (n_d1 > 0)
            {
                size_t n_d1_index_bytes = 0;
                if (!checkedByteCount(n_d1, sizeof(size_t), n_d1_index_bytes))
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return resourceLimit(
                        "GPU Laplacian adjacent dimension index allocation exceeds host limits");
                }

                err = cudaMemcpy(d_indices_d1, d1_indices_vec.data(), n_d1_index_bytes,
                                 cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }

                ::nerve::gpu::kernels::launchUpLaplacianTiledKernel(
                    d_boundary, d_indices, d_indices_d1, n_d, n_d1, d_result, size, nullptr);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }

                err = cudaMemcpy(upLaplacian.data(), d_result, laplacian_bytes,
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }
            }

            if (dim > 0)
            {
                ::nerve::gpu::kernels::launchDownLaplacianKernel(
                    d_boundary, d_dimensions, d_temp, size, dim, n_d, d_indices, nullptr);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }

                err = cudaMemcpy(downLaplacian.data(), d_temp, laplacian_bytes,
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }
            }

            if (n_d1 > 0 && dim > 0)
            {
                err = cudaMemcpy(d_result, upLaplacian.data(), laplacian_bytes,
                                 cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }

                ::nerve::gpu::kernels::launchAddMatricesKernel(d_result, d_temp, d_result, n_d,
                                                               nullptr);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
                    return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
                }
            }

            std::vector<std::vector<double>> up2d(n_d, std::vector<double>(n_d));
            std::vector<std::vector<double>> down2d(n_d, std::vector<double>(n_d));
            std::vector<std::vector<double>> hodge2d(n_d, std::vector<double>(n_d));

            for (size_t i = 0; i < n_d; ++i)
            {
                for (size_t j = 0; j < n_d; ++j)
                {
                    up2d[i][j] = upLaplacian[i * n_d + j];
                    down2d[i][j] = downLaplacian[i * n_d + j];
                    hodge2d[i][j] = up2d[i][j] + down2d[i][j];
                }
            }

            out_up_laplacians[dim] = std::move(up2d);
            out_down_laplacians[dim] = std::move(down2d);
            out_hodge_laplacians[dim] = std::move(hodge2d);
            out_laplacians[dim] = out_hodge_laplacians[dim];

            cudaFree(d_result);
            cudaFree(d_temp);
            d_result = nullptr;
            d_temp = nullptr;
        }

        cleanup(d_boundary, d_dimensions, d_result, d_temp, d_indices, d_indices_d1);
        return errors::ErrorResult<void>::success();
    }

private:
    static void cleanup(double *d_boundary, int *d_dimensions, double *d_result, double *d_temp,
                        size_t *d_indices, size_t *d_indices_d1)
    {
        if (d_boundary)
            cudaFree(d_boundary);
        if (d_dimensions)
            cudaFree(d_dimensions);
        if (d_result)
            cudaFree(d_result);
        if (d_temp)
            cudaFree(d_temp);
        if (d_indices)
            cudaFree(d_indices);
        if (d_indices_d1)
            cudaFree(d_indices_d1);
    }
};

errors::ErrorResult<void>
computeAllLaplaciansGpu(const std::vector<std::vector<double>> &boundary_matrix,
                        const std::vector<int> &simplex_dimensions, int max_dimension,
                        std::vector<std::vector<std::vector<double>>> &out_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_up_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_down_laplacians,
                        std::vector<std::vector<std::vector<double>>> &out_hodge_laplacians)
{
    return GPULaplacianCompute::computeAllLaplaciansGpu(
        boundary_matrix, simplex_dimensions, max_dimension, out_laplacians, out_up_laplacians,
        out_down_laplacians, out_hodge_laplacians);
}

} // namespace nerve::spectral::detail
