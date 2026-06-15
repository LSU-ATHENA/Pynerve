
// CUDA kernels for Dirac-operator assembly and Laplacian projection.
//
// This implementation currently uses:
// - dense CUDA kernels for Dirac assembly and D^2 projection
// - orientation-weighted diagonal Hodge-star approximation
// - host-side Clifford multivector algebra helper routines

#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace nerve
{
namespace spectral
{
namespace gpu
{

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_INT_SQUARE_DIMENSION = 46340;

namespace
{

int checkedAddInt(int lhs, int rhs, const char *context)
{
    if (rhs < 0)
    {
        throw std::invalid_argument(context);
    }
    if (lhs > std::numeric_limits<int>::max() - rhs)
    {
        throw std::length_error(context);
    }
    return lhs + rhs;
}

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

std::size_t checkedByteCount(std::size_t count, std::size_t element_size, const char *context)
{
    return checkedProduct(count, element_size, context);
}

std::size_t checkedSquareElements(int n, const char *context)
{
    if (n < 0)
    {
        throw std::invalid_argument(context);
    }
    const auto size = static_cast<std::size_t>(n);
    return checkedProduct(size, size, context);
}

int checkedIntSize(std::size_t size, const char *context)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(size);
}

int checkedGridBlocks(std::size_t elements, const char *context)
{
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    const std::size_t blocks = (elements + static_cast<std::size_t>(BLOCK_SIZE) - 1) /
                               static_cast<std::size_t>(BLOCK_SIZE);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<int>(blocks);
}

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

} // namespace

__device__ inline float deviceInfinityFloat()
{
    return __int_as_float(0x7f800000);
}

__device__ inline bool accumulateMatrixProduct(float lhs, float rhs, float &sum)
{
    const float product = lhs * rhs;
    const float next = sum + product;
    if (!isfinite(lhs) || !isfinite(rhs) || !isfinite(product) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

/**
 * @brief GPU Dirac operator matrix construction
 *
 * The Dirac operator D = d + d* where d is exterior derivative
 */
__global__ __launch_bounds__(BLOCK_SIZE) void diracOperatorKernel(
    const int *__restrict__ boundary_row_ptr, const int *__restrict__ boundary_col_idx,
    const float *__restrict__ boundary_values, const int *__restrict__ coboundary_row_ptr,
    const int *__restrict__ coboundary_col_idx, const float *__restrict__ coboundary_values,
    float *__restrict__ dirac_matrix, int total_dimension,
    int *__restrict__ dimension_offsets // [0, n0, n0+n1, n0+n1+n2, ...]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (total_dimension < 0 || total_dimension > MAX_INT_SQUARE_DIMENSION)
    {
        return;
    }

    const int total_entries = total_dimension * total_dimension;
    if (idx >= total_entries)
        return;

    int row = idx / total_dimension;
    int col = idx % total_dimension;

    // Determine which form degree each index belongs to
    int row_dim = 0, col_dim = 0;
    for (int d = 0; d < 4; ++d)
    {
        if (row >= dimension_offsets[d] && row < dimension_offsets[d + 1])
        {
            row_dim = d;
        }
        if (col >= dimension_offsets[d] && col < dimension_offsets[d + 1])
        {
            col_dim = d;
        }
    }

    float value = 0.0f;

    // Dirac operator couples forms of degree d and d+/-1
    if (col_dim == row_dim + 1)
    {
        // d* contribution (adjoint of d)
        int local_col = col - dimension_offsets[col_dim];
        int local_row = row - dimension_offsets[row_dim];

        // Look up in coboundary matrix
        int start = coboundary_row_ptr[local_row];
        int end = coboundary_row_ptr[local_row + 1];

        for (int j = start; j < end; ++j)
        {
            if (coboundary_col_idx[j] == local_col)
            {
                value = coboundary_values[j];
                break;
            }
        }
    }
    else if (col_dim == row_dim - 1)
    {
        // d contribution (boundary)
        int local_col = col - dimension_offsets[col_dim];
        int local_row = row - dimension_offsets[row_dim];

        // Look up in boundary matrix
        int start = boundary_row_ptr[local_row];
        int end = boundary_row_ptr[local_row + 1];

        for (int j = start; j < end; ++j)
        {
            if (boundary_col_idx[j] == local_col)
            {
                value = boundary_values[j];
                break;
            }
        }
    }

    dirac_matrix[row * total_dimension + col] = isfinite(value) ? value : deviceInfinityFloat();
}

/**
 * @brief Hodge star operator kernel
 *
 * Maps k-forms to (n-k)-forms with sign pattern
 */
__global__ __launch_bounds__(BLOCK_SIZE) void hodgeStarKernel(const float *__restrict__ input_form,
                                                              float *__restrict__ output_form,
                                                              int form_degree, int n,
                                                              const int *__restrict__ simplex_signs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of k-simplices (combinatorial count)
    int num_k_simplices = 1; // Will be computed by the loop below
    for (int i = 0; i <= form_degree; ++i)
    {
        num_k_simplices *= (n - i + 1);
        num_k_simplices /= (i + 1);
    }

    if (idx >= num_k_simplices)
        return;

    // Diagonal orientation-weighted Hodge-star approximation used by this
    // GPU path: *omega_i = sign_i * omega_i.
    float sign = (simplex_signs[idx] % 2 == 0) ? 1.0f : -1.0f;
    const float value = sign * input_form[idx];
    output_form[idx] = isfinite(input_form[idx]) && isfinite(value) ? value : deviceInfinityFloat();
}

/**
 * @brief Laplacian from Dirac operator: L = D^2 = (d + d*)^2 = dd* + d*d
 */
__global__
__launch_bounds__(BLOCK_SIZE) void laplacianFromDiracKernel(const float *__restrict__ dirac_matrix,
                                                            float *__restrict__ laplacian, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < 0 || n > MAX_INT_SQUARE_DIMENSION)
    {
        return;
    }

    const int total_entries = n * n;
    if (idx >= total_entries)
        return;

    int row = idx / n;
    int col = idx % n;

    // Compute matrix square: L = D^T * D
    float sum = 0.0f;
    bool valid_sum = true;
    for (int k = 0; k < n; ++k)
    {
        if (!accumulateMatrixProduct(dirac_matrix[k * n + row], dirac_matrix[k * n + col], sum))
        {
            valid_sum = false;
            break;
        }
    }

    laplacian[row * n + col] = valid_sum ? sum : deviceInfinityFloat();
}

/**
 * @brief GPU Dirac operator solver class
 */
class GPUDiracOperator
{
public:
    GPUDiracOperator(int max_dimension, const std::vector<int> &dimension_sizes)
        : max_dimension_(max_dimension)
    {
        if (max_dimension_ < 0)
        {
            throw std::invalid_argument("GPU Dirac max dimension must be nonnegative");
        }
        const auto dimension_count = static_cast<std::size_t>(max_dimension_) + 1;
        if (dimension_sizes.size() < dimension_count)
        {
            throw std::invalid_argument("GPU Dirac dimension sizes do not cover max dimension");
        }

        // Compute dimension offsets
        dimension_offsets_.resize(dimension_count + 1, 0);
        for (int d = 0; d <= max_dimension_; ++d)
        {
            dimension_offsets_[static_cast<std::size_t>(d) + 1] =
                checkedAddInt(dimension_offsets_[static_cast<std::size_t>(d)],
                              dimension_sizes[static_cast<std::size_t>(d)],
                              "GPU Dirac total dimension exceeds int range");
        }

        total_dimension_ = dimension_offsets_.back();
        const std::size_t matrix_entries =
            checkedSquareElements(total_dimension_, "GPU Dirac matrix size overflows");
        const std::size_t matrix_bytes = checkedByteCount(
            matrix_entries, sizeof(float), "GPU Dirac matrix allocation exceeds host limits");
        const std::size_t dimension_offset_bytes =
            checkedByteCount(dimension_offsets_.size(), sizeof(int),
                             "GPU Dirac dimension offset allocation exceeds host limits");

        try
        {
            // Allocate GPU memory
            GPU_CHECK(cudaMalloc(&d_dimension_offsets_, dimension_offset_bytes));
            GPU_CHECK(cudaMemcpy(d_dimension_offsets_, dimension_offsets_.data(),
                                 dimension_offset_bytes, cudaMemcpyHostToDevice));

            GPU_CHECK(cudaMalloc(&d_dirac_matrix_, matrix_bytes));
            GPU_CHECK(cudaMalloc(&d_laplacian_, matrix_bytes));
        }
        catch (...)
        {
            cleanup();
            throw;
        }
    }

    ~GPUDiracOperator() { cleanup(); }

    void buildDiracOperator(const std::vector<int> &boundary_row_ptr,
                            const std::vector<int> &boundary_col_idx,
                            const std::vector<float> &boundary_values,
                            const std::vector<int> &coboundary_row_ptr,
                            const std::vector<int> &coboundary_col_idx,
                            const std::vector<float> &coboundary_values)
    {
        // Allocate and copy boundary/coboundary to GPU
        int *d_boundary_row_ptr = nullptr;
        int *d_boundary_col_idx = nullptr;
        float *d_boundary_values = nullptr;
        int *d_coboundary_row_ptr = nullptr;
        int *d_coboundary_col_idx = nullptr;
        float *d_coboundary_values = nullptr;

        if (!valuesAreFinite(boundary_values) || !valuesAreFinite(coboundary_values))
        {
            throw std::invalid_argument("GPU Dirac boundary values must be finite");
        }

        const auto cleanup_sparse = [&]() {
            cudaFree(d_boundary_row_ptr);
            cudaFree(d_boundary_col_idx);
            cudaFree(d_boundary_values);
            cudaFree(d_coboundary_row_ptr);
            cudaFree(d_coboundary_col_idx);
            cudaFree(d_coboundary_values);
        };

        const std::size_t boundary_row_ptr_bytes = checkedByteCount(
            boundary_row_ptr.size(), sizeof(int), "GPU Dirac boundary row pointer size overflows");
        const std::size_t boundary_col_idx_bytes = checkedByteCount(
            boundary_col_idx.size(), sizeof(int), "GPU Dirac boundary column index size overflows");
        const std::size_t boundary_values_bytes = checkedByteCount(
            boundary_values.size(), sizeof(float), "GPU Dirac boundary value size overflows");
        const std::size_t coboundary_row_ptr_bytes =
            checkedByteCount(coboundary_row_ptr.size(), sizeof(int),
                             "GPU Dirac coboundary row pointer size overflows");
        const std::size_t coboundary_col_idx_bytes =
            checkedByteCount(coboundary_col_idx.size(), sizeof(int),
                             "GPU Dirac coboundary column index size overflows");
        const std::size_t coboundary_values_bytes = checkedByteCount(
            coboundary_values.size(), sizeof(float), "GPU Dirac coboundary value size overflows");

        try
        {
            GPU_CHECK(cudaMalloc(&d_boundary_row_ptr, boundary_row_ptr_bytes));
            GPU_CHECK(cudaMalloc(&d_boundary_col_idx, boundary_col_idx_bytes));
            GPU_CHECK(cudaMalloc(&d_boundary_values, boundary_values_bytes));
            GPU_CHECK(cudaMalloc(&d_coboundary_row_ptr, coboundary_row_ptr_bytes));
            GPU_CHECK(cudaMalloc(&d_coboundary_col_idx, coboundary_col_idx_bytes));
            GPU_CHECK(cudaMalloc(&d_coboundary_values, coboundary_values_bytes));

            GPU_CHECK(cudaMemcpy(d_boundary_row_ptr, boundary_row_ptr.data(),
                                 boundary_row_ptr_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_boundary_col_idx, boundary_col_idx.data(),
                                 boundary_col_idx_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_boundary_values, boundary_values.data(), boundary_values_bytes,
                                 cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_coboundary_row_ptr, coboundary_row_ptr.data(),
                                 coboundary_row_ptr_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_coboundary_col_idx, coboundary_col_idx.data(),
                                 coboundary_col_idx_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_coboundary_values, coboundary_values.data(),
                                 coboundary_values_bytes, cudaMemcpyHostToDevice));

            // Launch kernel to build Dirac operator
            const int blocks = checkedGridBlocks(
                checkedSquareElements(total_dimension_, "GPU Dirac grid size overflows"),
                "GPU Dirac grid exceeds CUDA limits");
            diracOperatorKernel<<<blocks, BLOCK_SIZE>>>(
                d_boundary_row_ptr, d_boundary_col_idx, d_boundary_values, d_coboundary_row_ptr,
                d_coboundary_col_idx, d_coboundary_values, d_dirac_matrix_, total_dimension_,
                d_dimension_offsets_);
            GPU_CHECK(cudaPeekAtLastError());
        }
        catch (...)
        {
            cleanup_sparse();
            throw;
        }

        // Cleanup
        cleanup_sparse();
    }

    void computeLaplacian()
    {
        const int blocks = checkedGridBlocks(
            checkedSquareElements(total_dimension_, "GPU Dirac grid size overflows"),
            "GPU Dirac Laplacian grid exceeds CUDA limits");
        laplacianFromDiracKernel<<<blocks, BLOCK_SIZE>>>(d_dirac_matrix_, d_laplacian_,
                                                         total_dimension_);
        GPU_CHECK(cudaPeekAtLastError());
    }

    void getLaplacian(std::vector<float> &laplacian)
    {
        const std::size_t matrix_entries =
            checkedSquareElements(total_dimension_, "GPU Dirac output size overflows");
        const std::size_t matrix_bytes = checkedByteCount(
            matrix_entries, sizeof(float), "GPU Dirac output allocation exceeds host limits");
        laplacian.resize(matrix_entries);
        GPU_CHECK(cudaMemcpy(laplacian.data(), d_laplacian_, matrix_bytes, cudaMemcpyDeviceToHost));
    }

    void applyHodgeStar(const std::vector<float> &input, std::vector<float> &output,
                        int form_degree)
    {
        if (form_degree < 0)
        {
            throw std::invalid_argument("GPU Dirac Hodge star form degree must be nonnegative");
        }

        // Allocate GPU memory
        float *d_input = nullptr;
        float *d_output = nullptr;
        int *d_signs = nullptr;

        const int n = checkedIntSize(input.size(), "GPU Dirac Hodge star input exceeds int range");
        if (!valuesAreFinite(input))
        {
            throw std::invalid_argument("GPU Dirac Hodge star input must be finite");
        }
        const std::size_t input_bytes =
            checkedByteCount(input.size(), sizeof(float),
                             "GPU Dirac Hodge star input allocation exceeds host limits");
        const std::size_t signs_bytes = checkedByteCount(
            input.size(), sizeof(int), "GPU Dirac Hodge star sign allocation exceeds host limits");

        try
        {
            GPU_CHECK(cudaMalloc(&d_input, input_bytes));
            GPU_CHECK(cudaMalloc(&d_output, input_bytes));
            GPU_CHECK(cudaMalloc(&d_signs, signs_bytes));

            GPU_CHECK(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

            // Initialize signs for form basis
            // In 3D, default orientation gives all positive signs
            std::vector<int> signs(input.size(), 1);
            GPU_CHECK(cudaMemcpy(d_signs, signs.data(), signs_bytes, cudaMemcpyHostToDevice));

            // Launch kernel
            const int blocks =
                checkedGridBlocks(input.size(), "GPU Dirac Hodge grid exceeds CUDA limits");
            hodgeStarKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, form_degree, n, d_signs);
            GPU_CHECK(cudaPeekAtLastError());

            // Copy back
            output.resize(input.size());
            GPU_CHECK(cudaMemcpy(output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));
        }
        catch (...)
        {
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_signs);
            throw;
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_signs);
    }

private:
    int max_dimension_;
    int total_dimension_;
    std::vector<int> dimension_offsets_;

    int *d_dimension_offsets_ = nullptr;
    float *d_dirac_matrix_ = nullptr;
    float *d_laplacian_ = nullptr;

    void cleanup() noexcept
    {
        cudaFree(d_dimension_offsets_);
        cudaFree(d_dirac_matrix_);
        cudaFree(d_laplacian_);
        d_dimension_offsets_ = nullptr;
        d_dirac_matrix_ = nullptr;
        d_laplacian_ = nullptr;
    }
};

/**
 * @brief Benchmark Dirac operator computation
 */
struct DiracBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int max_dimension;
    int total_dimension;
};

DiracBenchmark benchmarkDirac(int max_dim, const std::vector<int> &dim_sizes)
{
    if (max_dim < 0)
    {
        throw std::invalid_argument("Dirac benchmark max dimension must be nonnegative");
    }

    DiracBenchmark bench;
    bench.max_dimension = max_dim;

    int total_dim = 0;
    for (int s : dim_sizes)
    {
        total_dim =
            checkedAddInt(total_dim, s, "Dirac benchmark total dimension exceeds int range");
    }
    if (total_dim < 2)
    {
        throw std::invalid_argument("Dirac benchmark requires at least two basis elements");
    }
    bench.total_dimension = total_dim;
    const std::size_t matrix_entries =
        checkedSquareElements(total_dim, "Dirac benchmark matrix size overflows");
    const auto total_size = static_cast<std::size_t>(total_dim);

    // Generate test boundary/coboundary matrices
    std::vector<int> boundary_row_ptr(total_size + 1);
    std::vector<int> boundary_col_idx;
    std::vector<float> boundary_values;

    for (int i = 0; i <= total_dim; ++i)
    {
        boundary_row_ptr[static_cast<std::size_t>(i)] =
            checkedAddInt(i, i, "Dirac benchmark row pointer exceeds int range");
        if (i < total_dim)
        {
            boundary_col_idx.push_back(i % (total_dim / 2));
            boundary_col_idx.push_back((i + 1) % (total_dim / 2));
            boundary_values.push_back(1.0f);
            boundary_values.push_back(-1.0f);
        }
    }

    // CPU baseline - compute full Dirac operator
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_dirac(matrix_entries, 0.0f);

    // Build D = d + d^* from the synthetic sparse boundary operator used in this benchmark.
    for (int row = 0; row < total_dim; ++row)
    {
        const int start = boundary_row_ptr[static_cast<std::size_t>(row)];
        const int end = boundary_row_ptr[static_cast<std::size_t>(row) + 1];
        for (int k = start; k < end; ++k)
        {
            const int col = boundary_col_idx[static_cast<std::size_t>(k)] % total_dim;
            const float val = boundary_values[static_cast<std::size_t>(k)];
            cpu_dirac[static_cast<std::size_t>(row) * total_size + static_cast<std::size_t>(col)] =
                val;
            cpu_dirac[static_cast<std::size_t>(col) * total_size + static_cast<std::size_t>(row)] =
                val;
        }
    }

    // Compute D^2 (Dirac Laplacian) for benchmark
    std::vector<float> dirac_laplacian(matrix_entries, 0.0f);
    for (int i = 0; i < total_dim; ++i)
    {
        for (int j = 0; j < total_dim; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < total_dim; ++k)
            {
                sum += cpu_dirac[static_cast<std::size_t>(i) * total_size +
                                 static_cast<std::size_t>(k)] *
                       cpu_dirac[static_cast<std::size_t>(k) * total_size +
                                 static_cast<std::size_t>(j)];
            }
            dirac_laplacian[static_cast<std::size_t>(i) * total_size +
                            static_cast<std::size_t>(j)] = sum;
        }
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU version
    GPUDiracOperator gpu_dirac(max_dim, dim_sizes);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    gpu_dirac.buildDiracOperator(boundary_row_ptr, boundary_col_idx, boundary_values,
                                 boundary_row_ptr, boundary_col_idx, boundary_values);
    gpu_dirac.computeLaplacian();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);

    return bench;
}

} // namespace gpu
} // namespace spectral
} // namespace nerve
