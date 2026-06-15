
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{
namespace filtration
{
namespace vr
{
namespace sparse
{
namespace gpu
{
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_GRID_BLOCKS = 1024;

struct Point
{
    float x;
    float y;
    float z;
};

namespace
{

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCuSparse(cusparseStatus_t status, const char *operation)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(operation) + ": cuSPARSE status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::size_t checkedProduct(std::size_t lhs, std::size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        throw std::overflow_error(context);
    }
    return lhs * rhs;
}

std::size_t checkedByteCount(std::size_t count, std::size_t element_size, const char *context)
{
    return checkedProduct(count, element_size, context);
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

int checkedPointCount(std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("sparse GPU VR point count exceeds kernel limits");
    }
    return static_cast<int>(size);
}

void validateSparseInput(const std::vector<Point> &points, float threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0f)
    {
        throw std::invalid_argument("sparse GPU VR threshold must be finite and nonnegative");
    }
    checkedPointCount(points.size());
    for (const auto &point : points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            throw std::invalid_argument("sparse GPU VR points must contain only finite values");
        }
    }
}

float checkedThresholdSquare(float threshold)
{
    const double threshold_sq = static_cast<double>(threshold) * static_cast<double>(threshold);
    if (!std::isfinite(threshold_sq) ||
        threshold_sq > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::invalid_argument("sparse GPU VR threshold square exceeds supported range");
    }
    return static_cast<float>(threshold_sq);
}

float checkedPointDistanceSquare(const Point &a, const Point &b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz) || !std::isfinite(dist_sq) ||
        dist_sq > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::invalid_argument("sparse GPU VR point distance exceeds supported range");
    }
    return static_cast<float>(dist_sq);
}

void safeCudaFree(void *ptr)
{
    if (ptr)
    {
        cudaFree(ptr);
    }
}

} // namespace

__device__ inline bool sparseDistanceSquare(float dx, float dy, float dz, float &dist_sq)
{
    const float dx_sq = dx * dx;
    const float dy_sq = dy * dy;
    const float dz_sq = dz * dz;
    const float xy_sq = dx_sq + dy_sq;
    dist_sq = xy_sq + dz_sq;
    return isfinite(dx) && isfinite(dy) && isfinite(dz) && isfinite(dx_sq) && isfinite(dy_sq) &&
           isfinite(dz_sq) && isfinite(xy_sq) && isfinite(dist_sq);
}

__global__ void __launch_bounds__(256)
    sparseEdgeDetectionKernel(const float *__restrict__ points_x,
                              const float *__restrict__ points_y,
                              const float *__restrict__ points_z, int *__restrict__ row_counts,
                              int n, float threshold_sq)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float px = points_x[i];
        float py = points_y[i];
        float pz = points_z[i];

        int edge_count = 0;
        for (int j = i + 1; j < n; ++j)
        {
            float qx = points_x[j];
            float qy = points_y[j];
            float qz = points_z[j];

            float dx = px - qx;
            float dy = py - qy;
            float dz = pz - qz;
            float dist_sq = 0.0f;

            if (sparseDistanceSquare(dx, dy, dz, dist_sq) && dist_sq <= threshold_sq)
            {
                edge_count++;
            }
        }
        row_counts[i] = edge_count;
    }
}
__global__ void __launch_bounds__(256)
    fillCSRColIdxKernel(const float *__restrict__ points_x, const float *__restrict__ points_y,
                        const float *__restrict__ points_z, const int *__restrict__ row_ptr,
                        int *__restrict__ col_idx, float *__restrict__ data, int n,
                        float threshold_sq)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float px = points_x[i];
        float py = points_y[i];
        float pz = points_z[i];

        int write_idx = row_ptr[i];

        for (int j = i + 1; j < n; ++j)
        {
            float dx = px - points_x[j];
            float dy = py - points_y[j];
            float dz = pz - points_z[j];
            float dist_sq = 0.0f;

            if (sparseDistanceSquare(dx, dy, dz, dist_sq) && dist_sq <= threshold_sq)
            {
                col_idx[write_idx] = j;
                data[write_idx] = sqrtf(dist_sq);
                write_idx++;
            }
        }
    }
}
struct SparseVRComplex
{
    int n = 0;
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    std::vector<float> h_data;
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    float *d_data = nullptr;

    void allocateGPU()
    {
        freeGPU();
        if (h_row_ptr.empty())
        {
            h_row_ptr.assign(1, 0);
        }

        try
        {
            const std::size_t row_ptr_bytes = checkedByteCount(
                h_row_ptr.size(), sizeof(int), "sparse GPU VR row pointer exceeds host limits");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes),
                      "cudaMalloc sparse VR row pointer");
            checkCuda(
                cudaMemcpy(d_row_ptr, h_row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy sparse VR row pointer host-to-device");

            if (!h_col_idx.empty())
            {
                const std::size_t col_idx_bytes =
                    checkedByteCount(h_col_idx.size(), sizeof(int),
                                     "sparse GPU VR column indices exceed host limits");
                checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes),
                          "cudaMalloc sparse VR column indices");
                checkCuda(
                    cudaMemcpy(d_col_idx, h_col_idx.data(), col_idx_bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy sparse VR column indices host-to-device");
            }
            if (!h_data.empty())
            {
                const std::size_t data_bytes = checkedByteCount(
                    h_data.size(), sizeof(float), "sparse GPU VR data exceeds host limits");
                checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_data), data_bytes),
                          "cudaMalloc sparse VR data");
                checkCuda(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy sparse VR data host-to-device");
            }
        }
        catch (...)
        {
            freeGPU();
            throw;
        }
    }

    void freeGPU()
    {
        if (d_row_ptr)
        {
            cudaFree(d_row_ptr);
            d_row_ptr = nullptr;
        }
        if (d_col_idx)
        {
            cudaFree(d_col_idx);
            d_col_idx = nullptr;
        }
        if (d_data)
        {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }
};
SparseVRComplex buildSparseVRGPU(const std::vector<Point> &points, float threshold)
{
    validateSparseInput(points, threshold);
    const float threshold_sq = checkedThresholdSquare(threshold);
    SparseVRComplex complex;
    complex.n = checkedPointCount(points.size());
    complex.h_row_ptr.assign(points.size() + 1, 0);

    if (points.empty())
        return complex;
    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_z = nullptr;
    int *d_row_counts = nullptr;
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    float *d_data = nullptr;

    auto release = [&]() {
        safeCudaFree(d_x);
        safeCudaFree(d_y);
        safeCudaFree(d_z);
        safeCudaFree(d_row_counts);
        safeCudaFree(d_row_ptr);
        safeCudaFree(d_col_idx);
        safeCudaFree(d_data);
    };

    try
    {
        const std::size_t point_float_bytes = checkedByteCount(
            points.size(), sizeof(float), "sparse GPU VR point allocation exceeds host limits");
        const std::size_t point_int_bytes = checkedByteCount(
            points.size(), sizeof(int), "sparse GPU VR row count allocation exceeds host limits");
        const std::size_t row_ptr_bytes =
            checkedByteCount(complex.h_row_ptr.size(), sizeof(int),
                             "sparse GPU VR row pointer allocation exceeds host limits");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_x), point_float_bytes),
                  "cudaMalloc sparse VR x");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_y), point_float_bytes),
                  "cudaMalloc sparse VR y");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_z), point_float_bytes),
                  "cudaMalloc sparse VR z");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_row_counts), point_int_bytes),
                  "cudaMalloc sparse VR row counts");
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes),
                  "cudaMalloc sparse VR row pointer");

        std::vector<float> xs(points.size());
        std::vector<float> ys(points.size());
        std::vector<float> zs(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            xs[i] = points[i].x;
            ys[i] = points[i].y;
            zs[i] = points[i].z;
        }

        checkCuda(cudaMemcpy(d_x, xs.data(), point_float_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy sparse VR x host-to-device");
        checkCuda(cudaMemcpy(d_y, ys.data(), point_float_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy sparse VR y host-to-device");
        checkCuda(cudaMemcpy(d_z, zs.data(), point_float_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy sparse VR z host-to-device");
        const int blocks =
            std::max(1, std::min(MAX_GRID_BLOCKS,
                                 static_cast<int>((points.size() + BLOCK_SIZE - 1) / BLOCK_SIZE)));

        sparseEdgeDetectionKernel<<<blocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_row_counts, complex.n,
                                                          threshold_sq);
        checkCuda(cudaGetLastError(), "sparseEdgeDetectionKernel launch");
        checkCuda(cudaDeviceSynchronize(), "sparseEdgeDetectionKernel execution");

        std::vector<int> row_counts(points.size(), 0);
        checkCuda(
            cudaMemcpy(row_counts.data(), d_row_counts, point_int_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy sparse VR row counts device-to-host");
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            if (row_counts[i] < 0 ||
                complex.h_row_ptr[i] > std::numeric_limits<int>::max() - row_counts[i])
            {
                throw std::overflow_error("sparse GPU VR edge count exceeds kernel limits");
            }
            complex.h_row_ptr[i + 1] = complex.h_row_ptr[i] + row_counts[i];
        }
        const int total_edges = complex.h_row_ptr.back();
        checkCuda(
            cudaMemcpy(d_row_ptr, complex.h_row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy sparse VR row pointer host-to-device");

        if (total_edges > 0)
        {
            const std::size_t total_edge_count = static_cast<std::size_t>(total_edges);
            const std::size_t col_idx_bytes = checkedByteCount(
                total_edge_count, sizeof(int), "sparse GPU VR column indices exceed host limits");
            const std::size_t data_bytes = checkedByteCount(
                total_edge_count, sizeof(float), "sparse GPU VR data exceeds host limits");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes),
                      "cudaMalloc sparse VR column indices");
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_data), data_bytes),
                      "cudaMalloc sparse VR data");
            fillCSRColIdxKernel<<<blocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_row_ptr, d_col_idx, d_data,
                                                        complex.n, threshold_sq);
            checkCuda(cudaGetLastError(), "fillCSRColIdxKernel launch");
            checkCuda(cudaDeviceSynchronize(), "fillCSRColIdxKernel execution");

            complex.h_col_idx.resize(total_edges);
            complex.h_data.resize(total_edges);

            checkCuda(cudaMemcpy(complex.h_col_idx.data(), d_col_idx, col_idx_bytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy sparse VR column indices device-to-host");
            checkCuda(cudaMemcpy(complex.h_data.data(), d_data, data_bytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy sparse VR data device-to-host");
        }

        release();
    }
    catch (...)
    {
        release();
        throw;
    }

    return complex;
}
class cuSPARSEVR
{
public:
    cuSPARSEVR() { checkCuSparse(cusparseCreate(&handle_), "cusparseCreate"); }

    ~cuSPARSEVR()
    {
        if (handle_)
        {
            cusparseDestroy(handle_);
        }
    }

    void spmv(const SparseVRComplex &complex, const float *x, float *y)
    {
        if (complex.n < 0)
        {
            throw std::invalid_argument("cuSPARSE VR complex has invalid dimension");
        }
        if (complex.n == 0)
        {
            return;
        }
        if (!x || !y || !complex.d_row_ptr)
        {
            throw std::invalid_argument("cuSPARSE VR SpMV received an unallocated buffer");
        }
        if (complex.h_data.empty())
        {
            const std::size_t output_bytes =
                checkedByteCount(static_cast<std::size_t>(complex.n), sizeof(float),
                                 "cuSPARSE VR output exceeds host limits");
            checkCuda(cudaMemset(y, 0, output_bytes), "cudaMemset cuSPARSE VR output");
            return;
        }
        if (!complex.d_col_idx || !complex.d_data)
        {
            throw std::invalid_argument("cuSPARSE VR complex is missing nonzero buffers");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cusparseSpMatDescr_t matA = nullptr;
        cusparseDnVecDescr_t vecX = nullptr;
        cusparseDnVecDescr_t vecY = nullptr;
        void *dBuffer = nullptr;
        auto release = [&]() {
            safeCudaFree(dBuffer);
            if (matA)
                cusparseDestroySpMat(matA);
            if (vecX)
                cusparseDestroyDnVec(vecX);
            if (vecY)
                cusparseDestroyDnVec(vecY);
        };

        try
        {
            checkCuSparse(cusparseCreateCsr(&matA, complex.n, complex.n, complex.h_data.size(),
                                            complex.d_row_ptr, complex.d_col_idx, complex.d_data,
                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                          "cusparseCreateCsr");

            checkCuSparse(cusparseCreateDnVec(&vecX, complex.n, const_cast<float *>(x), CUDA_R_32F),
                          "cusparseCreateDnVec x");
            checkCuSparse(cusparseCreateDnVec(&vecY, complex.n, y, CUDA_R_32F),
                          "cusparseCreateDnVec y");

            size_t bufferSize = 0;
            checkCuSparse(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                                  matA, vecX, &beta, vecY, CUDA_R_32F,
                                                  CUSPARSE_SPMV_CSR_ALG1, &bufferSize),
                          "cusparseSpMV_bufferSize");

            if (bufferSize > 0)
            {
                checkCuda(cudaMalloc(&dBuffer, bufferSize), "cudaMalloc cuSPARSE VR buffer");
            }

            checkCuSparse(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                       vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
                                       dBuffer),
                          "cusparseSpMV");

            release();
        }
        catch (...)
        {
            release();
            throw;
        }
    }

private:
    cusparseHandle_t handle_ = nullptr;
};
struct SparseGPUBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    size_t num_points;
    size_t num_edges;
};

SparseGPUBenchmark benchmarkSparseGPU(int n_points, float threshold)
{
    if (n_points < 0 || !std::isfinite(threshold) || threshold < 0.0f)
    {
        throw std::invalid_argument("benchmarkSparseGPU received an invalid argument");
    }
    const float threshold_sq = checkedThresholdSquare(threshold);

    SparseGPUBenchmark bench;
    bench.num_points = n_points;
    std::vector<Point> points;
    points.reserve(n_points);
    std::mt19937 gen(0x53504152U);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n_points; ++i)
    {
        points.push_back({dis(gen), dis(gen), dis(gen)});
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> cpu_edges;
    for (int i = 0; i < n_points; ++i)
    {
        for (int j = i + 1; j < n_points; ++j)
        {
            if (checkedPointDistanceSquare(points[i], points[j]) <= threshold_sq)
            {
                cpu_edges.push_back({i, j});
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto complex = buildSparseVRGPU(points, threshold);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.num_edges = complex.h_data.size();
    bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);

    return bench;
}

} // namespace gpu
} // namespace sparse
} // namespace vr
} // namespace filtration
} // namespace nerve
