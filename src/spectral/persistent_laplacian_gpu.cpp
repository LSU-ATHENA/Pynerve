#include "nerve/errors/errors.hpp"
#include "nerve/runtime/hardware_probe.hpp"
#include "nerve/spectral/persistent_laplacian.hpp"
#include "persistent_laplacian_gpu_internal.hpp"

#ifdef NERVE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace nerve::spectral
{
namespace
{

struct GpuContext
{
    int device_id = 0;
    std::size_t free_memory_bytes = 0;
};

std::mutex g_host_gpu_limit_mutex;
std::unordered_map<const PersistentLaplacianSolverGPU *, std::size_t> g_host_gpu_limits_mb;

std::size_t getConfiguredLimitMb(const PersistentLaplacianSolverGPU *solver)
{
    std::lock_guard<std::mutex> lock(g_host_gpu_limit_mutex);
    const auto it = g_host_gpu_limits_mb.find(solver);
    return it != g_host_gpu_limits_mb.end() ? it->second : 0U;
}

} // namespace

PersistentLaplacianSolverGPU::PersistentLaplacianSolverGPU(const SpectralConfig &config)
    : config_(config)
    , gpu_context_(nullptr)
    , gpu_memory_pool_(nullptr)
{
    std::lock_guard<std::mutex> lock(g_host_gpu_limit_mutex);
    g_host_gpu_limits_mb[this] = 0U;
    initializeGPU();
}

PersistentLaplacianSolverGPU::~PersistentLaplacianSolverGPU()
{
    cleanupGPU();
    std::lock_guard<std::mutex> lock(g_host_gpu_limit_mutex);
    g_host_gpu_limits_mb.erase(this);
}

bool PersistentLaplacianSolverGPU::initializeGPU()
{
#ifdef NERVE_HAS_CUDA
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (!runtime::has_cuda_gpu(snapshot))
    {
        gpu_context_ = nullptr;
        return false;
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        gpu_context_ = nullptr;
        return false;
    }

    auto *ctx = new GpuContext();
    ctx->device_id = 0;
    cudaSetDevice(0);

    std::size_t free_mem = 0;
    std::size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    ctx->free_memory_bytes = free_mem;

    gpu_context_ = static_cast<void *>(ctx);
    return true;
#else
    (void)config_;
    gpu_context_ = nullptr;
    return false;
#endif
}

void PersistentLaplacianSolverGPU::cleanupGPU()
{
    if (gpu_context_ != nullptr)
    {
        delete static_cast<GpuContext *>(gpu_context_);
        gpu_context_ = nullptr;
    }
    gpu_memory_pool_ = nullptr;
}

errors::ErrorResult<SpectralDecomposition>
PersistentLaplacianSolverGPU::computeSpectrumGpu(const Eigen::SparseMatrix<double> &laplacian)
{
    if (!isAvailable())
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "GPU not available  --  no CUDA device found");
    }
#ifdef NERVE_HAS_CUDA
    const int n = static_cast<int>(laplacian.rows());
    if (n == 0)
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Empty Laplacian matrix");
    }

    CsrStorage csr = buildCsrOnDevice(laplacian);
    if (csr.nnz == 0)
    {
        csr.release();
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Laplacian matrix has no non-zero entries");
    }

    std::vector<double> v0(n, 1.0 / std::sqrt(static_cast<double>(n)));
    double *d_v0 = nullptr;
    cudaMalloc(&d_v0, n * sizeof(double));
    cudaMemcpy(d_v0, v0.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const int num_requested =
        config_.num_eigenvalues > 0 ? config_.num_eigenvalues + 10 : kDefaultKrylovDim;
    std::vector<double> alpha, beta;
    lanczosIteration(csr, d_v0, n, num_requested, alpha, beta);

    cudaFree(d_v0);
    csr.release();

    SpectralDecomposition result = tridiagToEigenpairs(alpha, beta, n, config_.num_eigenvalues);
    return errors::ErrorResult<SpectralDecomposition>::success(result);
#else
    return errors::ErrorResult<SpectralDecomposition>::error(errors::ErrorCode::E51_LAPLACIAN_ABORT,
                                                             "Built without CUDA support");
#endif
}

errors::ErrorResult<SpectralDecomposition>
PersistentLaplacianSolverGPU::computeSpectrumGpuWithWarmStart(
    const Eigen::SparseMatrix<double> &laplacian, const SpectralDecomposition &previous_result)
{
    if (!isAvailable())
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "GPU not available  --  no CUDA device found");
    }
#ifdef NERVE_HAS_CUDA
    const int n = static_cast<int>(laplacian.rows());
    if (n == 0)
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Empty Laplacian matrix");
    }

    CsrStorage csr = buildCsrOnDevice(laplacian);
    if (csr.nnz == 0)
    {
        csr.release();
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Laplacian matrix has no non-zero entries");
    }

    std::vector<double> v0(n, 0.0);
    if (previous_result.eigenvalues.empty())
    {
        for (int i = 0; i < n; ++i)
            v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }
    else
    {
        const double shift = previous_result.eigenvalues[0];
        for (int i = 0; i < n; ++i)
            v0[i] = (i < static_cast<int>(previous_result.eigenvalues.size()))
                        ? std::abs(previous_result.eigenvalues[i] - shift)
                        : 1.0;
        const double v_norm = std::sqrt(std::inner_product(v0.begin(), v0.end(), v0.begin(), 0.0));
        if (v_norm > kOrthoTol)
            for (auto &vi : v0)
                vi /= v_norm;
        else
            for (int i = 0; i < n; ++i)
                v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }

    double *d_v0 = nullptr;
    cudaMalloc(&d_v0, n * sizeof(double));
    cudaMemcpy(d_v0, v0.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const int num_requested =
        config_.num_eigenvalues > 0 ? config_.num_eigenvalues + 10 : kDefaultKrylovDim;
    std::vector<double> alpha, beta;
    lanczosIteration(csr, d_v0, n, num_requested, alpha, beta);

    cudaFree(d_v0);
    csr.release();

    SpectralDecomposition result = tridiagToEigenpairs(alpha, beta, n, config_.num_eigenvalues);
    return errors::ErrorResult<SpectralDecomposition>::success(result);
#else
    (void)previous_result;
    return errors::ErrorResult<SpectralDecomposition>::error(errors::ErrorCode::E51_LAPLACIAN_ABORT,
                                                             "Built without CUDA support");
#endif
}

errors::ErrorResult<SpectralDecomposition>
PersistentLaplacianSolverGPU::gpuArnoldi(const Eigen::SparseMatrix<double> &laplacian,
                                         const std::vector<Eigenpair> &warm_start)
{
    if (!isAvailable())
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "GPU not available  --  no CUDA device found");
    }
#ifdef NERVE_HAS_CUDA
    const int n = static_cast<int>(laplacian.rows());
    if (n == 0)
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Empty Laplacian matrix");
    }

    CsrStorage csr = buildCsrOnDevice(laplacian);
    if (csr.nnz == 0)
    {
        csr.release();
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Laplacian matrix has no non-zero entries");
    }

    const int block_size = kDefaultBlockSize;
    const int grid_size = (n + block_size - 1) / block_size;

    std::vector<double> v0(n, 0.0);
    if (warm_start.empty())
    {
        for (int i = 0; i < n; ++i)
            v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }
    else
    {
        for (int i = 0; i < n && i < static_cast<int>(warm_start.size()); ++i)
            v0[i] = warm_start[i].value;
        double v_norm = 0.0;
        for (int i = 0; i < n; ++i)
            v_norm += v0[i] * v0[i];
        v_norm = std::sqrt(v_norm);
        if (v_norm > kOrthoTol)
            for (auto &vi : v0)
                vi /= v_norm;
        else
            for (int i = 0; i < n; ++i)
                v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }

    const int krylov_dim = std::min(
        config_.num_eigenvalues > 0 ? config_.num_eigenvalues + 10 : kDefaultKrylovDim, n - 1);

    std::vector<double> H_data(krylov_dim * (krylov_dim + 1), 0.0);

    double *d_v_curr = nullptr;
    double *d_v_next = nullptr;
    double *d_w = nullptr;
    cudaMalloc(&d_v_curr, n * sizeof(double));
    cudaMalloc(&d_v_next, n * sizeof(double));
    cudaMalloc(&d_w, n * sizeof(double));
    cudaMemcpy(d_v_curr, v0.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    for (int j = 0; j < krylov_dim; ++j)
    {
        // cppcheck-suppress shiftTooManyBits
        csrSpMVKernel<<<grid_size, block_size>>>(csr.n, csr.d_row_offsets, csr.d_col_indices,
                                                 csr.d_values, d_v_curr, d_w);

        for (int i = 0; i <= j; ++i)
        {
            H_data[i * krylov_dim + j] = dotProduct(d_v_curr, d_w, n);
            // cppcheck-suppress shiftTooManyBits
            orthogonalizeKernel<<<grid_size, block_size>>>(n, d_v_curr, d_w,
                                                           H_data[i * krylov_dim + j]);
        }

        H_data[(j + 1) * krylov_dim + j] = norm(d_w, n);
        if (H_data[(j + 1) * krylov_dim + j] < kOrthoTol)
            break;

        // cppcheck-suppress shiftTooManyBits
        scaleKernel<<<grid_size, block_size>>>(n, 1.0 / H_data[(j + 1) * krylov_dim + j], d_w);
        cudaMemcpy(d_v_curr, d_w, n * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_v_curr);
    cudaFree(d_v_next);
    cudaFree(d_w);
    csr.release();

    SpectralDecomposition result;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(krylov_dim, krylov_dim);
    for (int i = 0; i < krylov_dim; ++i)
        for (int k = 0; k < krylov_dim; ++k)
            H(i, k) = H_data[i * krylov_dim + k];

    Eigen::EigenSolver<Eigen::MatrixXd> solver(H);
    if (solver.info() != Eigen::Success)
        return result;

    const int num_out =
        config_.num_eigenvalues > 0 ? std::min(config_.num_eigenvalues, krylov_dim) : krylov_dim;
    result.eigenvalues.resize(num_out);
    for (int i = 0; i < num_out; ++i)
        result.eigenvalues[i] = std::real(solver.eigenvalues()[i]);

    return errors::ErrorResult<SpectralDecomposition>::success(result);
#else
    (void)warm_start;
    return errors::ErrorResult<SpectralDecomposition>::error(errors::ErrorCode::E51_LAPLACIAN_ABORT,
                                                             "Built without CUDA support");
#endif
}

errors::ErrorResult<SpectralDecomposition>
PersistentLaplacianSolverGPU::gpuLanczos(const Eigen::SparseMatrix<double> &laplacian,
                                         const std::vector<Eigenpair> &warm_start)
{
    if (!isAvailable())
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "GPU not available  --  no CUDA device found");
    }
#ifdef NERVE_HAS_CUDA
    const int n = static_cast<int>(laplacian.rows());
    if (n == 0)
    {
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Empty Laplacian matrix");
    }

    CsrStorage csr = buildCsrOnDevice(laplacian);
    if (csr.nnz == 0)
    {
        csr.release();
        return errors::ErrorResult<SpectralDecomposition>::error(
            errors::ErrorCode::E51_LAPLACIAN_ABORT, "Laplacian matrix has no non-zero entries");
    }

    std::vector<double> v0(n, 0.0);
    if (warm_start.empty())
    {
        for (int i = 0; i < n; ++i)
            v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }
    else
    {
        for (int i = 0; i < n && i < static_cast<int>(warm_start.size()); ++i)
            v0[i] = warm_start[i].value;
        double v_norm = 0.0;
        for (int i = 0; i < n; ++i)
            v_norm += v0[i] * v0[i];
        v_norm = std::sqrt(v_norm);
        if (v_norm > kOrthoTol)
            for (auto &vi : v0)
                vi /= v_norm;
        else
            for (int i = 0; i < n; ++i)
                v0[i] = 1.0 / std::sqrt(static_cast<double>(n));
    }

    double *d_v0 = nullptr;
    cudaMalloc(&d_v0, n * sizeof(double));
    cudaMemcpy(d_v0, v0.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const int num_requested =
        config_.num_eigenvalues > 0 ? config_.num_eigenvalues + 10 : kDefaultKrylovDim;
    std::vector<double> alpha, beta;
    lanczosIteration(csr, d_v0, n, num_requested, alpha, beta);

    cudaFree(d_v0);
    csr.release();

    SpectralDecomposition result = tridiagToEigenpairs(alpha, beta, n, config_.num_eigenvalues);
    return errors::ErrorResult<SpectralDecomposition>::success(result);
#else
    (void)warm_start;
    return errors::ErrorResult<SpectralDecomposition>::error(errors::ErrorCode::E51_LAPLACIAN_ABORT,
                                                             "Built without CUDA support");
#endif
}

bool PersistentLaplacianSolverGPU::isAvailable() const
{
    return gpu_context_ != nullptr;
}

std::string PersistentLaplacianSolverGPU::getGpuInfo() const
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (!snapshot.gpus.ok() || snapshot.gpus.value.empty())
    {
        return "GPU probe: no CUDA devices detected";
    }

    const runtime::GpuDeviceInfo &gpu = snapshot.gpus.value.front();
    std::ostringstream ss;
    ss << "GPU probe device: " << gpu.name << ", cc " << gpu.compute_capability_major << "."
       << gpu.compute_capability_minor << ", total " << (gpu.total_memory_bytes / kBytesPerMb)
       << " MB" << ", free " << (gpu.free_memory_bytes / kBytesPerMb) << " MB"
       << ", spectral GPU execution: Lanczos CSR SpMV";
    return ss.str();
}

void PersistentLaplacianSolverGPU::setGpuMemoryLimit(std::size_t limit_mb)
{
    std::lock_guard<std::mutex> lock(g_host_gpu_limit_mutex);
    g_host_gpu_limits_mb[this] = limit_mb;
}

std::size_t PersistentLaplacianSolverGPU::getGpuMemoryUsage() const
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (!snapshot.gpus.ok() || snapshot.gpus.value.empty())
    {
        return 0U;
    }

    const runtime::GpuDeviceInfo &gpu = snapshot.gpus.value.front();
    const std::uint64_t used_bytes = gpu.total_memory_bytes > gpu.free_memory_bytes
                                         ? (gpu.total_memory_bytes - gpu.free_memory_bytes)
                                         : 0U;
    const std::size_t usage_mb = static_cast<std::size_t>(used_bytes / kBytesPerMb);
    const std::size_t limit_mb = getConfiguredLimitMb(this);
    return (limit_mb > 0U) ? std::min(usage_mb, limit_mb) : usage_mb;
}

} // namespace nerve::spectral
