// CUDA distance-matrix API wrappers and stateful class implementation.

#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"
#include "nerve/persistence/cuda/cuda_error_handling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nerve::persistence::accelerated
{

namespace
{

constexpr double kNeutralSpeedup = 1.0;

errors::ErrorResult<void> validateDistanceArguments(const core::BufferView<const double> &points,
                                                    const core::BufferView<double> &distances,
                                                    Size point_dim, double max_radius)
{
    if (point_dim == 0 || points.empty() || points.data() == nullptr ||
        (points.size() % point_dim) != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Invalid point buffer shape");
    }
    if (!std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Radius must be positive");
    }
    const Size n_points = points.size() / point_dim;
    Size distance_elements = 0;
    if (!detail::checkedSizeProduct(n_points, n_points, distance_elements))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Distance matrix size overflow");
    }
    if (distances.data() == nullptr || distances.size() < distance_elements)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Insufficient distance buffer size");
    }
    if (!std::all_of(points.begin(), points.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E20_NUM_NAN,
                                                "Point coordinates must be finite");
    }
    return errors::ErrorResult<void>::ok();
}

bool queryCudaAvailability()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr)
        {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }
    DeviceBuffer &operator=(DeviceBuffer &&) = delete;

    [[nodiscard]] double *data() const noexcept { return ptr_; }
    [[nodiscard]] bool isAllocated() const noexcept { return ptr_ != nullptr; }
    void reset(double *ptr) noexcept { ptr_ = ptr; }

private:
    double *ptr_ = nullptr;
};

errors::ErrorResult<void> allocateDeviceDoubles(Size elements, const std::string &label,
                                                DeviceBuffer &out)
{
    if (elements == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Device allocation requires a positive size");
    }
    Size bytes = 0;
    if (!detail::checkedSizeProduct(elements, sizeof(double), bytes))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Device allocation size overflow");
    }
    double *ptr = nullptr;
    const cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&ptr), bytes);
    if (status != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(status, "cudaMalloc:" + label,
                                                         std::source_location::current());
    }
    out.reset(ptr);
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> copyHostToDevice(DeviceBuffer &dst, const double *src, Size elements,
                                           const std::string &label)
{
    if (!dst.isAllocated() || src == nullptr)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Invalid host-to-device copy arguments");
    }
    Size bytes = 0;
    if (!detail::checkedSizeProduct(elements, sizeof(double), bytes))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Host-to-device copy size overflow");
    }
    const cudaError_t status = cudaMemcpy(dst.data(), src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(status, "cudaMemcpyH2D:" + label,
                                                         std::source_location::current());
    }
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> copyDeviceToHost(double *dst, const DeviceBuffer &src, Size elements,
                                           const std::string &label)
{
    if (!src.isAllocated() || dst == nullptr)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Invalid device-to-host copy arguments");
    }
    Size bytes = 0;
    if (!detail::checkedSizeProduct(elements, sizeof(double), bytes))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT,
                                                "Device-to-host copy size overflow");
    }
    const cudaError_t status = cudaMemcpy(dst, src.data(), bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return cuda_error_handling::check_cuda_operation(status, "cudaMemcpyD2H:" + label,
                                                         std::source_location::current());
    }
    return errors::ErrorResult<void>::ok();
}

void updateMemoryStats(MemoryUsageStats &stats, Size points_elements, Size distance_elements)
{
    const size_t bytes_points = detail::saturatingProduct(points_elements, sizeof(double));
    const size_t bytes_distances = detail::saturatingProduct(distance_elements, sizeof(double));
    stats.total_allocated = detail::saturatingAdd(bytes_points, bytes_distances);
    stats.peak_allocated = std::max(stats.peak_allocated, stats.total_allocated);
    stats.active_allocations = 2;
    stats.pool_bytes = stats.total_allocated;
    stats.pool_free_bytes = 0;
    stats.fragmentation_ratio = 0.0;
}

} // namespace

class CUDADistanceMatrix::Impl
{
public:
    explicit Impl(const CUDADistanceMatrixConfig &cfg)
        : config(cfg)
        , available(queryCudaAvailability())
    {}

    CUDADistanceMatrixConfig config;
    AcceleratedPerformanceStats perf_stats{};
    MemoryUsageStats memory_stats{};
    bool available = false;
};

CUDADistanceMatrix::CUDADistanceMatrix(const CUDADistanceMatrixConfig &config)
    : impl_(std::make_unique<Impl>(config))
{}

CUDADistanceMatrix::~CUDADistanceMatrix() = default;

errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>
CUDADistanceMatrix::create(const CUDADistanceMatrixConfig &config)
{
    auto cfg_result = config.validate();
    if (cfg_result.isError())
    {
        return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::error(
            cfg_result.errorCode(), "Invalid CUDA distance matrix configuration");
    }
    auto instance = std::unique_ptr<CUDADistanceMatrix>(new CUDADistanceMatrix(config));
    return errors::ErrorResult<std::unique_ptr<CUDADistanceMatrix>>::ok(std::move(instance));
}

errors::ErrorResult<void> CUDADistanceMatrix::compute(const core::BufferView<const double> &points,
                                                      core::BufferView<double> &distances,
                                                      Size point_dim, double max_radius)
{
    auto valid = validateDistanceArguments(points, distances, point_dim, max_radius);
    if (valid.isError())
    {
        return valid;
    }

    const Size n_points = points.size() / point_dim;
    auto launch_valid = cuda_host::validateLaunchParams(n_points, point_dim, impl_->config);
    if (launch_valid.isError())
    {
        return launch_valid;
    }
    if (!impl_->available)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "No CUDA device available");
    }

    const Size point_elements = points.size();
    Size distance_elements = 0;
    (void)detail::checkedSizeProduct(n_points, n_points, distance_elements);
    DeviceBuffer d_points;
    DeviceBuffer d_distances;
    auto alloc_points = allocateDeviceDoubles(point_elements, "distance_points", d_points);
    if (alloc_points.isError())
    {
        return alloc_points;
    }
    auto alloc_distances = allocateDeviceDoubles(distance_elements, "distance_matrix", d_distances);
    if (alloc_distances.isError())
    {
        return alloc_distances;
    }
    auto h2d = copyHostToDevice(d_points, points.data(), point_elements, "distance_points");
    if (h2d.isError())
    {
        return h2d;
    }

    const auto start = std::chrono::steady_clock::now();
    auto result = cuda_host::launchDistanceMatrixKernel(
        d_points.data(), d_distances.data(), n_points, point_dim, max_radius, impl_->config);
    if (result.isError())
    {
        return result;
    }
    auto d2h =
        copyDeviceToHost(distances.data(), d_distances, distance_elements, "distance_matrix");
    if (d2h.isError())
    {
        return d2h;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    impl_->perf_stats.total_time_ms += elapsed_ms;
    impl_->perf_stats.gpu_time_ms += elapsed_ms;
    impl_->perf_stats.problems_processed += 1;
    impl_->perf_stats.gpu_used = impl_->available;
    // No CPU baseline is measured in this API path; keep speedup neutral and finite.
    impl_->perf_stats.speedup = kNeutralSpeedup;
    impl_->perf_stats.average_speedup = kNeutralSpeedup;

    updateMemoryStats(impl_->memory_stats, point_elements, distance_elements);

    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void>
CUDADistanceMatrix::computeStreaming(const core::BufferView<const double> &points,
                                     core::BufferView<double> &distances, Size point_dim,
                                     double max_radius, Size stream_size)
{
    auto valid = validateDistanceArguments(points, distances, point_dim, max_radius);
    if (valid.isError())
    {
        return valid;
    }
    if (stream_size == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Stream size must be positive");
    }

    const Size n_points = points.size() / point_dim;
    auto launch_valid = cuda_host::validateLaunchParams(n_points, point_dim, impl_->config);
    if (launch_valid.isError())
    {
        return launch_valid;
    }
    if (!impl_->available)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                "No CUDA device available");
    }

    const Size point_elements = points.size();
    Size total_elements = 0;
    (void)detail::checkedSizeProduct(n_points, n_points, total_elements);
    DeviceBuffer d_points;
    DeviceBuffer d_distances;
    auto alloc_points = allocateDeviceDoubles(point_elements, "stream_points", d_points);
    if (alloc_points.isError())
    {
        return alloc_points;
    }
    auto alloc_distances = allocateDeviceDoubles(total_elements, "stream_distances", d_distances);
    if (alloc_distances.isError())
    {
        return alloc_distances;
    }
    auto h2d = copyHostToDevice(d_points, points.data(), point_elements, "stream_points");
    if (h2d.isError())
    {
        return h2d;
    }

    const auto start = std::chrono::steady_clock::now();
    for (Size offset = 0; offset < total_elements;)
    {
        const Size chunk = std::min(stream_size, total_elements - offset);
        auto result = cuda_host::launchDistanceMatrixKernel(d_points.data(), d_distances.data(),
                                                            n_points, point_dim, max_radius,
                                                            impl_->config, offset, chunk);
        if (result.isError())
        {
            return result;
        }
        offset += chunk;
    }
    auto d2h = copyDeviceToHost(distances.data(), d_distances, total_elements, "stream_distances");
    if (d2h.isError())
    {
        return d2h;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    impl_->perf_stats.total_time_ms += elapsed_ms;
    impl_->perf_stats.gpu_time_ms += elapsed_ms;
    impl_->perf_stats.problems_processed += 1;
    impl_->perf_stats.gpu_used = impl_->available;
    impl_->perf_stats.hybrid_used = true;
    // No CPU baseline is measured in this API path; keep speedup neutral and finite.
    impl_->perf_stats.speedup = kNeutralSpeedup;
    impl_->perf_stats.average_speedup = kNeutralSpeedup;
    updateMemoryStats(impl_->memory_stats, point_elements, total_elements);
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void>
CUDADistanceMatrix::computeBatch(const std::vector<core::BufferView<const double>> &points_batch,
                                 std::vector<core::BufferView<double>> &distances_batch,
                                 Size point_dim, double max_radius)
{
    if (points_batch.size() != distances_batch.size())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E51_PH_INPUT,
                                                "Batch input/output size mismatch");
    }
    for (Size i = 0; i < points_batch.size(); ++i)
    {
        auto result = compute(points_batch[i], distances_batch[i], point_dim, max_radius);
        if (result.isError())
        {
            return result;
        }
    }
    return errors::ErrorResult<void>::ok();
}

const AcceleratedPerformanceStats &CUDADistanceMatrix::getPerformanceStats() const
{
    return impl_->perf_stats;
}

errors::ErrorResult<MemoryUsageStats> CUDADistanceMatrix::getMemoryUsage() const
{
    MemoryUsageStats snapshot = impl_->memory_stats;
    return errors::ErrorResult<MemoryUsageStats>::ok(std::move(snapshot));
}

bool CUDADistanceMatrix::isAvailable() const
{
    return impl_->available;
}

errors::ErrorResult<DeviceInfo> CUDADistanceMatrix::getDeviceInfo() const
{
    if (!impl_->available)
    {
        return errors::ErrorResult<DeviceInfo>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                      "No CUDA device available");
    }

    int device_id = 0;
    cudaError_t status = cudaGetDevice(&device_id);
    if (status != cudaSuccess)
    {
        return errors::ErrorResult<DeviceInfo>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                      "Failed to query current CUDA device");
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device_id);
    if (status != cudaSuccess)
    {
        return errors::ErrorResult<DeviceInfo>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                      "Failed to query CUDA device properties");
    }

    DeviceInfo info;
    info.name = prop.name;
    info.device_id = device_id;
    info.total_memory = static_cast<Size>(prop.totalGlobalMem);

    size_t free_memory = 0;
    size_t total_memory = 0;
    status = cudaMemGetInfo(&free_memory, &total_memory);
    if (status != cudaSuccess)
    {
        return errors::ErrorResult<DeviceInfo>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL,
                                                      "Failed to query CUDA memory info");
    }
    if (total_memory != 0)
    {
        info.total_memory = static_cast<Size>(total_memory);
    }
    info.available_memory = static_cast<Size>(free_memory);
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.compute_capability =
        static_cast<double>(prop.major) + (0.1 * static_cast<double>(prop.minor));
    info.multiprocessor_count = prop.multiProcessorCount;
    return errors::ErrorResult<DeviceInfo>::ok(std::move(info));
}

} // namespace nerve::persistence::accelerated
