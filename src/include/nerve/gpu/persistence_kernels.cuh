#pragma once

/// @file persistence_kernels.cuh

#include "nerve/gpu/distance_kernels.hpp"
#include "nerve/gpu/persistence_memory_pool.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nerve::gpu::kernels
{

constexpr int kMinComputeCapability = 80; // Ampere.

enum class ArchFamily
{
    Ampere,
    Hopper,
    Blackwell,
    Unknown
};

class GpuCapabilities
{
public:
    static GpuCapabilities detect(int device_id = 0) noexcept
    {
        GpuCapabilities gpu;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        {
            return gpu;
        }

        gpu.device_id_ = std::clamp(device_id, 0, count - 1);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, gpu.device_id_) != cudaSuccess)
        {
            gpu.device_id_ = -1;
            return gpu;
        }

        gpu.cc_ = prop.major * 10 + prop.minor;
        gpu.sm_count_ = prop.multiProcessorCount;
        gpu.shared_mem_per_block_ = prop.sharedMemPerBlock;
        gpu.warp_size_ = prop.warpSize;
        gpu.name_ = prop.name;
        switch (prop.major)
        {
            case 8:
                gpu.family_ = ArchFamily::Ampere;
                break;
            case 9:
                gpu.family_ = ArchFamily::Hopper;
                break;
            case 10:
                gpu.family_ = ArchFamily::Blackwell;
                break;
            default:
                gpu.family_ = ArchFamily::Unknown;
                break;
        }
        return gpu;
    }

    [[nodiscard]] int deviceId() const noexcept { return device_id_; }
    [[nodiscard]] int computeCapability() const noexcept { return cc_; }
    [[nodiscard]] int smCount() const noexcept { return sm_count_; }
    [[nodiscard]] std::size_t sharedMemPerBlock() const noexcept { return shared_mem_per_block_; }
    [[nodiscard]] int warpSize() const noexcept { return warp_size_; }
    [[nodiscard]] ArchFamily family() const noexcept { return family_; }
    [[nodiscard]] bool isAvailable() const noexcept { return device_id_ >= 0; }
    [[nodiscard]] bool isSupported() const noexcept { return cc_ >= kMinComputeCapability; }
    [[nodiscard]] bool supportsTMA() const noexcept { return cc_ >= 90; }
    [[nodiscard]] bool supportsFP8Hardware() const noexcept { return cc_ >= 90; }
    [[nodiscard]] bool supportsFP4Hardware() const noexcept { return cc_ >= 100; }
    [[nodiscard]] bool supportsClusters() const noexcept { return cc_ >= 90; }
    [[nodiscard]] bool supportsDPX() const noexcept { return cc_ >= 90; }

    [[nodiscard]] bool supportsCUTLASS() const noexcept
    {
#ifdef NERVE_HAS_CUTLASS
        return cc_ >= kMinComputeCapability;
#else
        return false;
#endif
    }

    [[nodiscard]] bool supportsNVLinkPeerAccess(int other_device = 1) const noexcept
    {
        (void)other_device;
        return false;
    }

    [[nodiscard]] std::string getName() const
    {
        if (!name_.empty())
        {
            return name_;
        }
        switch (family_)
        {
            case ArchFamily::Ampere:
                return "Ampere";
            case ArchFamily::Hopper:
                return "Hopper";
            case ArchFamily::Blackwell:
                return "Blackwell";
            default:
                return "Unknown";
        }
    }

private:
    int device_id_ = -1;
    int cc_ = 0;
    int sm_count_ = 0;
    std::size_t shared_mem_per_block_ = 0;
    int warp_size_ = 32;
    ArchFamily family_ = ArchFamily::Unknown;
    std::string name_;
};

namespace detail
{

[[nodiscard]] inline bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t *out)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

[[nodiscard]] inline int pivotOfColumn(const std::vector<std::uint64_t> &columns, int column,
                                       int words_per_col)
{
    const std::size_t base = static_cast<std::size_t>(column) * words_per_col;
    for (int word = words_per_col - 1; word >= 0; --word)
    {
        const std::uint64_t value = columns[base + static_cast<std::size_t>(word)];
        if (value == 0)
        {
            continue;
        }
        for (int bit = 63; bit >= 0; --bit)
        {
            if ((value & (std::uint64_t{1} << bit)) != 0)
            {
                return word * 64 + bit;
            }
        }
    }
    return -1;
}

inline void xorColumn(std::vector<std::uint64_t> *columns, int dst, int src, int words_per_col)
{
    const std::size_t dst_base = static_cast<std::size_t>(dst) * words_per_col;
    const std::size_t src_base = static_cast<std::size_t>(src) * words_per_col;
    for (int word = 0; word < words_per_col; ++word)
    {
        (*columns)[dst_base + static_cast<std::size_t>(word)] ^=
            (*columns)[src_base + static_cast<std::size_t>(word)];
    }
}

} // namespace detail

class KernelDispatcher
{
public:
    explicit KernelDispatcher(int device_id = 0)
        : gpu_(GpuCapabilities::detect(device_id))
    {}

    template <typename T>
    [[nodiscard]] cudaError_t computeDistanceMatrix(const T *points, T *distances,
                                                    std::uint32_t n_points, std::uint32_t point_dim,
                                                    T max_radius, cudaStream_t stream = nullptr)
    {
        if (!std::isfinite(max_radius))
        {
            last_error_ = "max_radius must be finite";
            return cudaErrorInvalidValue;
        }
        if (n_points == 0)
        {
            return cudaSuccess;
        }
        if (point_dim == 0)
        {
            last_error_ = "point_dim must be positive when n_points is non-zero";
            return cudaErrorInvalidValue;
        }
        if (points == nullptr || distances == nullptr)
        {
            last_error_ = "points and distances must be non-null";
            return cudaErrorInvalidValue;
        }
        if (!gpu_.isSupported())
        {
            last_error_ = "CUDA device is missing or below sm80";
            return cudaErrorInvalidDevice;
        }

        if constexpr (std::is_same_v<T, float>)
        {
            return launch_pairwise_distance_radius_f32(
                points, static_cast<int>(point_dim), distances, static_cast<int>(n_points),
                static_cast<int>(n_points), static_cast<int>(point_dim), max_radius, stream);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return launch_pairwise_distance_radius_f64(
                points, static_cast<int>(point_dim), distances, static_cast<int>(n_points),
                static_cast<int>(n_points), static_cast<int>(point_dim), max_radius, stream);
        }
        else
        {
            last_error_ = "only float and double distance matrices are supported";
            return cudaErrorInvalidValue;
        }
    }

    [[nodiscard]] cudaError_t computeMatrixReduction(const std::uint64_t *boundary,
                                                     std::uint64_t *reduced, int n_cols,
                                                     int words_per_col, int *pivot_table,
                                                     cudaStream_t stream = nullptr)
    {
        if (n_cols < 0 || words_per_col < 0)
        {
            last_error_ = "invalid reduction buffers or dimensions";
            return cudaErrorInvalidValue;
        }
        if (n_cols == 0 || words_per_col == 0)
        {
            return cudaSuccess;
        }
        if (boundary == nullptr || reduced == nullptr)
        {
            last_error_ = "invalid reduction buffers";
            return cudaErrorInvalidValue;
        }

        std::size_t total_words = 0;
        if (!detail::checkedProduct(static_cast<std::size_t>(n_cols),
                                    static_cast<std::size_t>(words_per_col), &total_words))
        {
            last_error_ = "reduction matrix size overflows size_t";
            return cudaErrorInvalidValue;
        }
        std::size_t total_bytes = 0;
        if (!detail::checkedProduct(total_words, sizeof(std::uint64_t), &total_bytes))
        {
            last_error_ = "reduction matrix byte size overflows size_t";
            return cudaErrorInvalidValue;
        }

        std::vector<std::uint64_t> columns(total_words);
        cudaError_t status =
            cudaMemcpyAsync(columns.data(), boundary, total_bytes, cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess)
        {
            last_error_ = cudaGetErrorString(status);
            return status;
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
        {
            last_error_ = cudaGetErrorString(status);
            return status;
        }

        std::vector<int> pivots(static_cast<std::size_t>(n_cols), -1);
        std::unordered_map<int, int> pivot_to_column;
        pivot_to_column.reserve(static_cast<std::size_t>(n_cols));
        for (int col = 0; col < n_cols; ++col)
        {
            int pivot = detail::pivotOfColumn(columns, col, words_per_col);
            while (pivot >= 0)
            {
                const auto it = pivot_to_column.find(pivot);
                if (it == pivot_to_column.end())
                {
                    pivot_to_column.emplace(pivot, col);
                    break;
                }
                detail::xorColumn(&columns, col, it->second, words_per_col);
                pivot = detail::pivotOfColumn(columns, col, words_per_col);
            }
            pivots[static_cast<std::size_t>(col)] = pivot;
        }

        status =
            cudaMemcpyAsync(reduced, columns.data(), total_bytes, cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess)
        {
            last_error_ = cudaGetErrorString(status);
            return status;
        }
        if (pivot_table != nullptr)
        {
            std::size_t pivot_bytes = 0;
            if (!detail::checkedProduct(pivots.size(), sizeof(int), &pivot_bytes))
            {
                last_error_ = "pivot table byte size overflows size_t";
                return cudaErrorInvalidValue;
            }
            status = cudaMemcpyAsync(pivot_table, pivots.data(), pivot_bytes,
                                     cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess)
            {
                last_error_ = cudaGetErrorString(status);
                return status;
            }
        }
        return cudaStreamSynchronize(stream);
    }

    [[nodiscard]] std::string getLastError() const { return last_error_; }
    [[nodiscard]] const GpuCapabilities &capabilities() const noexcept { return gpu_; }

private:
    GpuCapabilities gpu_;
    std::string last_error_;
};

class NvLinkConfig
{
public:
    static NvLinkConfig detect()
    {
        NvLinkConfig config;
        if (cudaGetDeviceCount(&config.num_gpus_) != cudaSuccess || config.num_gpus_ <= 0)
        {
            config.num_gpus_ = 0;
            return config;
        }
        config.p2p_matrix_.assign(
            static_cast<std::size_t>(config.num_gpus_),
            std::vector<bool>(static_cast<std::size_t>(config.num_gpus_), false));
        for (int src = 0; src < config.num_gpus_; ++src)
        {
            for (int dst = 0; dst < config.num_gpus_; ++dst)
            {
                if (src == dst)
                {
                    config
                        .p2p_matrix_[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] =
                        true;
                    continue;
                }
                int can_access = 0;
                if (cudaDeviceCanAccessPeer(&can_access, src, dst) == cudaSuccess)
                {
                    config
                        .p2p_matrix_[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] =
                        can_access != 0;
                }
            }
        }
        return config;
    }

    [[nodiscard]] bool canP2P(int src_gpu, int dst_gpu) const
    {
        if (src_gpu < 0 || dst_gpu < 0 || src_gpu >= num_gpus_ || dst_gpu >= num_gpus_)
        {
            return false;
        }
        return p2p_matrix_[static_cast<std::size_t>(src_gpu)][static_cast<std::size_t>(dst_gpu)];
    }

    [[nodiscard]] float bandwidth(int src_gpu, int dst_gpu) const
    {
        (void)src_gpu;
        (void)dst_gpu;
        return 0.0f;
    }

    void enableP2P()
    {
        int previous_device = 0;
        const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
        for (int src = 0; src < num_gpus_; ++src)
        {
            if (cudaSetDevice(src) != cudaSuccess)
            {
                continue;
            }
            for (int dst = 0; dst < num_gpus_; ++dst)
            {
                if (src == dst || !canP2P(src, dst))
                {
                    continue;
                }
                const cudaError_t status = cudaDeviceEnablePeerAccess(dst, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled)
                {
                    (void)cudaGetLastError();
                }
            }
        }
        if (restore_device)
        {
            (void)cudaSetDevice(previous_device);
        }
    }

    [[nodiscard]] int numGPUs() const noexcept { return num_gpus_; }

private:
    int num_gpus_ = 0;
    std::vector<std::vector<bool>> p2p_matrix_;
};

struct TuningParams
{
    int blockSize = 256;
    int clusterSize = 1;
    int tileSize = 64;
    int numStages = 3;
    bool useTMA = false;
    bool useCUTLASS = false;

    static TuningParams load(int gpu_cc, const std::string &problem_type)
    {
        const auto key = std::make_pair(gpu_cc, problem_type);
        const auto &cache = processCache();
        const auto it = cache.find(key);
        if (it != cache.end())
        {
            return it->second;
        }

        TuningParams params;
        params.blockSize = gpu_cc >= 100 ? 512 : 256;
        params.clusterSize = gpu_cc >= 90 ? 4 : 1;
        params.tileSize = gpu_cc >= 90 ? 64 : 32;
        params.numStages = gpu_cc >= 90 ? 4 : 2;
        params.useTMA = gpu_cc >= 90;
        params.useCUTLASS =
#ifdef NERVE_HAS_CUTLASS
            gpu_cc >= kMinComputeCapability;
#else
            false;
#endif
        return params;
    }

    void save(int gpu_cc, const std::string &problem_type) const
    {
        processCache()[std::make_pair(gpu_cc, problem_type)] = *this;
    }

private:
    static std::map<std::pair<int, std::string>, TuningParams> &processCache()
    {
        static std::map<std::pair<int, std::string>, TuningParams> cache;
        return cache;
    }
};

} // namespace nerve::gpu::kernels
