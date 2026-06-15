#include "nerve/gpu/gpu_error.hpp"
#include "nerve/streaming/gpu_streaming.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

namespace nerve
{
namespace streaming
{
namespace gpu
{

constexpr int WINDOWED_BLOCK_SIZE = 256;
constexpr int WINDOWED_PAIR_STRIDE = 2;

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

__global__ __launch_bounds__(WINDOWED_BLOCK_SIZE) void windowSlideKernel(
    const float *__restrict__ window_old, const float *__restrict__ data_new,
    float *__restrict__ window_new, int window_size, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < window_size)
    {
        if (idx >= step)
        {
            window_new[idx - step] = window_old[idx];
        }
    }

    if (idx < step)
    {
        window_new[window_size - step + idx] = data_new[idx];
    }
}

__global__ __launch_bounds__(WINDOWED_BLOCK_SIZE) void windowPersistenceKernel(
    const float *__restrict__ window_data, float *__restrict__ persistence_pairs,
    int *__restrict__ birth_vertex, int *__restrict__ parent, int *__restrict__ rank,
    int window_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < window_size)
    {
        parent[idx] = idx;
        rank[idx] = 0;
        birth_vertex[idx] = idx;
    }
    __syncthreads();

    cg::thread_block block = cg::this_thread_block();
    (void)block;

    for (int i = 0; i < window_size; ++i)
    {
        for (int j = i + 1 + idx; j < window_size; j += blockDim.x * gridDim.x)
        {
            float dist = fabsf(window_data[i] - window_data[j]);

            int root_i = i;
            int root_j = j;

            while (parent[root_i] != root_i)
            {
                parent[root_i] = parent[parent[root_i]];
                root_i = parent[root_i];
            }
            while (parent[root_j] != root_j)
            {
                parent[root_j] = parent[parent[root_j]];
                root_j = parent[root_j];
            }

            if (root_i != root_j)
            {
                if (rank[root_i] < rank[root_j])
                {
                    int tmp = root_i;
                    root_i = root_j;
                    root_j = tmp;
                }

                parent[root_j] = root_i;
                if (rank[root_i] == rank[root_j])
                {
                    rank[root_i]++;
                }

                int pair_idx = atomicAdd(&birth_vertex[window_size], 1);
                if (pair_idx < window_size * window_size)
                {
                    float birth = window_data[birth_vertex[root_j]];
                    persistence_pairs[pair_idx * WINDOWED_PAIR_STRIDE] = birth;
                    persistence_pairs[pair_idx * WINDOWED_PAIR_STRIDE + 1] = dist;
                }

                birth_vertex[root_i] =
                    (window_data[birth_vertex[root_i]] < window_data[birth_vertex[root_j]])
                        ? birth_vertex[root_i]
                        : birth_vertex[root_j];
            }
        }
        __syncthreads();
    }
}

namespace
{

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

template <typename T>
std::size_t checkedBytes(std::size_t count, const char *label)
{
    std::size_t bytes = 0;
    if (!checkedProduct(count, sizeof(T), bytes))
    {
        throw std::length_error(label);
    }
    return bytes;
}

int checkedGridBlocks(std::size_t values, const char *label)
{
    const std::size_t blocks =
        (values / static_cast<std::size_t>(WINDOWED_BLOCK_SIZE)) +
        ((values % static_cast<std::size_t>(WINDOWED_BLOCK_SIZE)) != 0 ? 1U : 0U);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(blocks);
}

std::size_t checkedPairValueCount(int window_size)
{
    std::size_t pair_count = 0;
    if (!checkedProduct(static_cast<std::size_t>(window_size),
                        static_cast<std::size_t>(window_size), pair_count) ||
        pair_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("windowed persistence pair count exceeds CUDA limits");
    }
    std::size_t pair_values = 0;
    if (!checkedProduct(pair_count, static_cast<std::size_t>(WINDOWED_PAIR_STRIDE), pair_values))
    {
        throw std::length_error("windowed persistence pair buffer size overflows");
    }
    return pair_values;
}

bool valuesAreFinite(const std::vector<float> &values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

} // namespace

class GPUWindowedPersistence
{
public:
    GPUWindowedPersistence(int window_size, int max_dim = 1)
        : window_size_(window_size)
        , max_dim_(max_dim)
        , stream1_(nullptr)
    {
        if (window_size_ <= 0 || max_dim_ < 0)
        {
            throw std::invalid_argument("window_size must be positive and max_dim non-negative");
        }
        const auto window_values = static_cast<std::size_t>(window_size_);
        const auto pair_values = checkedPairValueCount(window_size_);
        try
        {
            GPU_CHECK(cudaMalloc(&d_window_, checkedBytes<float>(window_values, "window buffer")));
            GPU_CHECK(
                cudaMalloc(&d_pairs_, checkedBytes<float>(pair_values, "window pair buffer")));
            GPU_CHECK(cudaStreamCreate(&stream1_));
        }
        catch (...)
        {
            releaseDeviceState();
            throw;
        }
    }

    ~GPUWindowedPersistence() { releaseDeviceState(); }

    void updateWindow(const std::vector<float> &new_data)
    {
        if (new_data.size() != static_cast<size_t>(window_size_))
        {
            throw std::invalid_argument("new_data size must equal window_size");
        }
        if (!valuesAreFinite(new_data))
        {
            throw std::invalid_argument("new window data must be finite");
        }
        float *d_new = nullptr;
        try
        {
            GPU_CHECK(cudaMalloc(&d_new, checkedBytes<float>(new_data.size(), "new window data")));
            GPU_CHECK(cudaMemcpyAsync(d_new, new_data.data(),
                                      checkedBytes<float>(new_data.size(), "new window data"),
                                      cudaMemcpyHostToDevice, stream1_));
            GPU_CHECK(cudaStreamSynchronize(stream1_));
            std::swap(d_window_, d_new);
            window_loaded_ = true;
            cudaFree(d_new);
        }
        catch (...)
        {
            cudaFree(d_new);
            throw;
        }
    }

    std::vector<std::pair<float, float>> computePersistence()
    {
        if (!window_loaded_)
        {
            throw std::logic_error("window data must be loaded before computePersistence");
        }
        int *d_parent = nullptr;
        int *d_rank = nullptr;
        int *d_birth_vertex = nullptr;
        try
        {
            const auto window_values = static_cast<std::size_t>(window_size_);
            const auto pair_values = checkedPairValueCount(window_size_);
            GPU_CHECK(cudaMalloc(&d_parent, checkedBytes<int>(window_values, "window parent")));
            GPU_CHECK(cudaMalloc(&d_rank, checkedBytes<int>(window_values, "window rank")));
            GPU_CHECK(cudaMalloc(&d_birth_vertex,
                                 checkedBytes<int>(window_values + 1U, "window birth vertices")));
            GPU_CHECK(cudaMemsetAsync(d_birth_vertex + window_size_, 0, sizeof(int), stream1_));

            const int grid_size = checkedGridBlocks(window_values, "window persistence grid");
            windowPersistenceKernel<<<grid_size, WINDOWED_BLOCK_SIZE, 0, stream1_>>>(
                d_window_, d_pairs_, d_birth_vertex, d_parent, d_rank, window_size_);
            GPU_CHECK(cudaPeekAtLastError());

            int h_pair_count = 0;
            GPU_CHECK(cudaMemcpyAsync(&h_pair_count, &d_birth_vertex[window_size_], sizeof(int),
                                      cudaMemcpyDeviceToHost, stream1_));

            std::vector<float> pairs(pair_values);
            GPU_CHECK(cudaMemcpyAsync(pairs.data(), d_pairs_,
                                      checkedBytes<float>(pairs.size(), "window pairs"),
                                      cudaMemcpyDeviceToHost, stream1_));

            GPU_CHECK(cudaStreamSynchronize(stream1_));
            const auto max_pair_count = static_cast<int>(pair_values / WINDOWED_PAIR_STRIDE);
            if (h_pair_count < 0 || h_pair_count > max_pair_count)
            {
                throw std::runtime_error("windowed persistence returned invalid pair count");
            }

            std::vector<std::pair<float, float>> result;
            result.reserve(static_cast<std::size_t>(h_pair_count));
            for (int i = 0; i < h_pair_count; ++i)
            {
                const auto offset = static_cast<std::size_t>(i) * WINDOWED_PAIR_STRIDE;
                float birth = pairs[offset];
                float death = pairs[offset + 1U];
                if (!std::isfinite(birth) || !std::isfinite(death) || death < birth)
                {
                    throw std::runtime_error("windowed persistence produced invalid pair");
                }
                if (death > birth)
                {
                    result.push_back({birth, death});
                }
            }

            cudaFree(d_parent);
            cudaFree(d_rank);
            cudaFree(d_birth_vertex);
            return result;
        }
        catch (...)
        {
            cudaFree(d_parent);
            cudaFree(d_rank);
            cudaFree(d_birth_vertex);
            throw;
        }
    }

    void synchronize() { GPU_CHECK(cudaStreamSynchronize(stream1_)); }

private:
    void releaseDeviceState() noexcept
    {
        cudaFree(d_window_);
        cudaFree(d_pairs_);
        if (stream1_ != nullptr)
        {
            cudaStreamDestroy(stream1_);
        }
        d_window_ = nullptr;
        d_pairs_ = nullptr;
        stream1_ = nullptr;
    }

    int window_size_;
    int max_dim_;
    bool window_loaded_ = false;

    float *d_window_ = nullptr;
    float *d_pairs_ = nullptr;

    cudaStream_t stream1_;
};

class UnifiedMemoryStreaming
{
public:
    explicit UnifiedMemoryStreaming(size_t buffer_size)
        : buffer_size_(buffer_size)
    {
        if (buffer_size_ == 0)
        {
            throw std::invalid_argument("managed streaming buffer size must be positive");
        }
        GPU_CHECK(
            cudaMallocManaged(&buffer_, checkedBytes<float>(buffer_size, "managed stream buffer")));
    }

    ~UnifiedMemoryStreaming()
    {
        if (buffer_)
            cudaFree(buffer_);
    }

    float *getBuffer() { return buffer_; }

    void prefetchToGPU()
    {
        cudaMemLocation location{};
        location.type = cudaMemLocationTypeDevice;
        location.id = 0;
        GPU_CHECK(cudaMemPrefetchAsync(
            buffer_, checkedBytes<float>(buffer_size_, "managed stream buffer"), location, 0U));
    }

    void prefetchToCPU()
    {
        cudaMemLocation location{};
        location.type = cudaMemLocationTypeHost;
        location.id = 0;
        GPU_CHECK(cudaMemPrefetchAsync(
            buffer_, checkedBytes<float>(buffer_size_, "managed stream buffer"), location, 0U));
    }

private:
    float *buffer_ = nullptr;
    size_t buffer_size_;
};

WindowedBenchmark benchmarkWindowed(int window_size, int num_windows)
{
    if (window_size <= 0 || num_windows < 0)
    {
        throw std::invalid_argument("window_size must be positive and num_windows non-negative");
    }

    WindowedBenchmark bench;
    bench.window_size = window_size;
    bench.num_windows = num_windows;

    std::size_t total_data = 0;
    if (!checkedProduct(static_cast<std::size_t>(window_size) +
                            static_cast<std::size_t>(num_windows),
                        1U, total_data) ||
        total_data > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("windowed benchmark data size exceeds int range");
    }
    std::vector<float> data;
    data.reserve(total_data);
    for (int i = 0; i < static_cast<int>(total_data); ++i)
    {
        data.push_back(static_cast<float>(i));
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int w = 0; w < num_windows; ++w)
    {
        std::vector<float> window(data.begin() + w, data.begin() + w + window_size);

        std::vector<int> parent(window_size);
        std::iota(parent.begin(), parent.end(), 0);

        auto find = [&](int x, auto &&find_ref) -> int {
            if (parent[x] != x)
            {
                parent[x] = find_ref(parent[x], find_ref);
            }
            return parent[x];
        };

        for (int i = 0; i < window_size; ++i)
        {
            for (int j = i + 1; j < window_size; ++j)
            {
                float dist = std::abs(window[i] - window[j]);
                if (dist < 1.0f)
                {
                    int root_i = find(i, find);
                    int root_j = find(j, find);
                    if (root_i != root_j)
                    {
                        parent[root_i] = root_j;
                    }
                }
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    GPUWindowedPersistence gpu(window_size);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int w = 0; w < num_windows; ++w)
    {
        std::vector<float> window(data.begin() + w, data.begin() + w + window_size);
        gpu.updateWindow(window);
        gpu.computePersistence();
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.cpu_time_ms, bench.gpu_time_ms);

    return bench;
}

} // namespace gpu
} // namespace streaming
} // namespace nerve
