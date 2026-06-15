
// Optimizations:
// - CUDA streams for concurrent execution
// - Async memory transfers (cudaMemcpyAsync)
// - Stream priorities
// - Graph capture for repeated sequences

#include "nerve/core.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace async
{

/**
 * @brief CUDA stream wrapper with priority support
 */
class CUDAStream
{
public:
    explicit CUDAStream(int priority = 0)
    {
        cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority);
    }

    ~CUDAStream()
    {
        if (stream_)
        {
            cudaStreamDestroy(stream_);
        }
    }

    // Disable copy
    CUDAStream(const CUDAStream &) = delete;
    CUDAStream &operator=(const CUDAStream &) = delete;

    // Enable move
    CUDAStream(CUDAStream &&other) noexcept
        : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    cudaStream_t get() const { return stream_; }

    void synchronize() const { cudaStreamSynchronize(stream_); }

    bool query() const { return cudaStreamQuery(stream_) == cudaSuccess; }

    void waitEvent(cudaEvent_t event) const { cudaStreamWaitEvent(stream_, event, 0); }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @brief CUDA event wrapper
 */
class CUDAEvent
{
public:
    CUDAEvent() { cudaEventCreate(&event_); }

    explicit CUDAEvent(unsigned int flags) { cudaEventCreateWithFlags(&event_, flags); }

    ~CUDAEvent()
    {
        if (event_)
        {
            cudaEventDestroy(event_);
        }
    }

    void record(cudaStream_t stream) { cudaEventRecord(event_, stream); }

    void synchronize() { cudaEventSynchronize(event_); }

    float elapsedTime(const CUDAEvent &start) const
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start.event_, event_);
        return ms;
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @brief Async memory transfer helper
 */
class AsyncMemoryTransfer
{
public:
    // Host to device async
    static void h2d(void *dst, const void *src, size_t size, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    }

    // Device to host async
    static void d2h(void *dst, const void *src, size_t size, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    }

    // Device to device async
    static void d2d(void *dst, const void *src, size_t size, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }

    // Pinned memory allocation for async transfers
    static void *allocatePinned(size_t size)
    {
        void *ptr;
        cudaMallocHost(&ptr, size);
        return ptr;
    }

    static void freePinned(void *ptr) { cudaFreeHost(ptr); }
};

/**
 * @brief CUDA Graph for repeated kernel sequences
 *
 * Captures a sequence of kernels for optimized replay
 */
class CUDAGraph
{
public:
    CUDAGraph()
        : graph_(nullptr)
        , instance_(nullptr)
    {}

    ~CUDAGraph()
    {
        if (instance_)
            cudaGraphExecDestroy(instance_);
        if (graph_)
            cudaGraphDestroy(graph_);
    }

    // Begin capture
    void beginCapture(cudaStream_t stream)
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }

    // End capture
    void endCapture(cudaStream_t stream)
    {
        cudaStreamEndCapture(stream, &graph_);
        cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    }

    // Launch captured graph
    void launch(cudaStream_t stream) const
    {
        if (instance_)
        {
            cudaGraphLaunch(instance_, stream);
        }
    }

    bool isValid() const { return instance_ != nullptr; }

private:
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
};

/**
 * @brief Async task executor
 *
 * Manages concurrent execution of GPU tasks
 */
class AsyncExecutor
{
public:
    explicit AsyncExecutor(int num_streams = 4)
    {
        // Create streams with different priorities
        for (int i = 0; i < num_streams; ++i)
        {
            int priority = (i == 0) ? -1 : 0; // First stream high priority
            streams_.emplace_back(priority);
        }
    }

    // Submit async task
    template <typename F>
    std::future<void> submit(F &&func)
    {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(func));
        std::future<void> result = task->get_future();

        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push([task]() { (*task)(); });
        cv_.notify_one();

        return result;
    }

    // Get next available stream (round-robin)
    cudaStream_t getStream()
    {
        int idx = next_stream_.fetch_add(1) % streams_.size();
        return streams_[idx].get();
    }

    // Synchronize all streams
    void synchronizeAll()
    {
        for (auto &stream : streams_)
        {
            stream.synchronize();
        }
    }

    // Get stream by index
    cudaStream_t getStream(int index) { return streams_[index % streams_.size()].get(); }

private:
    std::vector<CUDAStream> streams_;
    std::atomic<int> next_stream_{0};

    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
};

/**
 * @brief Async data loader with prefetching
 *
 * Prefetches next batch while GPU processes current batch
 */
template <typename T>
class AsyncDataLoader
{
public:
    AsyncDataLoader(size_t batch_size, int num_prefetch = 2)
        : batch_size_(batch_size)
        , num_prefetch_(num_prefetch)
    {
        // Allocate pinned memory for batches
        for (int i = 0; i < num_prefetch; ++i)
        {
            void *ptr = AsyncMemoryTransfer::allocatePinned(batch_size * sizeof(T));
            pinned_batches_.push_back(static_cast<T *>(ptr));
        }

        // Allocate GPU memory
        for (int i = 0; i < num_prefetch; ++i)
        {
            T *d_ptr;
            cudaMalloc(&d_ptr, batch_size * sizeof(T));
            gpu_batches_.push_back(d_ptr);
        }
    }

    ~AsyncDataLoader()
    {
        for (void *ptr : pinned_batches_)
        {
            AsyncMemoryTransfer::freePinned(ptr);
        }
        for (auto *ptr : gpu_batches_)
        {
            cudaFree(ptr);
        }
    }

    // Load next batch async
    void prefetchNext(const std::vector<T> &data, cudaStream_t stream)
    {
        int idx = current_prefetch_ % num_prefetch_;

        // Copy to pinned memory
        std::memcpy(pinned_batches_[idx], data.data(),
                    std::min(data.size(), batch_size_) * sizeof(T));

        // Async transfer to GPU
        AsyncMemoryTransfer::h2d(gpu_batches_[idx], pinned_batches_[idx], batch_size_ * sizeof(T),
                                 stream);

        current_prefetch_++;
    }

    // Get GPU pointer for current batch
    T *getCurrentGPUBuffer()
    {
        int idx = (current_prefetch_ - 1) % num_prefetch_;
        return gpu_batches_[idx];
    }

private:
    size_t batch_size_;
    int num_prefetch_;
    std::atomic<int> current_prefetch_{0};

    std::vector<T *> pinned_batches_;
    std::vector<T *> gpu_batches_;
};

/**
 * @brief Benchmark async execution
 */
struct AsyncBenchmark
{
    double sync_time_ms;
    double async_time_ms;
    double speedup;
    int num_operations;
};

template <typename KernelFunc>
AsyncBenchmark benchmarkAsync(KernelFunc kernel, int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument("async benchmark iterations must be positive");
    }

    AsyncBenchmark bench{};
    bench.num_operations = iterations;

    // Synchronous baseline
    auto start_sync = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        kernel(nullptr); // Use default stream
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            throw std::runtime_error("async benchmark synchronous kernel failed");
        }
    }
    auto end_sync = std::chrono::high_resolution_clock::now();
    bench.sync_time_ms = std::chrono::duration<double, std::milli>(end_sync - start_sync).count();

    // Async version
    std::vector<cudaStream_t> streams(4);
    for (int i = 0; i < 4; ++i)
    {
        if (cudaStreamCreate(&streams[i]) != cudaSuccess)
        {
            for (cudaStream_t stream : streams)
            {
                if (stream != nullptr)
                {
                    cudaStreamDestroy(stream);
                }
            }
            throw std::runtime_error("async benchmark stream creation failed");
        }
    }

    auto start_async = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        kernel(streams[i % 4]);
    }
    for (auto s : streams)
    {
        if (cudaStreamSynchronize(s) != cudaSuccess)
        {
            for (cudaStream_t stream : streams)
            {
                cudaStreamDestroy(stream);
            }
            throw std::runtime_error("async benchmark asynchronous kernel failed");
        }
    }
    auto end_async = std::chrono::high_resolution_clock::now();
    bench.async_time_ms =
        std::chrono::duration<double, std::milli>(end_async - start_async).count();

    for (auto s : streams)
    {
        cudaStreamDestroy(s);
    }

    bench.speedup = (std::isfinite(bench.sync_time_ms) && std::isfinite(bench.async_time_ms) &&
                     bench.sync_time_ms >= 0.0 && bench.async_time_ms > 0.0)
                        ? bench.sync_time_ms / bench.async_time_ms
                        : 1.0;

    return bench;
}

} // namespace async
} // namespace gpu
} // namespace nerve
