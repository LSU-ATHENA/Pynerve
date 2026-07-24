
#pragma once
#include "nerve/core/memory/memory_pool.hpp"
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(CUDA_AVAILABLE) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA 1
#else
#define NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA 0
#endif

namespace nerve::core
{

template <typename T, Size SlabCapacity = 256>
class ThreadLocalPool
{
    static_assert(SlabCapacity > 0, "SlabCapacity must be positive");
    static_assert(SlabCapacity <= std::numeric_limits<Size>::max() / sizeof(T),
                  "Slab storage size overflow");

public:
    static ThreadLocalPool &instance()
    {
        thread_local ThreadLocalPool pool;
        return pool;
    }

    errors::ErrorResult<T *> allocate()
    {
        if (free_list_.empty())
        {
            auto result = addSlab();
            if (!result.isSuccess())
            {
                return errors::ErrorResult<T *>::error(result.errorCode());
            }
        }

        T *ptr = free_list_.back();
        free_list_.pop_back();

#ifndef NDEBUG
        live_allocations_.insert(ptr);
#endif

        return errors::ErrorResult<T *>::success(std::move(ptr));
    }

    errors::ErrorResult<void> deallocate(T *ptr)
    {
        if (!ptr)
            return errors::ErrorResult<void>::success();

#ifndef NDEBUG
        if (live_allocations_.find(ptr) == live_allocations_.end())
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E74_RACE_CONDITION);
        }
        live_allocations_.erase(ptr);
#endif

        free_list_.push_back(ptr);
        return errors::ErrorResult<void>::success();
    }

    struct Deleter
    {
        void operator()(T *p) noexcept
        {
            if (p)
            {
                p->~T();
            }
            ThreadLocalPool::instance().deallocate(p);
        }
    };
    using UniquePtr = std::unique_ptr<T, Deleter>;

    static UniquePtr makeUnique()
    {
        auto result = instance().allocate();
        if (!result.isSuccess())
        {
            throw std::runtime_error("Thread-local pool allocation failed");
        }

        T *ptr = result.value();
        new (ptr) T();
        return UniquePtr(ptr);
    }

    Size capacity() const noexcept { return slabs_.size() * SlabCapacity; }
    Size available() const noexcept { return free_list_.size(); }
    Size allocated() const noexcept { return capacity() - available(); }

private:
    struct Slab
    {
        alignas(T) std::byte storage[SlabCapacity * sizeof(T)];

        T *data() { return reinterpret_cast<T *>(storage); }
        T &operator[](Size i) { return data()[i]; }
    };

    std::vector<std::unique_ptr<Slab>> slabs_;
    std::vector<T *> free_list_;

#ifndef NDEBUG
    std::unordered_set<T *> live_allocations_;
#endif

    errors::ErrorResult<void> addSlab()
    {
        auto &slab = slabs_.emplace_back(std::make_unique<Slab>());
        for (Size i = 0; i < SlabCapacity; ++i)
        {
            free_list_.push_back(&(*slab)[i]);
        }
        return errors::ErrorResult<void>::success();
    }
};

class EnhancedThreadPool
{
public:
    explicit EnhancedThreadPool(Size numThreads = 0);
    ~EnhancedThreadPool();

    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>;

    void waitForAll();
    Size numThreads() const noexcept;

    template <typename Task>
    class ThreadSafeQueue
    {
    public:
        void push(Task task);
        bool pop(Task &task);
        bool empty() const;
        Size size() const;
        void shutdown();

    private:
        std::deque<Task> queue_;
        mutable std::mutex mutex_;
        std::condition_variable condition_;
        bool shutdown_ = false;
    };

private:
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<Size> pending_tasks_{0};
    std::mutex wait_mutex_;
    std::condition_variable wait_condition_;
    ThreadSafeQueue<std::function<void()>> task_queue_;

    void workerThread();
};

template <typename Task>
void EnhancedThreadPool::ThreadSafeQueue<Task>::push(Task task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_)
        {
            throw std::runtime_error("Cannot push to stopped EnhancedThreadPool queue");
        }
        queue_.push_back(std::move(task));
    }
    condition_.notify_one();
}

template <typename Task>
bool EnhancedThreadPool::ThreadSafeQueue<Task>::pop(Task &task)
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
    if (queue_.empty())
    {
        return false;
    }
    task = std::move(queue_.front());
    queue_.pop_front();
    return true;
}

template <typename Task>
bool EnhancedThreadPool::ThreadSafeQueue<Task>::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

template <typename Task>
Size EnhancedThreadPool::ThreadSafeQueue<Task>::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

template <typename Task>
void EnhancedThreadPool::ThreadSafeQueue<Task>::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    condition_.notify_all();
}

inline EnhancedThreadPool::EnhancedThreadPool(Size numThreads)
{
    if (numThreads == 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }
    if (numThreads == 0)
    {
        numThreads = 1;
    }

    workers_.reserve(numThreads);
    for (Size i = 0; i < numThreads; ++i)
    {
        workers_.emplace_back([this] { workerThread(); });
    }
}

inline EnhancedThreadPool::~EnhancedThreadPool()
{
    stop_flag_.store(true, std::memory_order_release);
    task_queue_.shutdown();
    for (auto &worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

template <typename F, typename... Args>
auto EnhancedThreadPool::submit(F &&f,
                                Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>
{
    if (stop_flag_.load(std::memory_order_acquire))
    {
        throw std::runtime_error("Cannot submit to stopped EnhancedThreadPool");
    }

    using ReturnType = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto future = task->get_future();

    pending_tasks_.fetch_add(1, std::memory_order_acq_rel);
    try
    {
        task_queue_.push([this, task] {
            try
            {
                (*task)();
            }
            catch (...)
            {
                // Keep worker threads alive if packaged_task itself fails.
            }
            if (pending_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                wait_condition_.notify_all();
            }
        });
    }
    catch (...)
    {
        if (pending_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            wait_condition_.notify_all();
        }
        throw;
    }

    return future;
}

inline void EnhancedThreadPool::waitForAll()
{
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_condition_.wait(lock,
                         [this] { return pending_tasks_.load(std::memory_order_acquire) == 0; });
}

inline Size EnhancedThreadPool::numThreads() const noexcept
{
    return workers_.size();
}

inline void EnhancedThreadPool::workerThread()
{
    std::function<void()> task;
    while (task_queue_.pop(task))
    {
        if (task)
        {
            task();
        }
        task = nullptr;
    }
}

template <typename T>
class EnhancedDeviceBuffer
{
public:
    static errors::ErrorResult<EnhancedDeviceBuffer<T>> allocate(Size count)
    {
        EnhancedDeviceBuffer<T> buffer;
        buffer.count_ = count;
        if (count > std::numeric_limits<Size>::max() / sizeof(T))
        {
            return errors::ErrorResult<EnhancedDeviceBuffer<T>>::error(
                errors::ErrorCode::E10_GPU_OOM);
        }

#if NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA
        cudaError_t err = cudaMalloc(&buffer.ptr_, count * sizeof(T));
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<EnhancedDeviceBuffer<T>>::error(
                errors::ErrorCode::E10_GPU_OOM);
        }

        return errors::ErrorResult<EnhancedDeviceBuffer<T>>::success(std::move(buffer));
#else
        (void)count;
        return errors::ErrorResult<EnhancedDeviceBuffer<T>>::error(errors::ErrorCode::E10_GPU_OOM);
#endif
    }

    ~EnhancedDeviceBuffer()
    {
#if NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA
        if (ptr_)
        {
            cudaFree(ptr_);
        }
#endif
    }

    EnhancedDeviceBuffer(const EnhancedDeviceBuffer &) = delete;
    EnhancedDeviceBuffer &operator=(const EnhancedDeviceBuffer &) = delete;

    EnhancedDeviceBuffer(EnhancedDeviceBuffer &&other) noexcept
        : ptr_(other.ptr_)
        , count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    EnhancedDeviceBuffer &operator=(EnhancedDeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
#if NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA
            if (ptr_)
                cudaFree(ptr_);
#endif
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T *get() const noexcept { return ptr_; }
    Size size() const noexcept { return count_; }

    errors::ErrorResult<void> copyFromHost(const T *host, Size count)
    {
        if (count > count_)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E88_INVALID_SIMPLICES);
        }
        if (count > std::numeric_limits<Size>::max() / sizeof(T))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }

#if NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA
        cudaError_t err = cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        return errors::ErrorResult<void>::success();
#else
        (void)host;
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
#endif
    }

    errors::ErrorResult<void> copyToHost(T *host, Size count) const
    {
        if (count > count_)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E88_INVALID_SIMPLICES);
        }
        if (count > std::numeric_limits<Size>::max() / sizeof(T))
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E10_GPU_OOM);
        }
#if NERVE_CORE_THREAD_SAFE_MEMORY_POOL_HAS_CUDA
        cudaError_t err = cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
        }

        return errors::ErrorResult<void>::success();
#else
        (void)host;
        (void)count;
        return errors::ErrorResult<void>::error(errors::ErrorCode::E11_GPU_LAUNCH_FAIL);
#endif
    }

private:
    T *ptr_ = nullptr;
    Size count_ = 0;
    EnhancedDeviceBuffer() = default;
};

} // namespace nerve::core
