
#include "nerve/persistence/accelerated/accelerated_api.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace nerve::persistence::accelerated::exception_safety
{

template <typename CleanupFunc>
class ExceptionGuard
{
public:
    explicit ExceptionGuard(CleanupFunc cleanup_func)
        : cleanup_func_(std::move(cleanup_func))
        , active_(true)
    {}

    ~ExceptionGuard()
    {
        if (!active_)
        {
            return;
        }
        try
        {
            cleanup_func_();
        }
        catch (...)
        {}
    }

    ExceptionGuard(const ExceptionGuard &) = delete;
    ExceptionGuard &operator=(const ExceptionGuard &) = delete;

    ExceptionGuard(ExceptionGuard &&other) noexcept
        : cleanup_func_(std::move(other.cleanup_func_))
        , active_(other.active_)
    {
        other.active_ = false;
    }

    ExceptionGuard &operator=(ExceptionGuard &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        if (active_)
        {
            try
            {
                cleanup_func_();
            }
            catch (...)
            {}
        }
        cleanup_func_ = std::move(other.cleanup_func_);
        active_ = other.active_;
        other.active_ = false;
        return *this;
    }

    void release() noexcept { active_ = false; }

private:
    CleanupFunc cleanup_func_;
    bool active_;
};

class SafeMutexGuard
{
public:
    explicit SafeMutexGuard(std::mutex &mutex)
        : lock_(mutex)
    {}

private:
    std::unique_lock<std::mutex> lock_;
};

class AtomicCounter
{
public:
    explicit AtomicCounter(size_t initial = 0)
        : value_(initial)
    {}

    size_t increment() noexcept { return value_.fetch_add(1, std::memory_order_acq_rel) + 1; }

    size_t decrement() noexcept { return value_.fetch_sub(1, std::memory_order_acq_rel) - 1; }

    size_t get() const noexcept { return value_.load(std::memory_order_acquire); }

private:
    std::atomic<size_t> value_;
};

errors::ErrorResult<std::unique_ptr<void, std::function<void(void *)>>>
allocate_cuda_guarded(size_t bytes)
{
    void *ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, bytes);
    if (status != cudaSuccess || ptr == nullptr)
    {
        return errors::ErrorResult<std::unique_ptr<void, std::function<void(void *)>>>::error(
            errors::ErrorCode::E10_GPU_OOM);
    }

    auto deleter = [](void *raw_ptr) {
        if (raw_ptr != nullptr)
        {
            cudaFree(raw_ptr);
        }
    };
    return errors::ErrorResult<std::unique_ptr<void, std::function<void(void *)>>>::success(
        std::unique_ptr<void, std::function<void(void *)>>(ptr, std::move(deleter)));
}

} // namespace nerve::persistence::accelerated::exception_safety
