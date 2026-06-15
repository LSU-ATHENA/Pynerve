
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

namespace nerve::persistence::lockfree
{

namespace
{

size_t checkedPowerOfTwoCapacity(size_t requested_capacity)
{
    size_t capacity = 1;
    const size_t minimum_capacity = std::max<size_t>(requested_capacity, 2);
    while (capacity < minimum_capacity)
    {
        if (capacity > std::numeric_limits<size_t>::max() / 2)
        {
            throw std::length_error("lockfree capacity exceeds addressable range");
        }
        capacity *= 2;
    }
    return capacity;
}

} // namespace

LockFreeWorkQueue::LockFreeWorkQueue(size_t capacity)
    : capacity_(checkedPowerOfTwoCapacity(capacity))
{
    buffer_.resize(capacity_);
}

LockFreeWorkQueue::~LockFreeWorkQueue()
{
    for (auto &entry : buffer_)
    {
        auto *task_ptr = entry.task_ptr.exchange(nullptr, std::memory_order_acq_rel);
        delete task_ptr;
        entry.ready.store(false, std::memory_order_release);
    }
}

void LockFreeWorkQueue::push(std::function<void()> task)
{
    size_t b = bottom_.load(std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);

    if (b - t >= capacity_)
    {
        throw std::runtime_error("lockfree work queue capacity exhausted");
    }

    TaskEntry &entry = buffer_[modCapacity(b)];
    auto *previous = entry.task_ptr.exchange(new std::function<void()>(std::move(task)),
                                             std::memory_order_acq_rel);
    delete previous;
    entry.ready.store(true, std::memory_order_release);

    bottom_.store(b + 1, std::memory_order_release);
}

std::optional<std::function<void()>> LockFreeWorkQueue::pop()
{
    size_t b = bottom_.load(std::memory_order_relaxed);
    if (top_.load(std::memory_order_acquire) >= b)
    {
        return std::nullopt;
    }

    --b;
    bottom_.store(b, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);

    size_t t = top_.load(std::memory_order_relaxed);

    std::optional<std::function<void()>> result;

    if (t <= b)
    {
        size_t idx = modCapacity(b);
        TaskEntry &entry = buffer_[idx];

        if (entry.ready.load(std::memory_order_acquire))
        {
            if (t == b)
            {
                if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                                  std::memory_order_relaxed))
                {
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }

            auto *task_ptr = entry.task_ptr.exchange(nullptr, std::memory_order_acq_rel);
            entry.ready.store(false, std::memory_order_release);
            if (task_ptr == nullptr)
            {
                return std::nullopt;
            }
            result = std::move(*task_ptr);
            delete task_ptr;
        }
        else
        {
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
    }
    else
    {
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    return result;
}

std::optional<std::function<void()>> LockFreeWorkQueue::steal()
{
    size_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);

    if (t < b)
    {
        size_t idx = modCapacity(t);
        TaskEntry &entry = buffer_[idx];

        if (entry.ready.load(std::memory_order_acquire))
        {
            if (top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                             std::memory_order_relaxed))
            {
                auto *task_ptr = entry.task_ptr.exchange(nullptr, std::memory_order_acq_rel);
                entry.ready.store(false, std::memory_order_release);
                if (task_ptr == nullptr)
                {
                    return std::nullopt;
                }
                std::optional<std::function<void()>> result = std::move(*task_ptr);
                delete task_ptr;
                return result;
            }
        }
    }

    return std::nullopt;
}

size_t LockFreeWorkQueue::size() const
{
    const size_t b = bottom_.load(std::memory_order_relaxed);
    const size_t t = top_.load(std::memory_order_relaxed);
    return b > t ? b - t : 0;
}

bool LockFreeWorkQueue::isEmpty() const
{
    return size() == 0;
}

} // namespace nerve::persistence::lockfree
