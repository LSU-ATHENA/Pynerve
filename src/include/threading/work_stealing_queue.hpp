
#pragma once
#include "nerve/error/error_registry.hpp"

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace nerve::threading
{

template <typename T>
concept TaskType = std::movable<T> && std::destructible<T>;

template <TaskType T>
class WorkStealingQueue
{
private:
    std::deque<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> size_{0};

public:
    using value_type = T;
    using size_type = size_t;

    WorkStealingQueue() = default;

    WorkStealingQueue(const WorkStealingQueue &) = delete;
    WorkStealingQueue &operator=(const WorkStealingQueue &) = delete;
    WorkStealingQueue(WorkStealingQueue &&) = delete;
    WorkStealingQueue &operator=(WorkStealingQueue &&) = delete;

    ~WorkStealingQueue() { shutdown(); }

    [[nodiscard]] error::Result<void> push(T task)
    {
        try
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (shutdown_.load(std::memory_order_acquire))
                {
                    return error::Result<void>::err(error::TDAErrorCode::ComputationTimeout,
                                                    "Cannot push to shutdown queue");
                }
                queue_.push_back(std::move(task));
                size_.fetch_add(1, std::memory_order_acq_rel);
            }
            cv_.notify_one();
            return error::Result<void>::ok();
        }
        catch (const std::exception &e)
        {
            return error::Result<void>::err(error::TDAErrorCode::AllocationFailed,
                                            std::string("Failed to push task: ") + e.what());
        }
    }

    void pushUnsafe(T task)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!shutdown_.load(std::memory_order_acquire))
            {
                queue_.push_back(std::move(task));
                size_.fetch_add(1, std::memory_order_acq_rel);
            }
        }
        cv_.notify_one();
    }

    error::Result<std::optional<T>> pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock,
                 [this] { return !queue_.empty() || shutdown_.load(std::memory_order_acquire); });

        if (shutdown_.load(std::memory_order_acquire) && queue_.empty())
        {
            return error::Result<std::optional<T>>::ok(std::nullopt);
        }

        if (queue_.empty())
        {
            return error::Result<std::optional<T>>::ok(std::nullopt);
        }

        T task = std::move(queue_.back());
        queue_.pop_back();
        size_.fetch_sub(1, std::memory_order_acq_rel);

        return error::Result<std::optional<T>>::ok(std::move(task));
    }

    error::Result<std::optional<T>> tryPop()
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock)
        {
            return error::Result<std::optional<T>>::err(error::TDAErrorCode::DataRace,
                                                        "Could not acquire lock for tryPop");
        }

        if (queue_.empty())
        {
            return error::Result<std::optional<T>>::ok(std::nullopt);
        }

        T task = std::move(queue_.back());
        queue_.pop_back();
        size_.fetch_sub(1, std::memory_order_acq_rel);

        return error::Result<std::optional<T>>::ok(std::move(task));
    }

    error::Result<std::optional<T>> steal()
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock)
        {
            return error::Result<std::optional<T>>::ok(std::nullopt);
        }

        if (queue_.empty())
        {
            return error::Result<std::optional<T>>::ok(std::nullopt);
        }

        T task = std::move(queue_.front());
        queue_.pop_front();
        size_.fetch_sub(1, std::memory_order_acq_rel);

        return error::Result<std::optional<T>>::ok(std::move(task));
    }

    size_type size() const { return size_.load(std::memory_order_acquire); }

    bool empty() const { return size_.load(std::memory_order_acquire) == 0; }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }

    bool isShutdown() const { return shutdown_.load(std::memory_order_acquire); }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
        size_.store(0, std::memory_order_release);
    }

    struct Stats
    {
        size_type size;
        bool isShutdown;
        bool isEmpty;
    };

    Stats getStats() const
    {
        return Stats{size_.load(std::memory_order_acquire),
                     shutdown_.load(std::memory_order_acquire),
                     size_.load(std::memory_order_acquire) == 0};
    }
};

template <typename T>
error::Result<std::unique_ptr<WorkStealingQueue<T>>> makeWorkStealingQueue()
{
    try
    {
        return error::Result<std::unique_ptr<WorkStealingQueue<T>>>::ok(
            std::make_unique<WorkStealingQueue<T>>());
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<WorkStealingQueue<T>>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create work-stealing queue: ") + e.what());
    }
}

using TaskQueue = WorkStealingQueue<std::function<void()>>;

inline error::Result<std::unique_ptr<TaskQueue>> makeTaskQueue()
{
    return makeWorkStealingQueue<std::function<void()>>();
}

} // namespace nerve::threading
