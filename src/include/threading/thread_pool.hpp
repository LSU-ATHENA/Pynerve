
#pragma once
#include "nerve/error/error_registry.hpp"
#include "threading/work_stealing_queue.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace nerve::threading
{

template <typename T>
concept ValidThreadCount = std::integral<T> && requires(T n) {
    { n } -> std::convertible_to<size_t>;
    requires n > 0;
};

// Thread pool with proper lifecycle management and work-stealing
class ThreadPool
{
private:
    std::vector<std::unique_ptr<TaskQueue>> task_queues_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<size_t> active_tasks_{0};
    std::atomic<size_t> total_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};
    std::atomic<size_t> failed_tasks_{0};
    std::atomic<size_t> next_queue_{0};
    size_t num_threads_;

    template <typename ReturnType, typename BoundTask>
    auto makeTrackedTask(BoundTask &&bound_task)
    {
        return std::make_shared<std::packaged_task<ReturnType()>>(
            [this, bound_task = std::forward<BoundTask>(bound_task)]() mutable -> ReturnType {
                try
                {
                    if constexpr (std::is_void_v<ReturnType>)
                    {
                        bound_task();
                        completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
                    }
                    else if constexpr (std::is_reference_v<ReturnType>)
                    {
                        ReturnType result = bound_task();
                        completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
                        return result;
                    }
                    else
                    {
                        ReturnType result = bound_task();
                        completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
                        return result;
                    }
                }
                catch (...)
                {
                    failed_tasks_.fetch_add(1, std::memory_order_acq_rel);
                    throw;
                }
            });
    }

public:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency())
        : num_threads_(numThreads > 0 ? numThreads : 1)
    {
        if (num_threads_ == 0)
        {
            num_threads_ = 1;
        }

        // Create task queues for each thread
        task_queues_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i)
        {
            auto queue_result = makeTaskQueue();
            if (queue_result.isErr())
            {
                for (auto &queue : task_queues_)
                {
                    if (queue)
                        queue->shutdown();
                }
                throw std::runtime_error("Failed to create thread-pool task queue");
            }
            task_queues_.push_back(std::move(queue_result.value()));
        }

        workers_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i)
        {
            workers_.emplace_back([this, i] { workerThread(i); });
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    ~ThreadPool()
    {
        shutdown();
        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>
    {
        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task =
            makeTrackedTask<ReturnType>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto future = task->get_future();

        auto wrapped_task = [this, task]() {
            try
            {
                ++active_tasks_;
                (*task)();
            }
            catch (...)
            {
                ++failed_tasks_;
                // Keep worker threads alive; task exceptions are retained by
                // packaged_task and can be observed via the returned future.
            }
            --active_tasks_;
        };

        total_tasks_.fetch_add(1, std::memory_order_acq_rel);
        size_t queue_index = next_queue_.fetch_add(1, std::memory_order_relaxed) % num_threads_;
        auto submit_result = task_queues_[queue_index]->push(std::move(wrapped_task));
        if (submit_result.isErr())
        {
            total_tasks_.fetch_sub(1, std::memory_order_acq_rel);
            throw std::runtime_error("Failed to submit task to thread pool");
        }

        return future;
    }

    template <typename F, typename... Args>
    auto submitToQueue(size_t queue_index, F &&f,
                       Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>
    {
        if (queue_index >= num_threads_)
        {
            throw std::out_of_range("Queue index out of range");
        }

        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task =
            makeTrackedTask<ReturnType>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto future = task->get_future();

        auto wrapped_task = [this, task]() {
            try
            {
                ++active_tasks_;
                (*task)();
            }
            catch (...)
            {
                ++failed_tasks_;
            }
            --active_tasks_;
        };

        total_tasks_.fetch_add(1, std::memory_order_acq_rel);
        auto submit_result = task_queues_[queue_index]->push(std::move(wrapped_task));
        if (submit_result.isErr())
        {
            total_tasks_.fetch_sub(1, std::memory_order_acq_rel);
            throw std::runtime_error("Failed to submit task to specific queue");
        }

        return future;
    }

    void shutdown()
    {
        if (stop_flag_.exchange(true, std::memory_order_acq_rel))
        {
            return;
        }

        for (auto &queue : task_queues_)
        {
            if (queue)
            {
                queue->shutdown();
            }
        }
    }

    void waitForAll()
    {
        while (!allTasksCompleted())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    template <typename Rep, typename Period>
    bool waitForAllFor(const std::chrono::duration<Rep, Period> &timeout)
    {
        auto start = std::chrono::steady_clock::now();
        while (!allTasksCompleted())
        {
            if (std::chrono::steady_clock::now() - start > timeout)
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    struct Stats
    {
        size_t numThreads;
        size_t activeTasks;
        size_t total_tasks;
        size_t completed_tasks;
        size_t failed_tasks;
        bool isShutdown;
        double completion_rate;
    };

    Stats getStats() const
    {
        size_t total = total_tasks_.load(std::memory_order_acquire);
        size_t completed = completed_tasks_.load(std::memory_order_acquire);
        size_t failed = failed_tasks_.load(std::memory_order_acquire);

        double completion_rate =
            (total > 0) ? static_cast<double>(completed + failed) / static_cast<double>(total)
                        : 0.0;

        return Stats{
            num_threads_, active_tasks_.load(std::memory_order_acquire), total,          completed,
            failed,       stop_flag_.load(std::memory_order_acquire),    completion_rate};
    }

    size_t numThreads() const { return num_threads_; }

    bool isShutdown() const { return stop_flag_.load(std::memory_order_acquire); }

    size_t activeTasks() const { return active_tasks_.load(std::memory_order_acquire); }

    bool allTasksCompleted() const
    {
        size_t total = total_tasks_.load(std::memory_order_acquire);
        size_t completed = completed_tasks_.load(std::memory_order_acquire);
        size_t failed = failed_tasks_.load(std::memory_order_acquire);
        return completed + failed >= total && active_tasks_.load(std::memory_order_acquire) == 0;
    }

private:
    void workerThread(size_t thread_id)
    {
        auto &local_queue = task_queues_[thread_id];

        while (true)
        {
            std::function<void()> task;
            bool got_task = false;

            auto pop_result = local_queue->tryPop();
            if (pop_result.isOk())
            {
                auto queued_task = std::move(pop_result.value());
                if (queued_task.has_value())
                {
                    task = std::move(*queued_task);
                    got_task = true;
                }
            }
            if (!got_task)
            {
                for (size_t i = 0; i < num_threads_; ++i)
                {
                    if (i == thread_id)
                        continue;

                    auto steal_result = task_queues_[i]->steal();
                    if (steal_result.isOk())
                    {
                        auto stolen_task = std::move(steal_result.value());
                        if (stolen_task.has_value())
                        {
                            task = std::move(*stolen_task);
                            got_task = true;
                            break;
                        }
                    }
                }
            }

            if (got_task)
            {
                task();
            }
            else if (stop_flag_.load(std::memory_order_acquire))
            {
                break;
            }
            else
            {
                auto wait_result = local_queue->pop();
                if (wait_result.isOk())
                {
                    auto queued_task = std::move(wait_result.value());
                    if (queued_task.has_value())
                    {
                        task = std::move(*queued_task);
                        task();
                    }
                }
            }
        }
    }
};

inline error::Result<std::unique_ptr<ThreadPool>> makeThreadPool(size_t numThreads = 0)
{
    try
    {
        if (numThreads == 0)
        {
            numThreads = std::thread::hardware_concurrency();
        }
        if (numThreads == 0)
        {
            numThreads = 1;
        }

        auto pool = std::make_unique<ThreadPool>(numThreads);
        return error::Result<std::unique_ptr<ThreadPool>>::ok(std::move(pool));
    }
    catch (const std::exception &e)
    {
        return error::Result<std::unique_ptr<ThreadPool>>::err(
            error::TDAErrorCode::AllocationFailed,
            std::string("Failed to create thread pool: ") + e.what());
    }
}

} // namespace nerve::threading
