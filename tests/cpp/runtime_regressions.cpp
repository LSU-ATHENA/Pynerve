#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/error/error_registry.hpp"
#include "threading/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Index;
using nerve::Size;
using nerve::core::BufferView;
using nerve::threading::ThreadPool;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

bool check_thread_pool_construction()
{
    ThreadPool pool(2);
    if (pool.numThreads() != 2)
    {
        std::cerr << "expected 2 threads, got " << pool.numThreads() << "\n";
        return false;
    }
    return true;
}

bool check_thread_pool_submit_single_task()
{
    ThreadPool pool(2);
    auto future = pool.submit([]() { return 42; });
    auto result = future.get();
    if (result != 42)
    {
        std::cerr << "expected 42, got " << result << "\n";
        return false;
    }
    return true;
}

bool check_thread_pool_submit_multiple_tasks()
{
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; ++i)
    {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }

    for (int i = 0; i < 10; ++i)
    {
        auto result = futures[static_cast<size_t>(i)].get();
        if (result != i * i)
        {
            std::cerr << "task " << i << " expected " << (i * i) << ", got " << result << "\n";
            return false;
        }
    }
    return true;
}

bool check_thread_pool_wait_for_all()
{
    ThreadPool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 20; ++i)
    {
        pool.submit([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.waitForAll();

    auto stats = pool.getStats();
    if (stats.completed_tasks + stats.failed_tasks < 20)
    {
        std::cerr << "expected at least 20 completed tasks, got " << stats.completed_tasks << "\n";
        return false;
    }
    return true;
}

bool check_thread_pool_all_tasks_completed()
{
    ThreadPool pool(2);

    for (int i = 0; i < 5; ++i)
    {
        pool.submit([]() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
    }

    pool.waitForAll();
    if (!pool.allTasksCompleted())
    {
        std::cerr << "allTasksCompleted should be true after waitForAll\n";
        return false;
    }
    return true;
}

bool check_thread_pool_submit_to_queue()
{
    ThreadPool pool(2);
    auto future = pool.submitToQueue(0, []() { return 1.0; });
    auto result = future.get();
    if (std::abs(result - 1.0) > 1e-12)
    {
        std::cerr << "expected 1.0\n";
        return false;
    }
    return true;
}

bool check_thread_pool_stats()
{
    ThreadPool pool(2);
    auto stats = pool.getStats();
    if (stats.numThreads != 2)
    {
        std::cerr << "stats.numThreads should be 2\n";
        return false;
    }
    if (pool.isShutdown())
    {
        std::cerr << "pool should not be shutdown\n";
        return false;
    }
    return true;
}

bool check_thread_pool_shutdown()
{
    ThreadPool pool(2);
    pool.shutdown();
    if (!pool.isShutdown())
    {
        std::cerr << "pool should be shutdown\n";
        return false;
    }
    return true;
}

bool check_make_thread_pool()
{
    auto pool_result = nerve::threading::makeThreadPool(2);
    if (pool_result.isErr())
    {
        std::cerr << "makeThreadPool failed\n";
        return false;
    }
    auto pool = std::move(pool_result.value());
    if (!pool || pool->numThreads() != 2)
    {
        std::cerr << "invalid thread pool from makeThreadPool\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_thread_pool_construction())
    {
        std::cerr << "FAIL: ThreadPool construction\n";
        return 1;
    }
    if (!check_thread_pool_submit_single_task())
    {
        std::cerr << "FAIL: submit single task\n";
        return 1;
    }
    if (!check_thread_pool_submit_multiple_tasks())
    {
        std::cerr << "FAIL: submit multiple tasks\n";
        return 1;
    }
    if (!check_thread_pool_wait_for_all())
    {
        std::cerr << "FAIL: waitForAll\n";
        return 1;
    }
    if (!check_thread_pool_all_tasks_completed())
    {
        std::cerr << "FAIL: allTasksCompleted\n";
        return 1;
    }
    if (!check_thread_pool_submit_to_queue())
    {
        std::cerr << "FAIL: submitToQueue\n";
        return 1;
    }
    if (!check_thread_pool_stats())
    {
        std::cerr << "FAIL: stats\n";
        return 1;
    }
    if (!check_thread_pool_shutdown())
    {
        std::cerr << "FAIL: shutdown\n";
        return 1;
    }
    if (!check_make_thread_pool())
    {
        std::cerr << "FAIL: makeThreadPool\n";
        return 1;
    }
    return 0;
}
