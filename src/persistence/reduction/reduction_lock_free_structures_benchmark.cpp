
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nerve::persistence::lockfree
{

namespace
{

void validateBenchmarkInputs(int num_threads, int operations_per_thread, size_t table_size)
{
    if (num_threads <= 0)
    {
        throw std::invalid_argument("lockfree benchmark requires at least one thread");
    }
    if (operations_per_thread <= 0)
    {
        throw std::invalid_argument("lockfree benchmark requires at least one operation");
    }
    if (table_size == 0)
    {
        throw std::invalid_argument("lockfree benchmark requires a non-empty table");
    }
    if (table_size > static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
    {
        throw std::overflow_error("lockfree benchmark table size exceeds int key range");
    }
}

} // namespace

LockFreeBenchmark benchmarkLockFree(int num_threads, int operations_per_thread, size_t table_size)
{
    validateBenchmarkInputs(num_threads, operations_per_thread, table_size);

    LockFreeBenchmark bench{};
    bench.num_threads = num_threads;
    bench.contention_level = table_size;

    std::vector<int> keys;
    keys.reserve(table_size);
    for (size_t i = 0; i < table_size; ++i)
    {
        keys.push_back(static_cast<int>(i * 2));
    }

    {
        std::mutex mtx;
        std::unordered_map<int, int> map;
        for (size_t i = 0; i < table_size; ++i)
        {
            map[keys[i]] = static_cast<int>(i);
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(num_threads));
        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&]() {
                for (int i = 0; i < operations_per_thread; ++i)
                {
                    int key = keys[i % keys.size()];
                    std::lock_guard<std::mutex> lock(mtx);
                    auto it = map.find(key);
                    if (it != map.end())
                    {
                        volatile int val = it->second;
                        (void)val;
                    }
                }
            });
        }

        for (auto &t : threads)
        {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        bench.mutex_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    {
        LockFreePivotTable table(table_size * 2);
        for (size_t i = 0; i < table_size; ++i)
        {
            table.tryInsert(keys[i], static_cast<int>(i));
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(num_threads));
        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&]() {
                for (int i = 0; i < operations_per_thread; ++i)
                {
                    int key = keys[i % keys.size()];
                    volatile int val = table.find(key);
                    (void)val;
                }
            });
        }

        for (auto &t : threads)
        {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        bench.lockfree_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    bench.speedup =
        bench.lockfree_time_ms > 0.0 ? bench.mutex_time_ms / bench.lockfree_time_ms : 0.0;

    return bench;
}

LockFreeConfig getOptimalLockFreeConfig(int num_threads)
{
    LockFreeConfig config;

    const int safe_threads = std::max(num_threads, 1);
    config.queue_capacity = 1024;
    config.table_initial_capacity = 1024;
    config.max_steal_attempts = std::min(safe_threads / 4, 8);

    return config;
}

LockFreeEstimate estimateLockFreeBenefit(int num_threads)
{
    LockFreeEstimate estimate{};

    if (num_threads < 8)
    {
        estimate.speedup = 1.0;
        estimate.recommended = false;
        return estimate;
    }

    if (num_threads < 16)
    {
        estimate.speedup = 1.2;
    }
    else if (num_threads < 32)
    {
        estimate.speedup = 1.5;
    }
    else
    {
        estimate.speedup = 2.0;
    }

    estimate.recommended = (num_threads >= 16);

    return estimate;
}

} // namespace nerve::persistence::lockfree
