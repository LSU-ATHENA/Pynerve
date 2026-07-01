#include "nerve/core/thread_affinity.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace
{

bool check_cpu_topology_detection_no_crash()
{
    auto topo = nerve::core::detectCpuTopology();
    (void)topo;
    return true;
}

bool check_cpu_topology_reports_non_zero_cores()
{
    auto topo = nerve::core::detectCpuTopology();
    if (topo.num_cores <= 0)
    {
        std::cerr << "num_cores should be positive, got " << topo.num_cores << "\n";
        return false;
    }
    return true;
}

bool check_thread_pool_construction_teardown()
{
    for (int i = 1; i <= 4; ++i)
    {
        nerve::core::ThreadPool pool(i);
        (void)pool;
    }
    return true;
}

bool check_thread_pool_enqueue_wait_completes_all()
{
    nerve::core::ThreadPool pool(4);
    std::atomic<int> counter{0};
    const int kTasks = 100;
    for (int i = 0; i < kTasks; ++i)
    {
        pool.enqueue([&counter](int) { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    pool.wait();
    for (int retry = 0; retry < 100 && counter.load() != kTasks; ++retry)
    {
        std::this_thread::yield();
    }
    if (counter.load() != kTasks)
    {
        std::cerr << "expected " << kTasks << " completions, got " << counter.load() << "\n";
        return false;
    }
    return true;
}

bool check_thread_pool_thread_count_matches()
{
    nerve::core::ThreadPool pool(8);
    if (pool.threadCount() != 8)
    {
        std::cerr << "expected thread count 8, got " << pool.threadCount() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_cpu_topology_detection_no_crash())
    {
        std::cerr << "FAIL: cpu topology detection no crash\n";
        return 1;
    }
    if (!check_cpu_topology_reports_non_zero_cores())
    {
        std::cerr << "FAIL: cpu topology reports non-zero cores\n";
        return 1;
    }
    if (!check_thread_pool_construction_teardown())
    {
        std::cerr << "FAIL: thread pool construction teardown\n";
        return 1;
    }
    if (!check_thread_pool_enqueue_wait_completes_all())
    {
        std::cerr << "FAIL: thread pool enqueue wait completes all\n";
        return 1;
    }
    if (!check_thread_pool_thread_count_matches())
    {
        std::cerr << "FAIL: thread pool thread count matches\n";
        return 1;
    }
    return 0;
}
