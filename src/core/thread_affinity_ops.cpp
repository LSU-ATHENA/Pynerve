#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#ifdef NERVE_HAS_NUMA
#include <numa.h>
#endif
#endif

#include "nerve/core/thread_affinity.hpp"

namespace nerve::core
{

CpuTopology detectCpuTopology()
{
    CpuTopology topo;
#ifdef __linux__
    topo.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    topo.num_packages = 1;
    topo.num_cores = topo.num_threads / 2;
    if (topo.num_cores < 1)
        topo.num_cores = topo.num_threads;
    topo.core_to_numa.resize(topo.num_threads, 0);
    topo.thread_to_core.resize(topo.num_threads);
    for (int i = 0; i < topo.num_threads; ++i)
    {
        topo.thread_to_core[i] = i % topo.num_cores;
    }
#ifdef NERVE_HAS_NUMA
    if (numa_available() >= 0)
    {
        topo.numa_nodes = numa_max_node() + 1;
        topo.num_packages = topo.numa_nodes;
        for (int i = 0; i < topo.num_threads && i < 1024; ++i)
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            topo.core_to_numa[i] = numa_node_of_cpu(i);
        }
    }
#endif
#else
    topo.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    topo.num_cores = topo.num_threads;
    topo.num_packages = 1;
    topo.numa_nodes = 1;
    topo.thread_to_core.resize(topo.num_threads);
    for (int i = 0; i < topo.num_threads; ++i)
        topo.thread_to_core[i] = i;
#endif
    return topo;
}

int CpuTopology::packageOf(int cpu_id) const
{
    if (numa_nodes > 1)
        return numaNodeOf(cpu_id);
    return 0;
}

int CpuTopology::numaNodeOf(int cpu_id) const
{
    if (cpu_id < static_cast<int>(core_to_numa.size()))
        return core_to_numa[cpu_id];
    return 0;
}

int CpuTopology::coreOf(int cpu_id) const
{
    if (cpu_id < static_cast<int>(thread_to_core.size()))
        return thread_to_core[cpu_id];
    return cpu_id;
}

bool CpuTopology::sameCoreAs(int a, int b) const
{
    return coreOf(a) == coreOf(b);
}

bool CpuTopology::sameNumaAs(int a, int b) const
{
    return numaNodeOf(a) == numaNodeOf(b);
}

void pinCurrentThreadToCore(int cpu_id)
{
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)cpu_id;
#endif
}

void pinCurrentThreadToPackage(int package_id)
{
#ifdef __linux__
    auto topo = detectCpuTopology();
    for (int i = 0; i < topo.num_threads; ++i)
    {
        if (topo.packageOf(i) == package_id)
        {
            pinCurrentThreadToCore(i);
            return;
        }
    }
#else
    (void)package_id;
#endif
}

void pinCurrentThreadToNumaNode(int numa_node)
{
#ifdef __linux__

#ifdef NERVE_HAS_NUMA
    if (numa_available() >= 0)
    {
        struct bitmask *mask = numa_allocate_cpumask();
        numa_node_to_cpus(numa_node, mask);
        numa_sched_setaffinity(0, mask);
        numa_free_cpumask(mask);
        return;
    }
#endif
    auto topo = detectCpuTopology();
    for (int i = 0; i < topo.num_threads; ++i)
    {
        if (topo.numaNodeOf(i) == numa_node)
        {
            pinCurrentThreadToCore(i);
            return;
        }
    }
#else
    (void)numa_node;
#endif
}

void pinThreadToCore(std::thread &t, int cpu_id)
{
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#else
    (void)t;
    (void)cpu_id;
#endif
}

int getCurrentCpu()
{
#ifdef __linux__
    return sched_getcpu();
#else
    return 0;
#endif
}

int getCurrentNumaNode()
{
#ifdef __linux__
#ifdef NERVE_HAS_NUMA
    if (numa_available() >= 0)
        return numa_node_of_cpu(sched_getcpu());
#endif
    return 0;
#else
    return 0;
#endif
}

struct ThreadPool::Impl
{
    int num_threads;
    std::vector<std::thread> workers;
    std::vector<std::function<void(int)>> queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::atomic<int> active{0};
    std::vector<int> cpu_mapping;

    explicit Impl(int n, bool pin, bool use_numa)
        : num_threads(n > 0 ? n : static_cast<int>(std::thread::hardware_concurrency()))
    {
        auto topo = detectCpuTopology();
        cpu_mapping.resize(num_threads);
        int next_cpu = 0;
        for (int i = 0; i < num_threads; ++i)
        {
            if (use_numa)
            {
                int node = i % topo.numa_nodes;
                for (int c = 0; c < topo.num_threads; ++c)
                {
                    if (topo.numaNodeOf(c) == node)
                    {
                        cpu_mapping[i] = c;
                        break;
                    }
                }
            }
            else
            {
                cpu_mapping[i] = next_cpu++ % topo.num_threads;
            }
        }

        for (int i = 0; i < num_threads; ++i)
        {
            workers.emplace_back([this, i, pin] {
                if (pin)
                    pinCurrentThreadToCore(cpu_mapping[i]);
                while (true)
                {
                    std::function<void(int)> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [this] { return stop || !queue.empty(); });
                        if (stop && queue.empty())
                            return;
                        task = std::move(queue.back());
                        queue.pop_back();
                    }
                    active++;
                    task(i);
                    active--;
                }
            });
        }
    }

    ~Impl()
    {
        stop = true;
        cv.notify_all();
        for (auto &w : workers)
            if (w.joinable())
                w.join();
    }
};

ThreadPool::ThreadPool(int num_threads, bool pin_to_cores, bool use_numa)
    : impl_(std::make_unique<Impl>(num_threads, pin_to_cores, use_numa))
{}

ThreadPool::~ThreadPool() = default;

void ThreadPool::enqueue(std::function<void(int)> task)
{
    {
        std::lock_guard<std::mutex> lock(impl_->queue_mutex);
        impl_->queue.push_back(std::move(task));
    }
    impl_->cv.notify_one();
}

void ThreadPool::wait()
{
    while (impl_->active > 0 || !impl_->queue.empty())
        std::this_thread::yield();
}

Size ThreadPool::threadCount() const
{
    return impl_->num_threads;
}

int ThreadPool::threadToCpu(int thread_id) const
{
    if (thread_id < static_cast<int>(impl_->cpu_mapping.size()))
        return impl_->cpu_mapping[thread_id];
    return thread_id;
}

struct NumaAwareThreadPool::Impl
{
    std::vector<std::unique_ptr<ThreadPool>> node_pools;
    int nodes;

    explicit Impl(int threads_per_node)
    {
        auto topo = detectCpuTopology();
        nodes = topo.numa_nodes;
        int per_node = threads_per_node > 0 ? threads_per_node : topo.num_threads / nodes;
        for (int n = 0; n < nodes; ++n)
        {
            node_pools.push_back(std::make_unique<ThreadPool>(per_node, true, true));
        }
    }
};

NumaAwareThreadPool::NumaAwareThreadPool(int threads_per_node)
    : impl_(std::make_unique<Impl>(threads_per_node))
{}

NumaAwareThreadPool::~NumaAwareThreadPool() = default;

void NumaAwareThreadPool::enqueueOnNode(int numa_node, std::function<void()> task)
{
    if (numa_node >= 0 && numa_node < static_cast<int>(impl_->node_pools.size()))
    {
        impl_->node_pools[numa_node]->enqueue([t = std::move(task)](int) { t(); });
    }
}

void NumaAwareThreadPool::waitAll()
{
    for (auto &pool : impl_->node_pools)
        pool->wait();
}

Size NumaAwareThreadPool::nodeCount() const
{
    return impl_->nodes;
}

} // namespace nerve::core
