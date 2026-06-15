#pragma once
#include "nerve/core_types.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace nerve::core
{

struct CpuTopology
{
    int num_packages = 0;
    int num_cores = 0;
    int num_threads = 0;
    int numa_nodes = 1;
    std::vector<int> core_to_numa;
    std::vector<int> thread_to_core;

    int packageOf(int cpu_id) const;
    int numaNodeOf(int cpu_id) const;
    int coreOf(int cpu_id) const;
    bool sameCoreAs(int a, int b) const;
    bool sameNumaAs(int a, int b) const;
};

CpuTopology detectCpuTopology();

class ThreadPool
{
public:
    explicit ThreadPool(int num_threads = 0, bool pin_to_cores = true, bool use_numa = false);
    ~ThreadPool();

    void enqueue(std::function<void(int thread_id)> task);
    void wait();
    Size threadCount() const;
    int threadToCpu(int thread_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NumaAwareThreadPool
{
public:
    explicit NumaAwareThreadPool(int threads_per_node = -1);
    ~NumaAwareThreadPool();

    void enqueueOnNode(int numa_node, std::function<void()> task);
    void waitAll();
    Size nodeCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void pinCurrentThreadToCore(int cpu_id);
void pinCurrentThreadToPackage(int package_id);
void pinCurrentThreadToNumaNode(int numa_node);
void pinThreadToCore(std::thread &t, int cpu_id);
int getCurrentCpu();
int getCurrentNumaNode();

} // namespace nerve::core
