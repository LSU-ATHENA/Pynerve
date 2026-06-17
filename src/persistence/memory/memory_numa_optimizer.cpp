
#include "nerve/persistence/memory/numa_memory_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#if ENABLE_NUMA
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace nerve::persistence::numa
{

namespace
{

constexpr size_t kBytesPerMb = 1024ULL * 1024ULL;
constexpr size_t kHugePageSize = 2ULL * 1024ULL * 1024ULL;
constexpr size_t kPrefetchPageSize = 4096ULL;
constexpr double kBaselineSpeedup = 1.0;

size_t roundUp(size_t value, size_t align)
{
    if (align == 0)
    {
        return value;
    }
    const size_t rem = value % align;
    return rem == 0 ? value : (value + align - rem);
}

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return kBaselineSpeedup;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : kBaselineSpeedup;
}

int safeHardwareThreads()
{
    const unsigned int hw = std::thread::hardware_concurrency();
    return static_cast<int>(hw == 0 ? 1U : hw);
}

#if ENABLE_NUMA
void tryBindAllocationToNode(void *ptr, size_t size, int node)
{
    if (ptr == nullptr || size == 0 || node < 0)
    {
        return;
    }
    if (numa_available() < 0 || numa_num_configured_nodes() <= 1)
    {
        return;
    }
    unsigned long nodemask = 1UL << static_cast<unsigned long>(node);
    const long maxnode = static_cast<long>(sizeof(nodemask) * 8U);
    (void)mbind(ptr, size, MPOL_BIND, &nodemask, maxnode, 0);
}

void tryInterleaveAllocation(void *ptr, size_t size)
{
    if (ptr == nullptr || size == 0)
    {
        return;
    }
    if (numa_available() < 0 || numa_num_configured_nodes() <= 1)
    {
        return;
    }
    const int nodes = numa_num_configured_nodes();
    unsigned long nodemask = 0UL;
    for (int node = 0; node < nodes; ++node)
    {
        nodemask |= (1UL << static_cast<unsigned long>(node));
    }
    const long maxnode = static_cast<long>(sizeof(nodemask) * 8U);
    (void)mbind(ptr, size, MPOL_INTERLEAVE, &nodemask, maxnode, 0);
}

void tryHugePageAdvice(void *ptr, size_t size)
{
    if (ptr == nullptr || size == 0)
    {
        return;
    }
    (void)madvise(ptr, size, MADV_HUGEPAGE);
}
#endif

} // namespace

class NumaMemoryChunk final
{
public:
    NumaMemoryChunk(size_t size, int node, bool use_huge_pages)
        : ptr_(nullptr)
        , size_(size)
        , node_(node)
        , uses_huge_pages_(use_huge_pages)
    {
        ptr_ = use_huge_pages ? allocateHugePages(size) : allocateOnNode(size, node);
        if (ptr_ == nullptr)
        {
            ptr_ = std::malloc(size);
            uses_huge_pages_ = false;
        }
    }

    ~NumaMemoryChunk() { std::free(ptr_); }

    NumaMemoryChunk(const NumaMemoryChunk &) = delete;
    NumaMemoryChunk &operator=(const NumaMemoryChunk &) = delete;

    NumaMemoryChunk(NumaMemoryChunk &&) = delete;
    NumaMemoryChunk &operator=(NumaMemoryChunk &&) = delete;

    [[nodiscard]] void *data() { return ptr_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] int node() const { return node_; }
    [[nodiscard]] bool usesHugePages() const { return uses_huge_pages_; }

private:
    void *ptr_;
    size_t size_;
    int node_;
    bool uses_huge_pages_;
};

bool isNumaAvailable()
{
#if ENABLE_NUMA
    return numa_available() >= 0;
#else
    return false;
#endif
}

NumaTopology detectNumaTopology()
{
    NumaTopology topology;
    topology.num_nodes = 1;
    topology.num_cpus = safeHardwareThreads();

#if ENABLE_NUMA
    if (!isNumaAvailable())
    {
        return topology;
    }

    topology.num_nodes = std::max(1, numa_num_configured_nodes());
    topology.num_cpus = std::max(1, numa_num_configured_cpus());
    topology.node_memory_mb.reserve(static_cast<size_t>(topology.num_nodes));
    topology.cpus_per_node.reserve(static_cast<size_t>(topology.num_nodes));

    for (int node = 0; node < topology.num_nodes; ++node)
    {
        long long free_mem = numa_node_size64(node, nullptr);
        topology.node_memory_mb.push_back(static_cast<long>(
            std::max<long long>(0, free_mem / static_cast<long long>(kBytesPerMb))));

        bitmask *cpumask = numa_allocate_cpumask();
        if (cpumask != nullptr && numa_node_to_cpus(node, cpumask) == 0)
        {
            topology.cpus_per_node.push_back(numa_bitmask_weight(cpumask));
        }
        else
        {
            topology.cpus_per_node.push_back(0);
        }
        if (cpumask != nullptr)
        {
            numa_bitmask_free(cpumask);
        }
    }
#endif

    return topology;
}

void *allocateOnNode(size_t size, int node)
{
    if (size == 0)
    {
        return nullptr;
    }
    void *ptr = std::malloc(size);
    if (ptr == nullptr)
    {
        return nullptr;
    }
#if ENABLE_NUMA
    tryBindAllocationToNode(ptr, size, node);
#else
    (void)node;
#endif
    return ptr;
}

void *allocateHugePages(size_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    const size_t alloc_size = roundUp(size, kHugePageSize);
    void *ptr = std::aligned_alloc(kHugePageSize, alloc_size);
    if (ptr == nullptr)
    {
        ptr = std::malloc(size);
    }
#if ENABLE_NUMA
    tryHugePageAdvice(ptr, alloc_size);
#endif
    return ptr;
}

void bindThreadToNode(int node)
{
#if ENABLE_NUMA
    if (!isNumaAvailable() || node < 0)
    {
        return;
    }
    (void)numa_run_on_node(node);
#else
    if (node < 0 || node > 0)
    {
        return;
    }
#endif
}

int getCurrentNode()
{
#if ENABLE_NUMA
    if (!isNumaAvailable())
    {
        return 0;
    }
    const int cpu = sched_getcpu();
    if (cpu < 0)
    {
        return 0;
    }
    const int node = numa_node_of_cpu(cpu);
    return node < 0 ? 0 : node;
#else
    return 0;
#endif
}

void *allocateInterleaved(size_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    void *ptr = std::malloc(size);
    if (ptr == nullptr)
    {
        return nullptr;
    }
#if ENABLE_NUMA
    tryInterleaveAllocation(ptr, size);
#endif
    return ptr;
}

void setMemoryPolicy(void *addr, size_t len, int node)
{
    if (addr == nullptr || len == 0 || node < 0)
    {
        return;
    }
#if ENABLE_NUMA
    tryBindAllocationToNode(addr, len, node);
#endif
}

void prefetchMemory(void *addr, size_t len)
{
    if (addr == nullptr || len == 0)
    {
        return;
    }
#if ENABLE_NUMA
    (void)madvise(addr, len, MADV_WILLNEED);
#endif
    auto *bytes = static_cast<const char *>(addr);
    for (size_t offset = 0; offset < len; offset += kPrefetchPageSize)
    {
        __builtin_prefetch(bytes + offset, 0, 3);
    }
}

NumaMatrixStorage createNumaMatrixStorage(size_t num_columns, size_t column_size_bytes,
                                          const NumaConfig &config)
{
    NumaMatrixStorage storage;
    storage.num_columns = static_cast<int>(num_columns);
    storage.column_size = column_size_bytes;
    if (num_columns == 0 || column_size_bytes == 0)
    {
        return storage;
    }

    const NumaTopology topology = detectNumaTopology();
    const int nodes = std::max(1, topology.num_nodes);
    const size_t columns_per_node =
        (num_columns + static_cast<size_t>(nodes) - 1) / static_cast<size_t>(nodes);

    for (int node = 0; node < nodes; ++node)
    {
        const size_t start_col = static_cast<size_t>(node) * columns_per_node;
        const size_t end_col = std::min(num_columns, start_col + columns_per_node);
        if (start_col >= end_col)
        {
            break;
        }
        const size_t chunk_cols = end_col - start_col;
        const size_t chunk_bytes = chunk_cols * column_size_bytes;
        storage.chunks.emplace_back(
            std::make_unique<NumaMemoryChunk>(chunk_bytes, node, config.use_huge_pages));
        storage.node_to_columns[node] = {static_cast<int>(start_col),
                                         static_cast<int>(end_col - 1)};
    }

    return storage;
}

NumaConfig getOptimalNumaConfig(size_t num_columns, size_t column_size)
{
    NumaConfig config;
    const NumaTopology topology = detectNumaTopology();
    const size_t total_bytes = num_columns * column_size;

    config.use_numa_allocation = topology.num_nodes > 1;
    config.bind_threads = topology.num_nodes > 1;
    config.use_huge_pages = total_bytes >= (64ULL * kBytesPerMb);
    config.interleave = total_bytes >= (512ULL * kBytesPerMb) && topology.num_nodes > 1;
    return config;
}

NumaBenchmark benchmarkNuma(size_t size_bytes, int iterations)
{
    NumaBenchmark bench{0.0, 0.0, 0.0, 1.0};
    if (size_bytes == 0 || iterations <= 0)
    {
        return bench;
    }

    const int loops = std::max(1, iterations);
    auto start_regular = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; ++i)
    {
        void *ptr = std::malloc(size_bytes);
        if (ptr == nullptr)
        {
            continue;
        }
        std::memset(ptr, 0, size_bytes);
        std::free(ptr);
    }
    auto end_regular = std::chrono::high_resolution_clock::now();
    bench.regular_time_ms =
        std::chrono::duration<double, std::milli>(end_regular - start_regular).count();

    auto start_numa = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; ++i)
    {
        void *ptr = allocateOnNode(size_bytes, 0);
        if (ptr == nullptr)
        {
            continue;
        }
        std::memset(ptr, 0, size_bytes);
        std::free(ptr);
    }
    auto end_numa = std::chrono::high_resolution_clock::now();
    bench.numa_time_ms = std::chrono::duration<double, std::milli>(end_numa - start_numa).count();

    auto start_huge = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; ++i)
    {
        void *ptr = allocateHugePages(size_bytes);
        if (ptr == nullptr)
        {
            continue;
        }
        std::memset(ptr, 0, size_bytes);
        std::free(ptr);
    }
    auto end_huge = std::chrono::high_resolution_clock::now();
    bench.hugepage_time_ms =
        std::chrono::duration<double, std::milli>(end_huge - start_huge).count();

    bench.speedup = finiteBenchmarkSpeedup(bench.regular_time_ms, bench.numa_time_ms);
    return bench;
}

NumaSpeedupEstimate estimateNumaSpeedup(size_t matrix_size, int num_sockets)
{
    NumaSpeedupEstimate estimate{};
    if (num_sockets <= 1)
    {
        estimate.locality_speedup = kBaselineSpeedup;
        estimate.bandwidth_speedup = kBaselineSpeedup;
        estimate.hugepage_speedup = kBaselineSpeedup;
        estimate.total_speedup = kBaselineSpeedup;
        return estimate;
    }

    const double sockets = static_cast<double>(num_sockets);
    const double size_mb = static_cast<double>(matrix_size) / static_cast<double>(kBytesPerMb);

    estimate.locality_speedup = std::min(1.0 + 0.15 * sockets, 2.0);
    estimate.bandwidth_speedup = std::min(1.0 + 0.10 * (sockets - 1.0), 1.6);
    estimate.hugepage_speedup = size_mb >= 128.0 ? 1.15 : 1.03;
    estimate.total_speedup =
        estimate.locality_speedup * estimate.bandwidth_speedup * estimate.hugepage_speedup;
    return estimate;
}

} // namespace nerve::persistence::numa
