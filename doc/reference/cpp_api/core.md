# `nerve::core` -- Thread Pools

### ThreadPool

Pinned thread pool with `pthread_setaffinity_np`. Each worker is pinned to a specific core.

```cpp
#include <nerve/core/thread_affinity.hpp>

namespace nerve::core;

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0,    // 0 = hardware_concurrency
                        bool pin_to_cores = true,
                        bool use_numa = false);
    ~ThreadPool();

    void enqueue(std::function<void(int thread_id)> task);
    void wait();
    Size threadCount() const;
    int threadToCpu(int thread_id) const;
};
```

**Cost:** O(1) enqueue. Threads created on construction, joined on destruction.

### NumaAwareThreadPool

Thread pool with per-node task dispatch.

```cpp
class NumaAwareThreadPool {
public:
    explicit NumaAwareThreadPool(int threads_per_node = -1);
    ~NumaAwareThreadPool();

    void enqueueOnNode(int numa_node, std::function<void()> task);
    void waitAll();
    Size nodeCount() const;
};
```

### Thread Affinity Utilities

```cpp
void pinCurrentThreadToCore(int cpu_id);
void pinCurrentThreadToPackage(int package_id);
void pinCurrentThreadToNumaNode(int numa_node);
void pinThreadToCore(std::thread& t, int cpu_id);
int getCurrentCpu();
int getCurrentNumaNode();

struct CpuTopology {
    int num_packages;
    int num_cores;
    int num_threads;
    int numa_nodes;
    std::vector<int> core_to_numa;
    std::vector<int> thread_to_core;

    int packageOf(int cpu_id) const;
    int numaNodeOf(int cpu_id) const;
    int coreOf(int cpu_id) const;
};

CpuTopology detectCpuTopology();
```

**Cost:** O(n_cores) to detect topology (reads /proc/cpuinfo or sysfs). O(1) for query functions.

### EnhancedThreadPool

Future-based thread pool with `submit()` / `waitForAll()`.

```cpp
#include <nerve/core/memory/thread_safe_memory_pool.hpp>

namespace nerve::core;

class EnhancedThreadPool {
public:
    explicit EnhancedThreadPool(Size numThreads = 0);
    ~EnhancedThreadPool();

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<invoke_result_t>;

    void waitForAll();
    Size numThreads() const noexcept;
};
```

**Example:**

```cpp
nerve::core::EnhancedThreadPool pool(4);
auto future = pool.submit([] { return 42; });
int result = future.get();  // blocks until done
pool.waitForAll();
```

<- [C++ API Overview](index.md)
