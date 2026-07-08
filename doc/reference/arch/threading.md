# Thread Model

### ThreadPool

Pinned thread pool with `nerve::sys::thread_set_affinity()`. Each worker thread is pinned to a specific core for cache locality and consistent scheduling. Round-robin core assignment by default.

```cpp
nerve::core::ThreadPool pool(/* threads = */ 8, /* pin_to_cores = */ true);
pool.enqueue([](int thread_id) {
    // thread_id maps to a specific core
});
pool.wait();
```

Constructor parameters: `num_threads` (0 = `hardware_concurrency()`), `pin_to_cores` (default `true`), `use_numa` (default `false`).

### NumaAwareThreadPool

NUMA-aware thread pool that dispatches tasks to threads on a specific NUMA node. Each node has its own thread pool. Used for workloads where memory affinity matters.

```cpp
nerve::core::NumaAwareThreadPool pool(/* threads_per_node = */ 4);
pool.enqueueOnNode(/* numa_node = */ 0, [](/* ... */));
pool.waitAll();
```

### CpuTopology Detection

The `detectCpuTopology()` function probes `/proc/cpuinfo` (Linux) or OS APIs to build a topology map: packages, cores per package, threads per core, NUMA nodes per package. Used by thread pools for pinning decisions.

```cpp
auto topo = nerve::core::detectCpuTopology();
int core = topo.coreOf(/* cpu_id */);
int node = topo.numaNodeOf(/* cpu_id */);
```

### Thread Affinity Utilities

```cpp
void pinCurrentThreadToCore(int cpu_id);
void pinCurrentThreadToPackage(int package_id);
void pinCurrentThreadToNumaNode(int numa_node);
void pinThreadToCore(std::thread& t, int cpu_id);
int getCurrentCpu();
int getCurrentNumaNode();
```

### EnhancedThreadPool

Alternative thread pool with `std::future`-based `submit()` and `waitForAll()`. Thread-safe task queue with condition variable.

```cpp
nerve::core::EnhancedThreadPool pool(4);
auto fut = pool.submit([] { return 42; });
int result = fut.get();
pool.waitForAll();
```


[Back to Architecture Index](index.md)
