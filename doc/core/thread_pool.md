## Thread pool

Per-core pinned worker threads using `pthread_setaffinity_np`. Workers are
created once and reused across submissions. Work is distributed via a
lock-free work-stealing queue.

```cpp
ThreadPool pool(8);  // 8 threads pinned to cores 0-7

auto fut = pool.submit([](int x) { return x * x; }, 42);
int result = fut.get();

std::vector<int> data(1000);
pool.map([&](int& v) { v *= 2; }, data.begin(), data.end());
```


### Configurable scheduling

- `submit()` -- single task, returns `std::future`
- `map()` -- parallel-for over iterator range
- `shutdown()` -- join all threads (RAII on destruction)

```python
from pynerve.core import ThreadPool

pool = ThreadPool(num_threads=8, pin_threads=True)
results = pool.map(process_item, items)
pool.shutdown()
```


### Lock-free work-stealing

Each worker thread maintains a deque of tasks:

```cpp
struct WorkerQueue {
    std::deque<std::function<void()>> tasks;
    std::atomic<size_t> head, tail;
};

thread_local WorkerQueue local_queue;

void worker_main(int thread_id) {
    pin_thread_to_core(thread_id);

    while (!stop_flag) {
        // Try local queue first
        if (auto task = local_queue.pop_front()) {
            task();
            continue;
        }

        // Steal from a random victim
        for (int attempt = 0; attempt < STEAL_ATTEMPTS; ++attempt) {
            int victim = random() % num_threads;
            if (auto task = steal(victim)) {
                task();
                break;
            }
        }

        // No work -> spin briefly, then sleep
        if (++idle_count > IDLE_LIMIT) {
            std::this_thread::yield();
        }
    }
}
```


### Thread pinning strategy

1. Detect topology (cores, packages, NUMA nodes)
2. Assign threads to physical cores first (skip HT siblings)
3. Fill packages round-robin
4. If more threads than physical cores, assign to HT siblings

```text
System: 2 sockets x 8 cores x 2 HT = 32 logical cores
ThreadPool(8): pins to cores 0-7 (all physical, socket 0)
ThreadPool(16): pins to cores 0-15 (physical across both sockets)
ThreadPool(32): pins to cores 0-31 (all logical)
```


### NUMA-aware thread pool

Multi-socket systems get per-NUMA-node thread pools. Work submitted to
`map_node()` runs on threads pinned to that node's cores, keeping memory
access local.

```cpp
NumaThreadPool pool(4);  // 4 threads per NUMA node

pool.map_node(0, [](const auto& item) { process_on_node(item); },
              data.begin(), data.end());
```

Each pool maintains separate work queues to avoid cross-node contention.
Detected via `libnuma` at build time.

```python
from pynerve.core import NumaThreadPool

pool = NumaThreadPool(threads_per_node=4)
print(f"NUMA nodes: {pool.get_node_count()}")

# Process data on node 0
pool.map_node(0, process_fn, data)
```


### Performance characteristics

With 8 threads pinned on an 8-core system, throughput reaches 7.8x with roughly one microsecond submit-to-start latency. With 8 threads unpinned, throughput drops to 6.5x with around 5 microseconds latency. Using 16 threads with hyperthreading pinned on the same 8-core system achieves 12x throughput with about 2 microseconds latency. Cross-NUMA-node access yields 0.8x bandwidth with an additional 200 nanoseconds latency.


### Common pitfalls

1. **Pin mismatch**: Pinning thread 0 to a core already running the main
   thread causes contention. Pin workers to cores NOT used by the main
   thread.

2. **NUMA allocation**: Memory allocated on NUMA node 0 but accessed from
   threads pinned to node 1 has ~1.5x higher latency. Use `NumaThreadPool`
   with matching allocation.

3. **Oversubscription**: More threads than cores does not improve
   throughput for compute-bound tasks.

4. **Blocking tasks**: If tasks block (I/O, mutex), use more threads than
   cores to hide latency.


### Cross-references

- `pynerve.core.core`: Topology detection
- `pynerve.memory.numa`: NUMA-aware memory allocation
- `pynerve.algorithms.distance`: Uses thread pool for batch computation
- `pynerve.dmt.parallel`: Parallel DMT uses thread pool
