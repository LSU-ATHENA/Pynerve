# NUMA affinity

Pynerve pins worker threads to cores respecting NUMA topology:

<img src="../img/performance_numa.svg" alt="NUMA socket layout" width="90%">

### Thread pool

The `ThreadPool` (`src/core/`) uses:

1. **Thread affinity**: `pthread_setaffinity_np` pinning to specific cores
2. **Per-core work queues**: Lock-free queues per core, work stealing via CAS
3. **NUMA-local allocation**: Large arrays allocated on the local NUMA node via `mbind`

```python
# Thread count and affinity
result = pynerve.compute_persistence(
    points,
    max_dim=2,
    threads=16,       # 0 = auto-detect (default)
)
```

Thread count is auto-detected from `/sys/devices/system/cpu/`. On NUMA systems, Pynerve creates one thread per physical core (not hyperthread).

### NUMA thread pinning protocol

```
1. Detect NUMA topology:
   - num_nodes: number of NUMA nodes (from libnuma or /sys)
   - cores_per_node: physical cores per node
   - node_of_core[c]: maps core id to NUMA node

2. Create thread pool:
   - Allocate threads = min(requested, physical_cores)
   - For i in 0..threads-1:
       core = i % (num_nodes * cores_per_node)
       node = node_of_core[core]
       Thread i: pthread_setaffinity_np to core
                 memory allocations via mbind to node

3. Work distribution:
   - Stage 1 (distance compute): thread-local slabs,
         each thread allocates from its NUMA-local memory
   - Stage 2 (reduction): work-stealing across all threads,
         but memory is already local to its NUMA node

4. Stealing across NUMA nodes:
   - If thread steals work from remote node:
         Access remote memory (higher latency)
         Poll remote queue via CAS
   - Backoff: spin 100 cycles, then pause instruction
```

[Back to index](index.md)
