## NUMA-aware allocator

```cpp
namespace nerve::memory {

class NumaAwareAllocator {
public:
    explicit NumaAwareAllocator(int preferred_node = -1);

    void* allocate(Size bytes, Size alignment = alignof(std::max_align_t));
    void deallocate(void* ptr, Size bytes);

    int getPreferredNode() const;
    void setPreferredNode(int node);
};

}
```

Uses `numa_alloc_onnode` to allocate memory on a specific NUMA node. Falls
back to aligned `malloc` when NUMA is unavailable.


## NUMA-aware memory pool (advanced)

```cpp
namespace nerve::core {

enum class NumaPolicy { AUTO, LOCAL, INTERLEAVE, PREFERRED, BIND };

struct NumaPoolConfig {
    NumaPolicy policy = NumaPolicy::AUTO;
    int preferredNode = -1;
    size_t poolSizePerNode = 64 * 1024 * 1024;
    bool enableNumaBinding = false;
    bool enableHotPathTracking = false;
};

class NumaAwareMemoryPool {
public:
    explicit NumaAwareMemoryPool(const NumaPoolConfig& config);

    void* allocate(size_t size, size_t alignment);
    void* allocateOnNode(size_t size, int node_id, size_t alignment);
    void deallocate(void* ptr, size_t size);

    int getCurrentNumaNode();
    bool bindToNode(int node);
    void setNumaPolicy(NumaPolicy policy, int preferred_node);

    size_t totalAllocated() const;
    size_t totalCapacity() const;
    void defragmentAllPools();
    void resetAllPools();

    bool isDeterministic() const;
    void setDeterminismContract(const DeterminismContract& contract);
};

class NumaPoolManager {
public:
    static NumaPoolManager& instance();
    NumaAwareMemoryPool& getGlobalPool();
    NumaAwareMemoryPool& getPoolForThread(std::thread::id id);
    void configureGlobalPool(const NumaPoolConfig& config);
    bool bindCurrentThreadToNode(int node);
    std::vector<NumaNodeInfo> getNumaTopology() const;
    size_t getTotalAllocated() const;
};

}
```


### Policies

Five NUMA policies are available. **`AUTO`** allocates on the current node. **`LOCAL`** allocates on the thread's NUMA node. **`INTERLEAVE`** distributes round-robin across all nodes. **`PREFERRED`** allocates on the preferred node with fallback to others. **`BIND`** allocates only on the specified node and fails if out of memory.


### When to use NUMA

NUMA recommendations vary by system type. For single socket systems there is no benefit from NUMA -- skip it. For dual socket systems, use NUMA for memory-intensive workloads. For 4+ socket systems, NUMA is strongly recommended for persistence computation. For HPC or large memory systems, use INTERLEAVE for distance matrices.

**Performance impact (dual-socket):**
- Local allocation + local access: 1.0x (baseline)
- Remote allocation + local access: 1.3x slower
- Local allocation + remote access: 1.5x slower
- Remote allocation + remote access: 2.0x slower


### Python

```python
import pynerve.memory as mem

alloc = mem.NumaAwareAllocator(preferred_node=0)

# Check NUMA topology
from pynerve.core import cpu_features
topology = cpu_features().numa_topology
print(f"NUMA nodes: {len(topology.nodes)}")
for node in topology.nodes:
    print(f"  Node {node.id}: cores {node.cores}")
```

## FAQ

**Does NUMA help on single-socket systems?**
No. On a single-socket system all memory is local, so NUMA-aware allocation adds complexity with no benefit. Skip NUMA entirely on single-socket hardware.

**What happens if NUMA is not available?**
`NumaAwareAllocator` falls back to aligned `malloc` when libnuma is unavailable. Always check NUMA availability at runtime before relying on NUMA semantics.

**What is the performance cost of remote NUMA access?**
On a dual-socket system, remote allocation with local access is ~1.3x slower than local, local allocation with remote access is ~1.5x slower, and remote allocation with remote access is ~2.0x slower than the local-local baseline.


### Cross-references

- `pynerve.core.thread_pool`: NUMA-aware thread pool
- `pynerve.memory.memory`: Memory management overview
- `pynerve.memory.raw_array_pool`: Raw array pool (NUMA-aware variant)
