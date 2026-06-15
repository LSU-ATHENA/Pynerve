# NVSHMEM bridge

For GPU-clusters with NVSHMEM, Pynerve provides direct GPU-GPU communication without CPU involvement:

```cpp
// src/distributed/nvshmem_bridge.cpp
class NvshmemBridge {
    void init(int* argc, char*** argv, int n_gpus_per_node);
    void finalize();
    void barrier();
    void* symmetric_malloc(size_t bytes);
    void put(void* dest, const void* source, size_t bytes, int pe);
    void get(void* dest, const void* source, size_t bytes, int pe);
    void reduce(void* dest, const void* source, size_t n, NvshmiOp op);
};
```

NVSHMEM enables one-sided `put`/`get` between GPU memories -- useful for column stealing and pivot exchange without MPI matching overhead.

### NVSHMEM column stealing

```cpp
// Column stealing using NVSHMEM put/get
void steal_column(NvshmemBridge& bridge, int victim_rank,
                  Column& stolen_column) {
    // 1. Victim's column location is in symmetric memory
    int64_t* avail_flag = static_cast<int64_t*>(
        bridge.symmetric_malloc(sizeof(int64_t)));

    // 2. Check if victim has available work via one-sided get
    int64_t has_work = 0;
    bridge.get(&has_work, avail_flag, sizeof(int64_t), victim_rank);

    // 3. If available, copy column via one-sided get
    if (has_work) {
        bridge.get(stolen_column.data(), victim_column_ptr,
                   stolen_column.size(), victim_rank);
    }
}
```

<- [Distributed Computing Overview](index.md)
