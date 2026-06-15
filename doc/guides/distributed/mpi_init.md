# MPI initialization and rank discovery

Pynerve initializes MPI lazily on first use. The communicator wraps `MPI_Comm_World`:

```cpp
// src/distributed/mpi_communicator.cpp
class MPICommunicator {
    int rank() const;
    int size() const;
    void broadcast(void* buf, int count, MPI_Datatype type, int root);
    void allgather(const void* send, int sendcount, void* recv,
                   int recvcount, MPI_Datatype type);
    void allgatherv(const void* send, int sendcount, void* recv,
                    const int* recvcounts, const int* displs, MPI_Datatype type);
    void barrier();
    void reduce(const void* send, void* recv, int count,
                MPI_Datatype type, MPI_Op op, int root);
    void allreduce(const void* send, void* recv, int count,
                   MPI_Datatype type, MPI_Op op);
    // Non-blocking variants with isend/irecv/wait/waitall
    MPI_Request isend(const void* buf, int count, MPI_Datatype type, int dest, int tag);
    MPI_Request irecv(void* buf, int count, MPI_Datatype type, int source, int tag);
    void wait(MPI_Request* req);
    void waitall(int count, MPI_Request* reqs);
};
```

```python
# Python-level rank discovery
# Pynerve exposes rank/size via the computation result
result = pynerve.distributed_persistence(points, max_dim=2)
diag = result.diagnostics
# diag includes rank, world_size, total_points, points_per_rank
```

### MPI communicator setup

```cpp
// src/distributed/mpi_communicator.cpp
// Lazy initialization on first MPI call
void ensure_initialized() {
    static bool initialized = false;
    if (!initialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            // Fall back to MPI_THREAD_SERIALIZED
        }
        initialized = true;
    }
}
```

### Communicator splitting

For hierarchical parallelism, Pynerve supports communicator splitting:

```cpp
// Split world into sub-communicators by node
int node_key = get_node_id();  // derived from hostname
MPI_Comm node_comm;
MPI_Comm_split(MPI_COMM_WORLD, node_key, rank, &node_comm);

// node_comm connects GPUs within a node (for NCCL)
// MPI_COMM_WORLD connects across nodes (for MPI)
```

<- [Distributed Computing Overview](index.md)
