# Multi-GPU + MPI hybrid

NCCL handles GPU collectives within a node; MPI coordinates across nodes:

<img src="../../img/distributed_hybrid.svg" alt="Multi-GPU + MPI hybrid diagram" width="90%">

```cpp
// src/include/nerve/distributed/nccl_mpi_bridge.hpp
class NcclMpiBridge {
    void allreduce(const float* input, float* output, size_t n);
    void broadcast(const float* input, size_t n, int root);
    void allgather(const float* input, float* output, size_t n);
    void cross_node_reduce(const float* input, float* output, size_t n);
};
```

`cross_node_reduce` uses NCCL within node then MPI Allreduce across nodes -- optimal for heterogeneous interconnects.

### Hybrid reduction protocol

```
cross_node_reduce(input, output, n):

1. Intra-node NCCL Allreduce:
   ncclAllReduce(input, node_local, n, ncclFloat, ncclSum,
                 nccl_comm[node_id], stream);

2. Inter-node MPI Allreduce (one rank per node participates):
   if (node_local_rank == 0):
       MPI_Allreduce(node_local, global, n, MPI_FLOAT, MPI_SUM, node_comm);

3. Intra-node NCCL Broadcast (root = node local rank 0):
   ncclBroadcast(global, output, n, ncclFloat, 0,
                 nccl_comm[node_id], stream);
```

### NCCL + MPI hybrid pattern

```cpp
// Complete hybrid setup
void setup_hybrid(MPI_Comm world_comm) {
    int world_rank, world_size;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    // Split into node-level communicators
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    int node_rank, node_size;
    MPI_Comm node_comm;
    MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED,
                        world_rank, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // NCCL communicator per node
    ncclUniqueId nccl_id;
    if (node_rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, node_comm);

    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, node_size, nccl_id, node_rank);

    // Cross-node communicator (one rank per node)
    int color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm cross_node_comm;
    MPI_Comm_split(world_comm, color, world_rank, &cross_node_comm);
}
```

<- [Distributed Computing Overview](index.md)
