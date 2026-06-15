# `nerve::distributed` -- MPI Persistence

```cpp
#include <nerve/persistence/distributed/mpi_distributed_ph.hpp>

namespace nerve::persistence::distributed;

struct DistributedConfig {
    Size max_dim = 2;
    double max_radius = 1.0;
    double overlap_ratio = 0.1;
    bool use_openmp = true;
    bool use_cuda = false;
};

struct DistributedResult {
    std::vector<Pair> pairs;
    double cover_time_ms;
    double local_computation_time_ms;
    double communication_time_ms;
    double total_time_ms;
    int mpi_rank;
    int mpi_size;
    size_t total_points;
    size_t points_per_rank;
    double estimated_speedup;
};

DistributedResult computeDistributedPH(
    const std::vector<double>& points,
    size_t point_dim,
    const DistributedConfig& config
);

void initializeDistributed();
void finalizeDistributed();
DistributedConfig getOptimalDistributedConfig(size_t num_points, size_t point_dim, int num_ranks);
bool shouldUseDistributed(size_t num_points, int available_cores);
DistributedSystemInfo getDistributedSystemInfo();
```

**Cost (computeDistributedPH):** O(n^2 / p * d) distance + O(m^3 / p) reduction + O(p) communication.

### NCCL/MPI Bridge

```cpp
#include <nerve/distributed/nccl_mpi_bridge.hpp>

namespace nerve::distributed;

class NcclMpiBridge {
public:
    NcclMpiBridge(MPI_Comm mpi_comm, int gpus_per_node);

    void allreduce(const float* send, float* recv, int count,
                   ncclRedOp_t op = ncclSum);
    void allgather(const float* send, float* recv, int count);

    int worldRank() const;
    int worldSize() const;
};
```

**Cost (allreduce):** O(count) communication, NCCL-optimized for NVLink.

### CUDA-Aware MPI Utilities

```cpp
#include <nerve/distributed/cuda_aware_mpi.hpp>

namespace nerve::distributed;

bool is_cuda_aware_mpi();

void mpi_send_device(const float* d_data, int count, int dest, int tag, MPI_Comm comm);
void mpi_recv_device(float* d_data, int count, int source, int tag, MPI_Comm comm);
void mpi_bcast_device(float* d_data, int count, int root, cudaStream_t stream, MPI_Comm comm);
void mpi_allreduce_device(const float* d_send, float* d_recv, int count,
                           MPI_Op op, cudaStream_t stream, MPI_Comm comm);
void mpi_allgather_device(const float* d_send, int send_count,
                           float* d_recv, cudaStream_t stream, MPI_Comm comm);
```

**Cost:** Zero-copy when CUDA-aware MPI is available. O(bytes) copy via host when not.

<- [C++ API Overview](index.md)
