# CUDA-aware MPI

When MPI supports CUDA awareness (`MPIX_Query_cuda_support`), GPU buffers are passed directly to MPI calls -- zero copies:

```cpp
// Zero-copy GPU send (CUDA-aware MPI)
MPI_Send(gpu_buffer, count, MPI_FLOAT, target_rank, tag, comm);

// Fallback: stage through pinned host memory
if (!is_cuda_aware_mpi()) {
    cudaMemcpy(host_buffer, gpu_buffer, bytes, cudaMemcpyDeviceToHost);
    MPI_Send(host_buffer, count, MPI_FLOAT, target_rank, tag, comm);
}
```

### CUDA-aware MPI detection

```cpp
// src/include/nerve/distributed/cuda_aware_mpi.hpp
bool is_cuda_aware_mpi();
// Returns true when MPI implementation supports GPU buffers directly
```

Automatic fallback to host staging when CUDA awareness is unavailable -- no code changes needed.

### CUDA-aware MPI setup

```bash
# OpenMPI + CUDA-aware configuration
export OMPI_MCA_btl=^openib    # disable InfiniBand BTL
export OMPI_MCA_pml=ob1        # use ob1 PML
export CUDA_aware_MPI=1        # enable CUDA awareness check

# MPICH + CUDA-aware configuration
export MPICH_GPU_SUPPORT_ENABLED=1
```

### API for CUDA-aware MPI

```cpp
// src/include/nerve/distributed/cuda_aware_mpi.hpp
namespace nerve::distributed {

// Send GPU buffer directly (zero-copy if CUDA-aware MPI)
void mpi_send_device(const float* d_data, int count, int dest, int tag, MPI_Comm comm);
void mpi_recv_device(float* d_data, int count, int source, int tag, MPI_Comm comm);
void mpi_bcast_device(float* d_data, int count, int root, cudaStream_t stream, MPI_Comm comm);
void mpi_allreduce_device(const float* d_send, float* d_recv, int count,
                           MPI_Op op, cudaStream_t stream, MPI_Comm comm);
void mpi_allgather_device(const float* d_send, int send_count,
                           float* d_recv, cudaStream_t stream, MPI_Comm comm);

// Explicit staging to pinned host memory
void stage_to_host(const float* d_src, float* h_dst, size_t bytes, cudaStream_t stream);
void stage_to_device(const float* h_src, float* d_dst, size_t bytes, cudaStream_t stream);
}
```

<- [Distributed Computing Overview](index.md)
