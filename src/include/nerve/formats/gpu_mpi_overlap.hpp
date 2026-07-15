#pragma once

#include "nerve/formats/packed_boundary_matrix.hpp"
#include "nerve/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace nerve::formats::mpi
{

struct OverlapConfig
{
    Size chunk_size = 10000;
    Size max_streams = 2;
    Size pinned_buffer_size = 256ULL * 1024 * 1024;
    bool use_gpu_direct = false;
    bool use_nvshmem = false;
    int communicator_handle = 0;
};

struct OverlapStats
{
    double compute_time_ms = 0.0;
    double comm_time_ms = 0.0;
    double total_time_ms = 0.0;
    Size chunks_processed = 0;
    Size bytes_transferred = 0;
    double overlap_efficiency = 0.0;
};

#if defined(NERVE_HAS_CUDA) && defined(NERVE_HAS_MPI)

#include <cuda_runtime.h>
#include <mpi.h>

class GpuMpiOverlapPipeline
{
public:
    explicit GpuMpiOverlapPipeline(const OverlapConfig &config);
    ~GpuMpiOverlapPipeline();

    GpuMpiOverlapPipeline(const GpuMpiOverlapPipeline &) = delete;
    GpuMpiOverlapPipeline &operator=(const GpuMpiOverlapPipeline &) = delete;

    using ChunkComputeFn = std::function<void(const Index *chunk_cols, Size chunk_size,
                                              Size chunk_offset, cudaStream_t stream)>;

    using ChunkReduceFn =
        std::function<void(const std::vector<Index> &chunk_cols, Size chunk_offset)>;

    errors::ErrorResult<void> executeOverlapped(const GpuPackedLayout &layout,
                                                const std::vector<Index> &column_order,
                                                ChunkComputeFn gpu_compute_fn,
                                                ChunkReduceFn cpu_reduce_fn, int device_id);

    [[nodiscard]] const OverlapStats &stats() const { return stats_; }

    [[nodiscard]] bool isCudaAwareMpi() const;

    errors::ErrorResult<void> sendPackedColumns(const PackedWord *d_data, Size n_words,
                                                Size n_columns, int dest_rank, int tag,
                                                cudaStream_t stream);

    errors::ErrorResult<void> recvPackedColumns(PackedWord *d_data, Size n_words, Size n_columns,
                                                int src_rank, int tag, cudaStream_t stream);

    errors::ErrorResult<void> allgatherPivots(const Index *d_pivots, Size n_pivots,
                                              std::vector<Index> &out_pivots, cudaStream_t stream);

private:
    OverlapConfig config_;
    OverlapStats stats_;

    cudaStream_t compute_stream_;
    cudaStream_t comm_stream_;
    cudaEvent_t compute_done_;
    cudaEvent_t comm_done_;

    PackedWord *pinned_send_buf_;
    PackedWord *pinned_recv_buf_;
    Size pinned_capacity_;

    int world_rank_;
    int world_size_;
    bool cuda_aware_;

    void ensurePinnedCapacity(Size required_words);
};

} // namespace nerve::formats::mpi

#endif // NERVE_HAS_CUDA && NERVE_HAS_MPI
