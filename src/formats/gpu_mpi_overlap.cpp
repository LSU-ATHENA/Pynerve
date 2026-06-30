#include "nerve/distributed/cuda_aware_mpi.hpp"
#include "nerve/formats/gpu_mpi_overlap.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nerve::formats::mpi
{

namespace
{

double wallClockMs()
{
    using Clock = std::chrono::high_resolution_clock;
    static const Clock::time_point t0 = Clock::now();
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

Size safeMul(Size a, Size b) noexcept
{
    constexpr Size max_val = std::numeric_limits<Size>::max();
    if (a != 0 && b > max_val / a)
        return max_val;
    return a * b;
}

} // namespace

GpuMpiOverlapPipeline::GpuMpiOverlapPipeline(const OverlapConfig &config)
    : config_(config)
    , compute_stream_(nullptr)
    , comm_stream_(nullptr)
    , compute_done_(nullptr)
    , comm_done_(nullptr)
    , pinned_send_buf_(nullptr)
    , pinned_recv_buf_(nullptr)
    , pinned_capacity_(0)
    , world_rank_(0)
    , world_size_(1)
    , cuda_aware_(false)
{
    MPI_Comm comm = (config_.communicator_handle != 0) ? MPI_Comm_f2c(config_.communicator_handle)
                                                       : MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &world_rank_);
    MPI_Comm_size(comm, &world_size_);

    cuda_aware_ = nerve::distributed::is_cuda_aware_mpi();

    cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&comm_stream_, cudaStreamNonBlocking);
    cudaEventCreate(&compute_done_);
    cudaEventCreate(&comm_done_);
}

GpuMpiOverlapPipeline::~GpuMpiOverlapPipeline()
{
    if (pinned_send_buf_)
        cudaFreeHost(pinned_send_buf_);
    if (pinned_recv_buf_)
        cudaFreeHost(pinned_recv_buf_);
    if (compute_stream_)
        cudaStreamDestroy(compute_stream_);
    if (comm_stream_)
        cudaStreamDestroy(comm_stream_);
    if (compute_done_)
        cudaEventDestroy(compute_done_);
    if (comm_done_)
        cudaEventDestroy(comm_done_);
}

bool GpuMpiOverlapPipeline::isCudaAwareMpi() const
{
    return cuda_aware_;
}

void GpuMpiOverlapPipeline::ensurePinnedCapacity(Size required_words)
{
    if (pinned_capacity_ >= required_words)
        return;

    if (pinned_send_buf_)
    {
        cudaFreeHost(pinned_send_buf_);
        pinned_send_buf_ = nullptr;
    }
    if (pinned_recv_buf_)
    {
        cudaFreeHost(pinned_recv_buf_);
        pinned_recv_buf_ = nullptr;
    }

    cudaHostAlloc(&pinned_send_buf_, required_words * kPackedWordBytes, cudaHostAllocDefault);
    cudaHostAlloc(&pinned_recv_buf_, required_words * kPackedWordBytes, cudaHostAllocDefault);

    pinned_capacity_ = required_words;
}

errors::ErrorResult<void> GpuMpiOverlapPipeline::executeOverlapped(
    const GpuPackedLayout &layout, const std::vector<Index> &column_order,
    ChunkComputeFn gpu_compute_fn, ChunkReduceFn cpu_reduce_fn, int device_id)
{
    const double t_start = wallClockMs();
    double t_compute = 0.0;
    double t_comm = 0.0;
    stats_ = OverlapStats{};

    cudaSetDevice(device_id);

    const Size n_cols = layout.num_columns;
    const Size n_words = layout.max_words_per_column;
    if (n_cols == 0 || n_words == 0)
        return errors::ErrorResult<void>::success();

    const Size chunk_size = std::min(config_.chunk_size, n_cols);
    const Size num_chunks = (n_cols + chunk_size - 1) / chunk_size;
    const Size chunk_words = safeMul(chunk_size, n_words);

    PackedWord *d_columns_device = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_columns_device, layout.total_packed_bytes);
    if (alloc_err != cudaSuccess || d_columns_device == nullptr)
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);

    cudaMemcpyAsync(d_columns_device, layout.columns_flat.data(), layout.total_packed_bytes,
                    cudaMemcpyHostToDevice, compute_stream_);
    cudaStreamSynchronize(compute_stream_);

    ensurePinnedCapacity(chunk_words * 2);

    PackedWord *ping = pinned_send_buf_;
    PackedWord *pong = pinned_send_buf_ + chunk_words;
    std::vector<MPI_Request> send_reqs(num_chunks, MPI_REQUEST_NULL);
    Size active_sends = 0;

    for (Size c = 0; c < num_chunks; ++c)
    {
        const Size offset = c * chunk_size;
        const Size current_chunk = std::min(chunk_size, n_cols - offset);

        std::vector<Index> chunk_indices;
        chunk_indices.reserve(current_chunk);
        for (Size i = 0; i < current_chunk; ++i)
        {
            Size global_idx = offset + i;
            if (global_idx < column_order.size())
                chunk_indices.push_back(column_order[global_idx]);
            else
                chunk_indices.push_back(static_cast<Index>(global_idx));
        }

        const double t_comp_start = wallClockMs();

        if (gpu_compute_fn)
        {
            gpu_compute_fn(chunk_indices.data(), current_chunk, offset, compute_stream_);
        }

        cudaEventRecord(compute_done_, compute_stream_);

        if (world_size_ > 1)
        {
            cudaStreamWaitEvent(comm_stream_, compute_done_, 0);

            if (cuda_aware_)
            {
                PackedWord *d_chunk = d_columns_device + offset * n_words;
                const int send_bytes = static_cast<int>(current_chunk * n_words * kPackedWordBytes);
                MPI_Isend(d_chunk, send_bytes, MPI_BYTE, (world_rank_ + 1) % world_size_,
                          static_cast<int>(c), MPI_COMM_WORLD, &send_reqs[c]);
            }
            else
            {
                PackedWord *cur_pinned = (active_sends % 2 == 0) ? ping : pong;
                const Size words_to_copy = safeMul(current_chunk, n_words);
                cudaMemcpyAsync(cur_pinned, d_columns_device + offset * n_words,
                                words_to_copy * kPackedWordBytes, cudaMemcpyDeviceToHost,
                                comm_stream_);

                cudaEventRecord(comm_done_, comm_stream_);
                cudaStreamSynchronize(comm_stream_);

                const int send_bytes = static_cast<int>(words_to_copy * kPackedWordBytes);
                MPI_Isend(cur_pinned, send_bytes, MPI_BYTE, (world_rank_ + 1) % world_size_,
                          static_cast<int>(c), MPI_COMM_WORLD, &send_reqs[c]);
            }
            ++active_sends;
        }

        cudaStreamSynchronize(compute_stream_);
        t_compute += (wallClockMs() - t_comp_start);

        if (cpu_reduce_fn)
        {
            const double t_cpu_start = wallClockMs();
            cpu_reduce_fn(chunk_indices, offset);
            t_compute += (wallClockMs() - t_cpu_start);
        }

        if (active_sends > 0 && (c >= num_chunks - 1 || active_sends >= 2))
        {
            const double t_comm_start = wallClockMs();
            int completed = 0;
            for (Size s = 0; s <= c && completed < active_sends; ++s)
            {
                if (send_reqs[s] != MPI_REQUEST_NULL)
                {
                    MPI_Wait(&send_reqs[s], MPI_STATUS_IGNORE);
                    send_reqs[s] = MPI_REQUEST_NULL;
                    ++completed;
                }
            }
            t_comm += (wallClockMs() - t_comm_start);
            active_sends -= static_cast<Size>(completed);
        }
    }

    if (active_sends > 0)
    {
        const double t_comm_start = wallClockMs();
        MPI_Waitall(static_cast<int>(active_sends), send_reqs.data(), MPI_STATUSES_IGNORE);
        t_comm += (wallClockMs() - t_comm_start);
    }

    cudaFree(d_columns_device);

    stats_.compute_time_ms = t_compute;
    stats_.comm_time_ms = t_comm;
    stats_.total_time_ms = wallClockMs() - t_start;
    stats_.chunks_processed = num_chunks;
    stats_.bytes_transferred = safeMul(safeMul(n_cols, n_words), kPackedWordBytes);

    const double serial_time = t_compute + t_comm;
    if (serial_time > 0.0)
        stats_.overlap_efficiency = (serial_time - stats_.total_time_ms) / serial_time;
    else
        stats_.overlap_efficiency = 0.0;

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> GpuMpiOverlapPipeline::sendPackedColumns(const PackedWord *d_data,
                                                                   Size n_words, Size n_columns,
                                                                   int dest_rank, int tag,
                                                                   cudaStream_t stream)
{
    const Size total_words = safeMul(n_words, n_columns);
    if (total_words == 0)
        return errors::ErrorResult<void>::success();

    cudaStreamSynchronize(stream);

    if (cuda_aware_)
    {
        MPI_Send(d_data, static_cast<int>(total_words * kPackedWordBytes), MPI_BYTE, dest_rank, tag,
                 MPI_COMM_WORLD);
    }
    else
    {
        ensurePinnedCapacity(total_words);
        cudaMemcpy(pinned_send_buf_, d_data, total_words * kPackedWordBytes,
                   cudaMemcpyDeviceToHost);
        MPI_Send(pinned_send_buf_, static_cast<int>(total_words * kPackedWordBytes), MPI_BYTE,
                 dest_rank, tag, MPI_COMM_WORLD);
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> GpuMpiOverlapPipeline::recvPackedColumns(PackedWord *d_data, Size n_words,
                                                                   Size n_columns, int src_rank,
                                                                   int tag, cudaStream_t stream)
{
    const Size total_words = safeMul(n_words, n_columns);
    if (total_words == 0)
        return errors::ErrorResult<void>::success();

    if (cuda_aware_)
    {
        MPI_Recv(d_data, static_cast<int>(total_words * kPackedWordBytes), MPI_BYTE, src_rank, tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaStreamSynchronize(stream);
    }
    else
    {
        ensurePinnedCapacity(total_words);
        MPI_Recv(pinned_recv_buf_, static_cast<int>(total_words * kPackedWordBytes), MPI_BYTE,
                 src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaMemcpyAsync(d_data, pinned_recv_buf_, total_words * kPackedWordBytes,
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return errors::ErrorResult<void>::success();
}

errors::ErrorResult<void> GpuMpiOverlapPipeline::allgatherPivots(const Index *d_pivots,
                                                                 Size n_pivots,
                                                                 std::vector<Index> &out_pivots,
                                                                 cudaStream_t stream)
{
    out_pivots.clear();
    if (n_pivots == 0)
        return errors::ErrorResult<void>::success();

    cudaStreamSynchronize(stream);

    std::vector<Index> host_pivots(n_pivots);
    cudaMemcpy(host_pivots.data(), d_pivots, n_pivots * sizeof(Index), cudaMemcpyDeviceToHost);

    std::vector<Index> all_pivots(safeMul(n_pivots, static_cast<Size>(world_size_)));
    MPI_Allgather(host_pivots.data(), static_cast<int>(n_pivots * sizeof(Index)), MPI_BYTE,
                  all_pivots.data(), static_cast<int>(n_pivots * sizeof(Index)), MPI_BYTE,
                  MPI_COMM_WORLD);

    out_pivots = std::move(all_pivots);
    return errors::ErrorResult<void>::success();
}

} // namespace nerve::formats::mpi
