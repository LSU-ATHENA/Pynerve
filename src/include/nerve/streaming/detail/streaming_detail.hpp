#pragma once

#include "nerve/core_types.hpp"
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/gpu_streaming.hpp"
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/lock_free_streaming.hpp"
#include "nerve/streaming/streaming_laplacian.hpp"
#include "nerve/streaming/windowed_ph.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

namespace nerve::streaming
{

void batchVectorAddSimd(double *a, const double *b, size_t n);
void batchScaleSimd(double *data, double scale, size_t n);
void batchThresholdSimd(double *data, size_t n, double low, double high);

class StreamingLaplacian
{
public:
    explicit StreamingLaplacian(size_t max_vertices);
    void addEdge(int u, int v, double weight);
    std::vector<double> computeEigenvalues();
};

} // namespace nerve::streaming

namespace nerve::streaming::lockfree
{

template <typename T>
class LockFreeSPSCQueue
{
public:
    explicit LockFreeSPSCQueue(size_t capacity)
        : capacity_(capacity)
        , buffer_(capacity)
    {}
    bool push(const T &item)
    {
        if (size() >= capacity_)
            return false;
        buffer_[write_pos_ % capacity_] = item;
        ++write_pos_;
        return true;
    }
    std::optional<T> pop()
    {
        if (empty())
            return std::nullopt;
        T item = buffer_[read_pos_ % capacity_];
        ++read_pos_;
        return item;
    }
    bool empty() const { return read_pos_ >= write_pos_; }
    size_t size() const { return write_pos_ - read_pos_; }
    bool isFull() const { return size() >= capacity_; }

private:
    size_t capacity_;
    std::atomic<size_t> write_pos_{0};
    size_t read_pos_{0};
    std::vector<T> buffer_;
};

} // namespace nerve::streaming::lockfree

#ifdef NERVE_HAS_CUDA

namespace nerve::streaming::gpu
{

struct MultiStreamContext
{
    static cudaError_t create(int num_streams, int device_id, MultiStreamContext &ctx);
    static void destroy(MultiStreamContext &ctx);
};

} // namespace nerve::streaming::gpu

#endif
