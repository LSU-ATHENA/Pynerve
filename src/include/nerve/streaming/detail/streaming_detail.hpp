#pragma once

#include "nerve/core_types.hpp"
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/gpu_streaming.hpp"
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/lock_free_streaming.hpp"
#include "nerve/streaming/streaming_laplacian.hpp"
#include "nerve/streaming/windowed_ph.hpp"

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
    explicit LockFreeSPSCQueue(size_t capacity);
    bool push(const T &item);
    std::optional<T> pop();
    bool empty() const;
    size_t size() const;
    bool isFull() const;
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
