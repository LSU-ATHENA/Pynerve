
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <vector>

#ifdef NERVE_HAS_CUDA
#include "nerve/streaming/gpu_streaming.hpp"
#endif
namespace nerve
{
namespace streaming
{
struct StreamingConfig
{
    size_t window_size = 1000;
    size_t stride = 100;
    double approximation_error_budget = 0.01;
    size_t max_points_per_window = 10000;
    bool enable_parallel_processing = true;
    size_t num_processing_threads = 4;
    double processing_interval_ms = 100.0;
};
struct StreamDataPoint
{
    std::vector<float> coordinates;
    int64_t timestamp_ns = 0;
    uint32_t point_id = 0;
    std::vector<float> attributes;
    float weight = 1.0f;
    bool isValid() const
    {
        return !coordinates.empty() && timestamp_ns > 0 && std::isfinite(weight) &&
               std::all_of(coordinates.begin(), coordinates.end(),
                           [](float value) { return std::isfinite(value); }) &&
               std::all_of(attributes.begin(), attributes.end(),
                           [](float value) { return std::isfinite(value); });
    }
};
struct StreamingPersistenceDiagram
{
    std::vector<std::pair<float, float>> points;
    int64_t window_start_time_ns = 0;
    int64_t window_end_time_ns = 0;
    size_t window_size = 0;
    double computation_time_ms = 0.0;
    double approximation_error = 0.0;
    bool is_approximate = false;
    bool isValid() const
    {
        return !points.empty() && window_start_time_ns < window_end_time_ns &&
               std::isfinite(computation_time_ms) && computation_time_ms >= 0.0 &&
               std::isfinite(approximation_error) && approximation_error >= 0.0 &&
               std::all_of(
                   points.begin(), points.end(),
                   [](const auto &point) {
                       return std::isfinite(point.first) &&
                              (std::isfinite(point.second) ||
                               point.second == std::numeric_limits<float>::infinity()) &&
                              point.second >= point.first;
                   });
    }
};
class ApproximateStreamingPH
{
public:
    explicit ApproximateStreamingPH(const StreamingConfig &config);
    void addDataPoint(const StreamDataPoint &point);
    void addDataPoints(const std::vector<StreamDataPoint> &points);
    StreamingPersistenceDiagram computeCurrentWindow();
    std::vector<StreamingPersistenceDiagram> computeRecentWindows(size_t num_windows);

private:
    StreamingConfig config_;
    std::queue<StreamDataPoint> data_buffer_;
    mutable std::mutex buffer_mutex_;
    std::vector<StreamDataPoint> current_window_;
    void updateSlidingWindow();
    StreamingPersistenceDiagram
    computePersistenceApproximate(const std::vector<StreamDataPoint> &window);
};
class ExactStreamingPH
{
public:
    explicit ExactStreamingPH(const StreamingConfig &config);
    void addDataPoint(const StreamDataPoint &point);
    void addDataPoints(const std::vector<StreamDataPoint> &points);
    StreamingPersistenceDiagram computeCurrentWindowExact();
    std::vector<StreamingPersistenceDiagram> computeRecentWindowsExact(size_t num_windows);

private:
    StreamingConfig config_;
    std::queue<StreamDataPoint> data_buffer_;
    mutable std::mutex buffer_mutex_;
    StreamingPersistenceDiagram computePersistenceExact(const std::vector<StreamDataPoint> &window);
};
class StreamingTDAManager
{
public:
    static StreamingTDAManager &instance();
    void setStreamingConfig(const StreamingConfig &config);
    std::shared_ptr<ApproximateStreamingPH> getApproximateProcessor();
    std::shared_ptr<ExactStreamingPH> getExactProcessor();
    void startStreamingProcessing();
    void stopStreamingProcessing();
    void addStreamData(const StreamDataPoint &point);
    void addStreamDataBatch(const std::vector<StreamDataPoint> &points);
    StreamingPersistenceDiagram getLatestPersistenceDiagram();
    std::vector<StreamingPersistenceDiagram> getRecentDiagrams(size_t count);

private:
    StreamingTDAManager() = default;
    StreamingConfig streaming_config_;
    std::shared_ptr<ApproximateStreamingPH> approximate_processor_;
    std::shared_ptr<ExactStreamingPH> exact_processor_;
    mutable std::shared_mutex mutex_;
};
} // namespace streaming
} // namespace nerve
