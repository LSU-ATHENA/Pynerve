#include "nerve/streaming/streaming_tda.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nerve::streaming
{

ApproximateStreamingPH::ApproximateStreamingPH(const StreamingConfig &config)
    : config_(config)
{}

void ApproximateStreamingPH::addDataPoint(const StreamDataPoint &point)
{
    if (!point.isValid())
    {
        return;
    }
    std::lock_guard lock(buffer_mutex_);
    data_buffer_.push(point);
    while (data_buffer_.size() > config_.window_size)
    {
        data_buffer_.pop();
    }
}

void ApproximateStreamingPH::addDataPoints(const std::vector<StreamDataPoint> &points)
{
    for (const auto &point : points)
    {
        addDataPoint(point);
    }
}

StreamingPersistenceDiagram ApproximateStreamingPH::computeCurrentWindow()
{
    updateSlidingWindow();
    if (current_window_.empty())
    {
        StreamingPersistenceDiagram empty_diagram;
        empty_diagram.points = {};
        empty_diagram.window_start_time_ns = 0;
        empty_diagram.window_end_time_ns = 0;
        empty_diagram.window_size = 0;
        empty_diagram.computation_time_ms = 0.0;
        empty_diagram.approximation_error = 0.0;
        empty_diagram.is_approximate = true;
        return empty_diagram;
    }
    return computePersistenceApproximate(current_window_);
}

std::vector<StreamingPersistenceDiagram>
ApproximateStreamingPH::computeRecentWindows(size_t num_windows)
{
    std::vector<StreamingPersistenceDiagram> results;
    results.push_back(computeCurrentWindow());
    return results;
}

void ApproximateStreamingPH::updateSlidingWindow()
{
    std::lock_guard lock(buffer_mutex_);
    current_window_.clear();
    std::queue<StreamDataPoint> copy(data_buffer_);
    while (!copy.empty())
    {
        current_window_.push_back(copy.front());
        copy.pop();
    }
}

StreamingPersistenceDiagram
ApproximateStreamingPH::computePersistenceApproximate(const std::vector<StreamDataPoint> &window)
{
    (void)window;
    StreamingPersistenceDiagram diagram;
    diagram.is_approximate = true;
    diagram.window_size = window.size();
    diagram.points = {};
    return diagram;
}

} // namespace nerve::streaming
