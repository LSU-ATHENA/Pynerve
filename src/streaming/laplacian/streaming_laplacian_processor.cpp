#include "nerve/streaming/streaming_laplacian.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nerve::streaming
{

StreamingLaplacianProcessor::StreamingLaplacianProcessor(const IncrementalLaplacianConfig &config)
    : config_(config)
    , laplacian_(config)
    , window_size_(1000)
    , processed_count_(0)
    , edge_threshold_(1.0)
{}

void StreamingLaplacianProcessor::ingestPoints(const std::vector<double> &points, Size dim,
                                               const std::vector<Index> &indices)
{
    if (indices.empty())
        return;
    for (Index idx : indices)
        laplacian_.addVertex(idx);
    for (Size i = 0; i < indices.size(); ++i)
    {
        for (Size j = i + 1; j < indices.size(); ++j)
        {
            double d = pairwiseDistance(&points[indices[i] * dim], &points[indices[j] * dim], dim);
            if (d <= edge_threshold_)
            {
                laplacian_.addEdge(indices[i], indices[j], std::exp(-d * d));
            }
        }
    }
    processed_count_ += indices.size();
}

double StreamingLaplacianProcessor::pairwiseDistance(const double *a, const double *b,
                                                     Size dim) const
{
    double s = 0.0;
    for (Size i = 0; i < dim; ++i)
    {
        double df = a[i] - b[i];
        s += df * df;
    }
    return std::sqrt(s);
}

void StreamingLaplacianProcessor::ingestPointBatch(const std::vector<double> &points, Size dim,
                                                   Size batch_size)
{
    Size total = points.size() / dim;
    for (Size start = 0; start < total; start += batch_size)
    {
        Size end = std::min(start + batch_size, total);
        std::vector<Index> indices(end - start);
        std::iota(indices.begin(), indices.end(), static_cast<Index>(start));
        ingestPoints(points, dim, indices);
    }
}

void StreamingLaplacianProcessor::flushWindow()
{
    auto spectrum = laplacian_.computeSpectrum();
    spectrum_timeline_.push_back(spectrum);
    if (spectrum_timeline_.size() > window_size_)
    {
        spectrum_timeline_.erase(spectrum_timeline_.begin());
    }
}

LaplacianSpectrum StreamingLaplacianProcessor::getLatestSpectrum() const
{
    return laplacian_.computeSpectrum();
}

std::vector<LaplacianSpectrum> StreamingLaplacianProcessor::getSpectrumTimeline() const
{
    return spectrum_timeline_;
}

double StreamingLaplacianProcessor::getStabilityScore() const
{
    if (spectrum_timeline_.size() < 2)
        return 1.0;
    const auto &prev = spectrum_timeline_[spectrum_timeline_.size() - 2];
    const auto &curr = spectrum_timeline_.back();
    Size n = std::min(prev.eigenvalues.size(), curr.eigenvalues.size());
    double diff = 0.0;
    double sum = 0.0;
    for (Size i = 0; i < n; ++i)
    {
        double d = curr.eigenvalues[i] - prev.eigenvalues[i];
        diff += d * d;
        sum += curr.eigenvalues[i] * curr.eigenvalues[i];
    }
    if (sum < config_.eigenvalue_tolerance)
        return 1.0;
    return 1.0 - std::sqrt(diff / sum);
}

Size StreamingLaplacianProcessor::getProcessedCount() const
{
    return processed_count_;
}

void StreamingLaplacianProcessor::setWindowSize(Size window_size)
{
    window_size_ = window_size;
}

void StreamingLaplacianProcessor::setEdgeThreshold(double radius)
{
    edge_threshold_ = radius;
}

} // namespace nerve::streaming
