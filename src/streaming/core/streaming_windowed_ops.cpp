
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/incremental.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace nerve::streaming
{

namespace
{

Index nextVertexIndex(std::queue<Simplex> window)
{
    Index next = 0;
    while (!window.empty())
    {
        const auto simplex = window.front();
        window.pop();
        for (const auto vertex : simplex.vertices())
        {
            next = std::max(next, static_cast<Index>(vertex + 1));
        }
    }
    return next;
}

} // namespace

WindowedPH::WindowedPH()
    : window_size_(100)
    , overlap_size_(20)
    , update_threshold_(0.1)
    , max_dimension_(3)
    , simplex_window_()
    , incremental_ph_(max_dimension_)
    , current_diagram_()
    , last_stability_(0.0)
{}

WindowedPH::WindowedPH(Size window_size, Size max_dimension)
    : window_size_(window_size)
    , overlap_size_(20)
    , update_threshold_(0.1)
    , max_dimension_(max_dimension)
    , simplex_window_()
    , incremental_ph_(max_dimension_)
    , current_diagram_()
    , last_stability_(0.0)
{}

void WindowedPH::addDataPoint(const std::vector<double> &point)
{
    if (!std::ranges::all_of(point, [](double coordinate) { return std::isfinite(coordinate); }))
    {
        throw std::invalid_argument("streaming data point coordinates must be finite");
    }
    const Index vertex = nextVertexIndex(simplex_window_);
    addSimplexToWindow(Simplex({vertex}));
}

void WindowedPH::addSimplexToWindow(const Simplex &simplex)
{
    simplex_window_.push(simplex);
    incremental_ph_.addSimplex(simplex);
    while (simplex_window_.size() > window_size_)
    {
        removeOldestSimplex();
    }
    updateWindowPersistence();
}

void WindowedPH::slideWindow()
{
    if (window_size_ == 0 || overlap_size_ >= window_size_)
    {
        return;
    }
    const Size removals = window_size_ - overlap_size_;
    for (Size i = 0; i < removals && !simplex_window_.empty(); ++i)
    {
        removeOldestSimplex();
    }
    updateWindowPersistence();
}

Diagram WindowedPH::getWindowPersistence() const
{
    return current_diagram_;
}

std::vector<Pair> WindowedPH::getWindowPairs() const
{
    const auto &pairs = current_diagram_.getPairs();
    return std::vector<Pair>(pairs.begin(), pairs.end());
}

double WindowedPH::getWindowStability() const
{
    return last_stability_;
}

void WindowedPH::setWindowSize(Size size)
{
    window_size_ = std::max<Size>(1, size);
    while (simplex_window_.size() > window_size_)
    {
        removeOldestSimplex();
    }
    updateWindowPersistence();
}

void WindowedPH::setOverlapSize(Size overlap)
{
    overlap_size_ = std::min(overlap, window_size_);
}

void WindowedPH::setUpdateThreshold(double threshold)
{
    update_threshold_ = std::max(0.0, threshold);
}

void WindowedPH::removeOldestSimplex()
{
    if (simplex_window_.empty())
    {
        return;
    }
    const Simplex oldest = simplex_window_.front();
    simplex_window_.pop();
    incremental_ph_.removeSimplex(oldest);
}

void WindowedPH::updateWindowPersistence()
{
    const Diagram next = incremental_ph_.getPersistenceDiagram();
    const double stability = computeStability(next);
    if (stability >= update_threshold_ || current_diagram_.isEmpty())
    {
        current_diagram_ = next;
    }
    last_stability_ = stability;
}

double WindowedPH::computeStability(const Diagram &diagram) const
{
    return detail::diagramSupDistance(current_diagram_, diagram);
}

} // namespace nerve::streaming
