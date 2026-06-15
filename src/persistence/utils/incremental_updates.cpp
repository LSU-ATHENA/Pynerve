#include "nerve/persistence/utils/incremental_updates.hpp"
#include "nerve/persistence/utils/incremental_updates_internal.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <ranges>
#include <stdexcept>
namespace nerve::persistence
{
IncrementalPersistence::IncrementalPersistence(const PersistenceBudget &budget)
    : budget_(budget)
    , current_diagram_(std::make_unique<::nerve::Diagram>())
    , summary_(std::make_unique<::nerve::CompactSummary>())
    , certificate_(StabilityCertificate::createPh5Ph6Certificate(0, 0, budget.memory_limit_mb,
                                                                 budget.time_limit_ms))
    , update_strategy_(IncrementalStrategy::LOCAL_UPDATE)
    , hints_optimization_enabled_(true)
    , summary_refresh_threshold_(0.8)
    , budget_exceeded_(false)
    , streaming_mode_enabled_(false)
    , streaming_window_size_(1000)
    , streaming_stride_(100)
    , total_operations_(0)
    , total_update_time_(0.0)
    , update_times_()
    , point_to_simplices_()
    , streaming_buffer_()
{}
ResultType IncrementalPersistence::incrementalAddPoint(
    const PointCloud &current_points, const std::vector<double> &new_point, size_t max_dimension,
    const std::vector<size_t> &affected_simplices)
{
    IncrementalOperation operation;
    operation.type = IncrementalOperation::Type::ADD_POINT;
    operation.point_index = current_points.size();
    operation.point = new_point;
    return performIncrementalUpdate(current_points, operation, max_dimension, affected_simplices);
}
ResultType
IncrementalPersistence::incrementalRemovePoint(const PointCloud &current_points, size_t point_index,
                                               size_t max_dimension,
                                               const std::vector<size_t> &affected_simplices)
{
    if (point_index >= current_points.size())
    {
        return ResultType::error(ErrorCode::E54_PH4_INVALID_INPUT, "point index out of range");
    }
    IncrementalOperation operation;
    operation.type = IncrementalOperation::Type::REMOVE_POINT;
    operation.point_index = point_index;
    operation.point = current_points[point_index];
    return performIncrementalUpdate(current_points, operation, max_dimension, affected_simplices);
}
ResultType IncrementalPersistence::incrementalAddPoints(
    const PointCloud &current_points, const std::vector<std::vector<double>> &new_points,
    size_t max_dimension, const std::vector<std::vector<size_t>> &batch_hints)
{
    PointCloud working = current_points;
    for (size_t i = 0; i < new_points.size(); ++i)
    {
        const std::vector<size_t> hints =
            i < batch_hints.size() ? batch_hints[i] : std::vector<size_t>{};
        auto result = incrementalAddPoint(working, new_points[i], max_dimension, hints);
        if (!result.ok())
        {
            return result;
        }
        working.push_back(new_points[i]);
    }
    return ResultType::success(*current_diagram_);
}
ResultType IncrementalPersistence::incrementalRemovePoints(
    const PointCloud &current_points, const std::vector<size_t> &point_indices,
    size_t max_dimension, const std::vector<std::vector<size_t>> &batch_hints)
{
    std::vector<size_t> sorted = point_indices;
    std::ranges::sort(sorted, std::greater{});
    PointCloud working = current_points;
    for (size_t i = 0; i < sorted.size(); ++i)
    {
        const std::vector<size_t> hints =
            i < batch_hints.size() ? batch_hints[i] : std::vector<size_t>{};
        auto result = incrementalRemovePoint(working, sorted[i], max_dimension, hints);
        if (!result.ok())
        {
            return result;
        }
        if (sorted[i] < working.size())
        {
            working.erase(working.begin() + static_cast<std::ptrdiff_t>(sorted[i]));
        }
    }
    return ResultType::success(*current_diagram_);
}
ResultType IncrementalPersistence::performIncrementalUpdate(const PointCloud &points,
                                                            const IncrementalOperation &operation,
                                                            size_t max_dimension,
                                                            const std::vector<size_t> &hints)
{
    const auto start = std::chrono::steady_clock::now();
    if (!checkBudget())
    {
        handleBudgetExceeded();
        return ResultType::error(ErrorCode::E41_RESOURCE_LIMIT, "budget exceeded");
    }
    PointCloud updated_points = points;
    if (operation.type == IncrementalOperation::Type::ADD_POINT)
    {
        updated_points.push_back(operation.point);
    }
    else if (operation.type == IncrementalOperation::Type::REMOVE_POINT)
    {
        if (operation.point_index < updated_points.size())
        {
            updated_points.erase(updated_points.begin() +
                                 static_cast<std::ptrdiff_t>(operation.point_index));
        }
    }
    else if (operation.type == IncrementalOperation::Type::ADD_BATCH)
    {
        updated_points.insert(updated_points.end(), operation.batch_points.begin(),
                              operation.batch_points.end());
    }
    else if (operation.type == IncrementalOperation::Type::REMOVE_BATCH)
    {
        std::vector<size_t> erase_indices;
        erase_indices.reserve(operation.batch_points.size());
        for (size_t i = 0; i < operation.batch_points.size(); ++i)
        {
            erase_indices.push_back(operation.point_index + i);
        }
        std::sort(erase_indices.rbegin(), erase_indices.rend());
        for (const size_t idx : erase_indices)
        {
            if (idx < updated_points.size())
            {
                updated_points.erase(updated_points.begin() + static_cast<std::ptrdiff_t>(idx));
            }
        }
    }
    updateSimplicialComplex(updated_points, operation);
    const std::vector<size_t> affected = computeAffectedSimplices(operation, hints);
    auto diagram_result = updatePersistenceDiagram(updated_points, max_dimension, affected);
    if (!diagram_result.ok())
    {
        return diagram_result;
    }
    certificate_ = StabilityCertificate::createPh5Ph6Certificate(
        updated_points.size(), max_dimension, budget_.memory_limit_mb, budget_.time_limit_ms);
    if (shouldRefreshSummary())
    {
        refreshSummary();
    }
    const auto end = std::chrono::steady_clock::now();
    recordUpdateTime(std::chrono::duration<double>(end - start).count());
    updatePerformanceMetrics();
    total_operations_ += 1;
    return ResultType::success(*current_diagram_);
}
void IncrementalPersistence::updateSimplicialComplex(const PointCloud &points,
                                                     const IncrementalOperation &operation)
{
    for (size_t point_index = 0; point_index < points.size(); ++point_index)
    {
        point_to_simplices_.try_emplace(point_index);
    }
    std::vector<size_t> stale_points;
    for (const auto &[point_index, simplices] : point_to_simplices_)
    {
        if (point_index >= points.size())
        {
            stale_points.push_back(point_index);
        }
    }
    for (size_t point_index : stale_points)
    {
        point_to_simplices_.erase(point_index);
    }
    if (operation.type == IncrementalOperation::Type::ADD_POINT)
    {
        point_to_simplices_[operation.point_index] = {};
    }
    else if (operation.type == IncrementalOperation::Type::REMOVE_POINT)
    {
        point_to_simplices_.erase(operation.point_index);
    }
}
std::vector<size_t>
IncrementalPersistence::computeAffectedSimplices(const IncrementalOperation &operation,
                                                 const std::vector<size_t> &hints)
{
    if (hints_optimization_enabled_ && !hints.empty())
    {
        return hints;
    }
    std::vector<size_t> affected;
    if (operation.type == IncrementalOperation::Type::ADD_POINT ||
        operation.type == IncrementalOperation::Type::REMOVE_POINT)
    {
        affected.push_back(operation.point_index);
    }
    else
    {
        affected.resize(operation.batch_points.size());
        std::iota(affected.begin(), affected.end(), operation.point_index);
    }
    return affected;
}
ResultType
IncrementalPersistence::updatePersistenceDiagram(const PointCloud &points, size_t max_dimension,
                                                 const std::vector<size_t> &affected_simplices)
{
    if (!current_diagram_)
    {
        current_diagram_ = std::make_unique<::nerve::Diagram>();
    }
    if (points.empty())
    {
        current_diagram_->clear();
        return ResultType::success(*current_diagram_);
    }
    // Build a closure-complete local complex from the updated point set and run
    // exact Z2 persistence to keep incremental results consistent with core PH.
    auto complex_result = buildIncrementalComplex(points, max_dimension);
    if (!complex_result.ok())
    {
        return ResultType::error(complex_result.errorCode(), complex_result.errorMessage());
    }
    const algebra::SimplicialComplex complex = std::move(complex_result.value());
    const auto exact = computeExactPersistenceZ2(complex, max_dimension);
    overwriteDiagram(exact, current_diagram_.get());
    // Preserve affected simplex bookkeeping for downstream observability hooks.
    for (const size_t simplex_id : affected_simplices)
    {
        point_to_simplices_[simplex_id].push_back(simplex_id);
    }
    return ResultType::success(*current_diagram_);
}
bool IncrementalPersistence::shouldRefreshSummary()
{
    if (update_times_.empty())
    {
        return false;
    }
    const double average = getAverageUpdateTime();
    return average > summary_refresh_threshold_;
}
void IncrementalPersistence::refreshSummary()
{
    if (!summary_)
    {
        summary_ = std::make_unique<::nerve::CompactSummary>();
    }
}
void IncrementalPersistence::processStreamingBuffer()
{
    if (!streaming_mode_enabled_)
    {
        return;
    }
    while (streaming_buffer_.size() > streaming_window_size_)
    {
        streaming_buffer_.erase(streaming_buffer_.begin());
    }
}
void IncrementalPersistence::updateStreamingWindow(const std::vector<double> &new_point)
{
    streaming_buffer_.push_back(new_point);
    processStreamingBuffer();
}
bool IncrementalPersistence::checkBudget()
{
    if (budget_.memory_limit_mb == 0)
    {
        return false;
    }
    if (budget_.time_limit_ms == 0)
    {
        return false;
    }
    return true;
}
void IncrementalPersistence::handleBudgetExceeded()
{
    budget_exceeded_ = true;
    refreshSummary();
}
void IncrementalPersistence::recordUpdateTime(double time)
{
    update_times_.push_back(time);
    total_update_time_ += time;
    if (update_times_.size() > 1024)
    {
        total_update_time_ -= update_times_.front();
        update_times_.erase(update_times_.begin());
    }
}
void IncrementalPersistence::updatePerformanceMetrics()
{
    if (streaming_mode_enabled_ && !streaming_buffer_.empty())
    {
        const double stride = static_cast<double>(std::max<size_t>(1, streaming_stride_));
        summary_refresh_threshold_ = std::max(0.1, summary_refresh_threshold_ / stride);
    }
}
void IncrementalPersistence::enableStreamingMode(bool enable)
{
    streaming_mode_enabled_ = enable;
}
void IncrementalPersistence::setStreamingWindowSize(size_t window_size)
{
    streaming_window_size_ = std::max<size_t>(1, window_size);
}
void IncrementalPersistence::setStreamingStride(size_t stride)
{
    streaming_stride_ = std::max<size_t>(1, stride);
}
void IncrementalPersistence::setBudget(const PersistenceBudget &budget)
{
    budget_ = budget;
}
const PersistenceBudget &IncrementalPersistence::getBudget() const
{
    return budget_;
}
const ::nerve::Diagram &IncrementalPersistence::getDiagram() const
{
    return *current_diagram_;
}
const ::nerve::CompactSummary &IncrementalPersistence::getSummary() const
{
    return *summary_;
}
const StabilityCertificate &IncrementalPersistence::getCertificate() const
{
    return certificate_;
}
bool IncrementalPersistence::hasValidDiagram() const
{
    return current_diagram_ != nullptr && !current_diagram_->empty();
}
bool IncrementalPersistence::budgetExceeded() const
{
    return budget_exceeded_;
}
bool IncrementalPersistence::streamingModeActive() const
{
    return streaming_mode_enabled_;
}
size_t IncrementalPersistence::getTotalOperations() const
{
    return total_operations_;
}
double IncrementalPersistence::getAverageUpdateTime() const
{
    if (update_times_.empty())
    {
        return 0.0;
    }
    return total_update_time_ / static_cast<double>(update_times_.size());
}
void IncrementalPersistence::setUpdateStrategy(IncrementalStrategy strategy)
{
    update_strategy_ = strategy;
}
void IncrementalPersistence::enableHintsOptimization(bool enable)
{
    hints_optimization_enabled_ = enable;
}
void IncrementalPersistence::setSummaryRefreshThreshold(double threshold)
{
    summary_refresh_threshold_ = std::max(0.0, threshold);
}
StreamingPersistence::StreamingPersistence(const PersistenceBudget &budget, size_t window_size,
                                           size_t stride)
    : incremental_(budget)
    , window_size_(std::max<size_t>(1, window_size))
    , stride_(std::max<size_t>(1, stride))
    , sliding_window_enabled_(true)
    , current_window_()
    , points_processed_(0)
    , windows_processed_(0)
    , processing_times_()
{}
ResultType StreamingPersistence::processStreamPoint(const std::vector<double> &point)
{
    const auto start = std::chrono::steady_clock::now();
    updateWindow(point);
    processCurrentWindow();
    const auto end = std::chrono::steady_clock::now();
    processing_times_.push_back(std::chrono::duration<double>(end - start).count());
    points_processed_ += 1;
    return ResultType::success(incremental_.getDiagram());
}
ResultType StreamingPersistence::processStreamBatch(const std::vector<std::vector<double>> &batch)
{
    for (const auto &point : batch)
    {
        auto result = processStreamPoint(point);
        if (!result.ok())
        {
            return result;
        }
    }
    return ResultType::success(incremental_.getDiagram());
}
void StreamingPersistence::setWindowSize(size_t size)
{
    window_size_ = std::max<size_t>(1, size);
    if (current_window_.size() > window_size_)
    {
        current_window_.erase(current_window_.begin(),
                              current_window_.begin() + static_cast<std::ptrdiff_t>(
                                                            current_window_.size() - window_size_));
    }
}
void StreamingPersistence::setStride(size_t stride)
{
    stride_ = std::max<size_t>(1, stride);
}
void StreamingPersistence::enableSlidingWindow(bool enable)
{
    sliding_window_enabled_ = enable;
}
const ::nerve::Diagram &StreamingPersistence::getCurrentDiagram() const
{
    return incremental_.getDiagram();
}
const ::nerve::CompactSummary &StreamingPersistence::getWindowSummary() const
{
    return incremental_.getSummary();
}
size_t StreamingPersistence::getPointsProcessed() const
{
    return points_processed_;
}
size_t StreamingPersistence::getWindowsProcessed() const
{
    return windows_processed_;
}
double StreamingPersistence::getAverageProcessingTime() const
{
    if (processing_times_.empty())
    {
        return 0.0;
    }
    const double total = std::accumulate(processing_times_.begin(), processing_times_.end(), 0.0);
    return total / static_cast<double>(processing_times_.size());
}
void StreamingPersistence::updateWindow(const std::vector<double> &point)
{
    current_window_.push_back(point);
    if (sliding_window_enabled_ && current_window_.size() > window_size_)
    {
        slideWindow();
    }
}
void StreamingPersistence::processCurrentWindow()
{
    if (current_window_.empty())
    {
        return;
    }
    std::vector<size_t> affected;
    if (!current_window_.empty())
    {
        affected.push_back(current_window_.size() - 1);
    }
    auto result =
        incremental_.incrementalAddPoint(current_window_, current_window_.back(), 1, affected);
    if (!result.ok())
    {
        throw std::runtime_error("stream update failed");
    }
    windows_processed_ += 1;
}
void StreamingPersistence::slideWindow()
{
    const size_t erase_count = std::min(stride_, current_window_.size());
    current_window_.erase(current_window_.begin(),
                          current_window_.begin() + static_cast<std::ptrdiff_t>(erase_count));
}
} // namespace nerve::persistence
